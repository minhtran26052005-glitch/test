"""Load the two Zipformer checkpoints as trainable icefall models.

GipFormer 1.5 (g-group-ai-lab/gipformer1.5-68M-rnnt): icefall Zipformer2 + pruned RNN-T, released as model.pt
(state dict) and built with the default icefall zipformer arguments, exactly as its own infer_pytorch.py does.

ZipFormer-150M (hynt/ZipFormer-150M-CR-CTC-RNNT-6000h): icefall Zipformer2 trained with CR-CTC + CTC + RNN-T,
released only as jit_script.pt (icefall export.py --jit 1) and ONNX. The architecture is recovered from the
TorchScript module (its attributes, else weight shapes), the weights are loaded strictly, and the rebuilt
encoder/decoder are checked against the TorchScript ones on random input before anything is trained.
"""
import argparse
import re
import sys
from pathlib import Path

ICEFALL_REPO = 'https://github.com/k2-fsa/icefall.git'
ICEFALL_COMMIT = '9c966762e058bd7ba79d9da71e42fc06b64b703e'  # 2026-10-28, the code this package was written against
LAYER_KEY = re.compile(r'^encoder\.encoders\.(\d+)\.(?:encoder\.)?layers\.(\d+)\.')


def setup_icefall(icefall_dir):
    """Put icefall and its zipformer recipe on sys.path; returns the recipe's train module."""
    root = Path(icefall_dir).expanduser().resolve()
    recipe = root / 'egs' / 'librispeech' / 'ASR' / 'zipformer'
    if not (recipe / 'train.py').is_file():
        raise RuntimeError(
            f'Không thấy icefall tại {root}. Clone đúng commit:\n'
            f'  git clone {ICEFALL_REPO} {root} && git -C {root} checkout {ICEFALL_COMMIT}\n'
            'và cài k2, lhotse theo https://k2-fsa.github.io/icefall/installation/')
    for path in (str(root), str(recipe)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import train  # noqa: E402  (icefall zipformer/train.py)
    return train


def read_tokens(path):
    """(vocab_size, blank_id) as icefall computes them from tokens.txt."""
    symbols = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            symbol, index = line.rsplit(' ', 1)
            symbols[symbol] = int(index)
    return len([s for s in symbols if not s.startswith('#')]), symbols['<blk>']


def default_params(train, **overrides):
    params = train.get_params()
    parser = argparse.ArgumentParser()
    train.add_model_arguments(parser)
    params.update(vars(parser.parse_args([])))
    params.update(overrides)
    return params


def heads_from_shapes(state, prefix, value_head_dim, query_head_dim, pos_head_dim):
    """Number of heads of one stack from its first layer, given the head sizes."""
    heads = state[prefix + 'self_attn1.in_proj.weight'].shape[0] // value_head_dim
    expected = (2 * query_head_dim + pos_head_dim) * heads
    actual = state[prefix + 'self_attn_weights.in_proj.weight'].shape[0]
    if actual != expected:
        raise ValueError(f'Head sizes do not fit {prefix}: in_proj has {actual} rows, expected {expected}')
    return heads


def _conv_kernel(state, prefix):
    if prefix + 'weight' in state:
        return state[prefix + 'weight'].shape[-1]  # non-causal: nn.Conv1d
    return state[prefix + 'chunkwise_conv.weight'].shape[-1]  # causal: ChunkCausalDepthwiseConv1d


def config_from_state_dict(state, attributes=None, defaults=None):
    """Zipformer2/decoder/joiner arguments (icefall CLI names) from weights; TorchScript attributes win when present."""
    attributes = attributes or {}
    defaults = defaults or {}
    first = {}
    layers = {}
    for key in state:
        match = LAYER_KEY.match(key)
        if match:
            stack, layer = map(int, match.groups())
            layers.setdefault(stack, set()).add(layer)
            first.setdefault(stack, key[:match.end()].rsplit('layers.', 1)[0] + 'layers.0.')
    stacks = sorted(layers)
    if not stacks:
        raise ValueError('No Zipformer2 encoder layers in this state dict')
    def stack_value(fn):
        return ','.join(str(fn(stack, first[stack])) for stack in stacks)
    downsample = [state[f'encoder.encoders.{s}.downsample.bias'].shape[0] if f'encoder.encoders.{s}.downsample.bias' in state else 1
                  for s in stacks]
    value_head = str(attributes.get('value_head_dim', defaults.get('value_head_dim', 12)))
    query_head = str(attributes.get('query_head_dim', defaults.get('query_head_dim', 32)))
    value_head_dim, query_head_dim = int(value_head.split(',')[0]), int(query_head.split(',')[0])
    pos_head_dim_guess = int(str(defaults.get('pos_head_dim', 4)).split(',')[0])
    if 'num_heads' in attributes:
        heads = [int(h) for h in str(attributes['num_heads']).split(',')]
    else:
        heads = [heads_from_shapes(state, first[s], value_head_dim, query_head_dim, pos_head_dim_guess) for s in stacks]
    linear_pos = [state[first[s] + 'self_attn_weights.linear_pos.weight'] for s in stacks]
    config = {
        'num_encoder_layers': ','.join(str(len(layers[s])) for s in stacks),
        'downsampling_factor': ','.join(map(str, downsample)),
        'encoder_dim': stack_value(lambda s, p: state[p + 'feed_forward1.in_proj.weight'].shape[1]),
        'feedforward_dim': stack_value(lambda s, p: state[p + 'feed_forward2.in_proj.weight'].shape[0]),
        'num_heads': ','.join(map(str, heads)),
        'query_head_dim': query_head,
        'value_head_dim': value_head,
        'pos_head_dim': ','.join(str(w.shape[0] // h) for w, h in zip(linear_pos, heads)),
        'pos_dim': int(linear_pos[0].shape[1]),
        'cnn_module_kernel': stack_value(lambda s, p: _conv_kernel(state, p + 'conv_module1.depthwise_conv.')),
    }
    if 'encoder_unmasked_dim' in attributes:
        config['encoder_unmasked_dim'] = attributes['encoder_unmasked_dim']
    else:
        # Only used for feature masking in training mode, which LoRA fine-tuning never enables (frozen base in eval).
        dims = [int(d) for d in config['encoder_dim'].split(',')]
        config['encoder_unmasked_dim'] = ','.join(str(min(d, min(dims))) for d in dims)
    for name in ('causal',):
        if name in attributes:
            config[name] = bool(attributes[name])
    if 'decoder.embedding.weight' in state:
        config['decoder_dim'] = int(state['decoder.embedding.weight'].shape[1])
        config['context_size'] = int(state['decoder.conv.weight'].shape[-1]) if 'decoder.conv.weight' in state else 1
        config['joiner_dim'] = int(state['joiner.output_linear.weight'].shape[1])
    config['use_transducer'] = 'joiner.output_linear.weight' in state
    config['use_ctc'] = any(key.startswith('ctc_output.') for key in state)
    config['use_attention_decoder'] = any(key.startswith('attention_decoder.') for key in state)
    return config


def _as_cli(value):
    if isinstance(value, (tuple, list)):
        return ','.join(str(int(v)) for v in value)
    return value


def jit_attributes(jit_model):
    """Zipformer2 constructor values kept as TorchScript attributes (tuples become 'a,b,c' like the CLI)."""
    encoder = jit_model.encoder.encoder
    found = {}
    for name in ('downsampling_factor', 'encoder_dim', 'encoder_unmasked_dim', 'num_encoder_layers', 'query_head_dim',
                 'value_head_dim', 'num_heads', 'cnn_module_kernel', 'causal'):
        try:
            found[name] = _as_cli(getattr(encoder, name))
        except (AttributeError, RuntimeError):
            pass
    return found


def jit_to_eager_keys(state):
    """icefall export.py wraps encoder + encoder_embed in EncoderModel before scripting."""
    renamed = {}
    for key, value in state.items():
        if key.startswith('encoder.encoder_embed.'):
            key = key[len('encoder.'):]
        elif key.startswith('encoder.encoder.'):
            key = key[len('encoder.'):]
        renamed[key] = value
    return renamed


def build(train, config, vocab_size, blank_id):
    params = default_params(train, vocab_size=vocab_size, blank_id=blank_id, **config)
    params.sos_id = params.eos_id = 1
    model = train.get_model(params)
    return model, params


def load_weights(model, state):
    """Every model weight must come from the checkpoint; extra checkpoint keys are only reported."""
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(f'Checkpoint lacks {len(result.missing_keys)} weights of the rebuilt model, e.g. {result.missing_keys[:5]}; '
                           'the architecture differs: pass --model-config')
    return sorted(result.unexpected_keys)


def load_gipformer(train, model_dir, override=None):
    import torch
    model_dir = Path(model_dir)
    vocab_size, blank_id = read_tokens(model_dir / 'tokens.txt')
    checkpoint = torch.load(model_dir / 'model.pt', map_location='cpu', weights_only=False)
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    defaults = default_params(train)
    # GipFormer's own loader uses the default icefall arguments; the weights only confirm/adjust the sizes.
    config = config_from_state_dict(state, defaults=defaults)
    config['encoder_unmasked_dim'] = defaults['encoder_unmasked_dim']
    config.update(override or {})
    model, params = build(train, config, vocab_size, blank_id)
    unexpected = load_weights(model, state)
    return model.eval(), params, {'checkpoint': str(model_dir / 'model.pt'), 'config': config, 'unexpected_keys': unexpected}


def load_hynt(train, model_dir, override=None, check=True, tolerance=1e-3):
    import torch
    model_dir = Path(model_dir)
    vocab_size, blank_id = read_tokens(model_dir / 'tokens.txt')
    jit_model = torch.jit.load(str(model_dir / 'jit_script.pt'), map_location='cpu').eval()
    state = jit_to_eager_keys(jit_model.state_dict())
    attributes = jit_attributes(jit_model)
    config = config_from_state_dict(state, attributes, default_params(train))
    config.update(override or {})
    model, params = build(train, config, vocab_size, blank_id)
    unexpected = load_weights(model, state)
    model.eval()
    report = {'checkpoint': str(model_dir / 'jit_script.pt'), 'config': config, 'jit_attributes': attributes, 'unexpected_keys': unexpected}
    if check:
        report['check'] = compare_with_jit(model, jit_model, tolerance)
    return model, params, report


def load_model(train, kind, model_dir, override=None):
    """kind: 'gipformer' (model.pt, RNN-T) or 'zipformer' (hynt jit_script.pt, CR-CTC + CTC + RNN-T)."""
    if kind == 'gipformer':
        return load_gipformer(train, model_dir, override)
    if kind == 'zipformer':
        return load_hynt(train, model_dir, override)
    raise ValueError(f'Unknown model kind {kind}')


def compare_with_jit(model, jit_model, tolerance=1e-3, seed=0):
    """Rebuilt eager model must reproduce the TorchScript encoder and decoder before we train on it."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(2, 250, 80, generator=generator)
    lengths = torch.tensor([250, 180])
    with torch.no_grad():
        expected, expected_lens = jit_model.encoder(features, lengths)
        actual, actual_lens = model.forward_encoder(features, lengths)
        result = {'encoder_max_abs_diff': float((expected - actual).abs().max()),
                  'encoder_scale': float(expected.abs().max())}
        if not torch.equal(expected_lens, actual_lens):
            raise RuntimeError('Rebuilt encoder gives different output lengths than jit_script.pt')
        if hasattr(model, 'decoder') and model.decoder is not None:
            context = torch.zeros(2, model.decoder.context_size, dtype=torch.int64)
            context[1, -1] = 5
            result['decoder_max_abs_diff'] = float((jit_model.decoder(context, False) - model.decoder(context, need_pad=False)).abs().max())
    worst = max(result['encoder_max_abs_diff'] / max(result['encoder_scale'], 1.0), result.get('decoder_max_abs_diff', 0.0))
    if worst > tolerance:
        raise RuntimeError(f'Rebuilt model does not match jit_script.pt ({result}); pass --model-config to fix the architecture')
    return result
