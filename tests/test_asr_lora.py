"""LoRA injection/merge on stand-ins that mirror icefall Zipformer2 module names and attributes.

The real icefall classes are exercised by train_asr_lora.py / export_asr_lora.py on the training machine, which
also check the rebuilt hynt model against jit_script.pt and the merged ONNX against PyTorch.
"""
import unittest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # CI installs no torch; these tests run where training happens
    torch = None


def icefall_swoosh_l(x):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float('inf'), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


if torch is not None:
    class ActivationDropoutAndLinear(nn.Module):  # same attributes and eval path as icefall scaling.py
        def __init__(self, i, o):
            super().__init__()
            l = nn.Linear(i, o)
            self.weight = l.weight
            self.register_parameter('bias', l.bias)
            self.activation, self.dropout_p, self.dropout_shared_dim = 'SwooshL', 0.1, -1
        def forward(self, x):
            return F.linear(icefall_swoosh_l(x), self.weight, self.bias)

    class FeedforwardModule(nn.Module):
        def __init__(self, d, f):
            super().__init__()
            self.in_proj, self.out_proj = nn.Linear(d, f), ActivationDropoutAndLinear(f, d)
        def forward(self, x):
            return self.out_proj(self.in_proj(x))

    class SelfAttention(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.in_proj, self.out_proj = nn.Linear(d, d), nn.Linear(d, d)
        def forward(self, x):
            return self.out_proj(self.in_proj(x))

    class RelPositionMultiheadAttentionWeights(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.in_proj = nn.Linear(d, 2 * d)
        def forward(self, x):
            return self.in_proj(x)[..., : x.shape[-1]]

    class Layer(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.self_attn_weights = RelPositionMultiheadAttentionWeights(d)
            self.self_attn1 = SelfAttention(d)
            self.feed_forward1 = FeedforwardModule(d, 3 * d)
            self.dropout = nn.Dropout(0.3)
        def forward(self, x):
            return x + self.feed_forward1(self.self_attn1(self.self_attn_weights(self.dropout(x))))

    class Toy(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.encoder = nn.Sequential(Layer(d), Layer(d))
            self.joiner = SelfAttention(d)  # outside the encoder scope: must stay untouched
        def forward(self, x):
            return self.joiner(self.encoder(x))


@unittest.skipIf(torch is None, 'torch not installed')
class LoraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Toy().eval()
        self.x = torch.randn(3, 5, 8)
        self.base_out = self.model(self.x).detach()
        self.base_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def test_wraps_icefall_targets_only_and_freezes_base(self):
        from asr_lora.lora import LoRAActivationLinear, LoRALinear, add_lora, trainable_summary
        wrapped = add_lora(self.model, r=4, alpha=8)
        self.assertEqual(len(wrapped), 8)  # per layer: attn weights in_proj, self_attn in_proj, ff in_proj + out_proj
        self.assertIsInstance(self.model.encoder[0].feed_forward1.out_proj, LoRAActivationLinear)
        self.assertIsInstance(self.model.encoder[0].self_attn1.in_proj, LoRALinear)
        self.assertIsInstance(self.model.encoder[0].self_attn1.out_proj, nn.Linear)  # not in icefall's placement
        self.assertIsInstance(self.model.joiner.in_proj, nn.Linear)
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.assertTrue(names and all('.lora_A' in n or '.lora_B' in n for n in names))
        self.assertLess(trainable_summary(self.model)['trainable_percent'], 100)
        torch.testing.assert_close(self.model(self.x), self.base_out)  # B = 0: starts exactly at the checkpoint

    def test_merge_is_exact_and_restores_architecture(self):
        from asr_lora.lora import add_lora, lora_modules, merge_lora
        parameters = sum(p.numel() for p in self.model.parameters())
        add_lora(self.model, r=4, alpha=8)
        for _, module in lora_modules(self.model):
            nn.init.normal_(module.lora_B, std=0.2)
        with_adapter = self.model(self.x).detach()
        self.assertGreater(float((with_adapter - self.base_out).abs().max()), 1e-3)
        self.assertEqual(merge_lora(self.model), 8)
        self.assertEqual(lora_modules(self.model), [])
        self.assertEqual(type(self.model.encoder[1].feed_forward1.out_proj).__name__, 'ActivationDropoutAndLinear')
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), parameters)
        torch.testing.assert_close(self.model(self.x), with_adapter, rtol=1e-5, atol=1e-5)

    def test_activation_matches_icefall_formula(self):
        from asr_lora.lora import ACTIVATIONS
        x = torch.linspace(-30, 30, 101)
        torch.testing.assert_close(ACTIVATIONS['SwooshL'](x), icefall_swoosh_l(x))

    def test_train_mode_and_state_dict_roundtrip(self):
        from asr_lora.lora import add_lora, load_lora_state_dict, lora_modules, lora_state_dict, set_lora_train_mode
        add_lora(self.model, r=2, alpha=4, dropout=0.1)
        set_lora_train_mode(self.model)
        self.assertFalse(self.model.encoder[0].dropout.training)  # frozen network stays in eval, as icefall
        self.assertTrue(all(m.lora_dropout.training for _, m in lora_modules(self.model)))
        for _, module in lora_modules(self.model):
            nn.init.normal_(module.lora_B)
        state = lora_state_dict(self.model)
        fresh = Toy()
        fresh.load_state_dict(self.base_state)  # wrapped modules rename base keys (.base.weight): load before wrapping
        add_lora(fresh, r=2, alpha=4)
        load_lora_state_dict(fresh, state)
        self.model.eval(), fresh.eval()
        torch.testing.assert_close(fresh(self.x), self.model(self.x))
        with self.assertRaises(ValueError):
            load_lora_state_dict(fresh, {k: v for k, v in list(state.items())[1:]})


def synthetic_zipformer_state(dims=(16, 24), layers=(2, 1), heads=(2, 4), ff=(32, 48), downsample=(1, 2), kernel=(7, 5),
                              qhd=6, vhd=3, phd=2, pos_dim=10, vocab=50, decoder_dim=12, joiner_dim=14, context=2, ctc=True):
    """Weights with icefall key names/shapes for Zipformer2 + decoder + joiner (+ ctc_output)."""
    state = {}
    for s, (d, n, h, f, ds, k) in enumerate(zip(dims, layers, heads, ff, downsample, kernel)):
        base = f'encoder.encoders.{s}.' + ('encoder.' if ds > 1 else '')
        if ds > 1:
            state[f'encoder.encoders.{s}.downsample.bias'] = torch.zeros(ds)
        for j in range(n):
            p = f'{base}layers.{j}.'
            state[p + 'self_attn_weights.in_proj.weight'] = torch.zeros((2 * qhd + phd) * h, d)
            state[p + 'self_attn_weights.linear_pos.weight'] = torch.zeros(phd * h, pos_dim)
            state[p + 'self_attn1.in_proj.weight'] = torch.zeros(h * vhd, d)
            state[p + 'feed_forward1.in_proj.weight'] = torch.zeros(f * 3 // 4, d)
            state[p + 'feed_forward2.in_proj.weight'] = torch.zeros(f, d)
            state[p + 'conv_module1.depthwise_conv.weight'] = torch.zeros(d, 1, k)
    state['decoder.embedding.weight'] = torch.zeros(vocab, decoder_dim)
    state['decoder.conv.weight'] = torch.zeros(decoder_dim, 4, context)
    state['joiner.output_linear.weight'] = torch.zeros(vocab, joiner_dim)
    if ctc:
        state['ctc_output.1.weight'] = torch.zeros(vocab, max(dims))
    return state


@unittest.skipIf(torch is None, 'torch not installed')
class ConfigRecoveryTests(unittest.TestCase):
    def test_from_shapes_with_head_sizes(self):
        from asr_lora.models import config_from_state_dict
        config = config_from_state_dict(synthetic_zipformer_state(), defaults={'query_head_dim': '6', 'value_head_dim': '3', 'pos_head_dim': '2'})
        self.assertEqual(config['num_encoder_layers'], '2,1')
        self.assertEqual(config['downsampling_factor'], '1,2')
        self.assertEqual(config['encoder_dim'], '16,24')
        self.assertEqual(config['feedforward_dim'], '32,48')
        self.assertEqual(config['num_heads'], '2,4')
        self.assertEqual(config['pos_head_dim'], '2,2')
        self.assertEqual(config['pos_dim'], 10)
        self.assertEqual(config['cnn_module_kernel'], '7,5')
        self.assertEqual((config['decoder_dim'], config['context_size'], config['joiner_dim']), (12, 2, 14))
        self.assertEqual((config['use_transducer'], config['use_ctc'], config['use_attention_decoder']), (True, True, False))

    def test_jit_attributes_take_precedence_and_wrong_heads_fail(self):
        from asr_lora.models import config_from_state_dict
        state = synthetic_zipformer_state()
        config = config_from_state_dict(state, {'num_heads': '2,4', 'query_head_dim': '6', 'value_head_dim': '3', 'encoder_unmasked_dim': '16,16'})
        self.assertEqual(config['encoder_unmasked_dim'], '16,16')
        with self.assertRaises(ValueError):
            config_from_state_dict(state, defaults={'query_head_dim': '32', 'value_head_dim': '3', 'pos_head_dim': '4'})

    def test_jit_keys_and_tokens(self):
        import tempfile
        from pathlib import Path
        from asr_lora.models import jit_to_eager_keys, read_tokens
        renamed = jit_to_eager_keys({'encoder.encoder.encoders.0.x': 1, 'encoder.encoder_embed.conv.0.weight': 2, 'decoder.embedding.weight': 3})
        self.assertEqual(set(renamed), {'encoder.encoders.0.x', 'encoder_embed.conv.0.weight', 'decoder.embedding.weight'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tokens.txt'
            path.write_text('<blk> 0\n<sos/eos> 1\n<unk> 2\n▁HAI 3\n#0 4\n', encoding='utf-8')
            self.assertEqual(read_tokens(path), (4, 0))


@unittest.skipIf(torch is None, 'torch not installed')
class LossTests(unittest.TestCase):
    def test_icefall_loss_weighting_and_empty_terms(self):
        from types import SimpleNamespace
        from train_asr_lora import combine_losses, get_parser, loss_scales, word_errors
        args = get_parser().parse_args(['--model', 'zipformer', '--model-dir', 'm', '--train-manifest', 't', '--valid-manifest', 'v', '--exp-dir', 'e'])
        scales = loss_scales(args, 1)
        self.assertEqual((scales['simple'], scales['pruned'], scales['ctc'], scales['cr']), (0.5, 1.0, 0.2, 0.2))
        args.warm_step = 100
        self.assertEqual(loss_scales(args, 0)['cr'], 0.0)
        params = SimpleNamespace(use_transducer=True, use_ctc=True, use_cr_ctc=True, use_attention_decoder=False)
        t = lambda v: torch.tensor(float(v))
        self.assertAlmostEqual(float(combine_losses((t(2), t(3), t(10), torch.empty(0), t(5)), params, scales)), 0.5 * 2 + 3 + 0.2 * 10 + 0.2 * 5)
        validation = combine_losses((t(2), t(3), t(10), torch.empty(0), torch.empty(0)), params, scales)
        self.assertEqual(validation.shape, torch.Size([]))  # empty CR term during validation must not empty the sum
        self.assertEqual(word_errors('A B C', 'A X C D'), (2, 3))


try:
    import lhotse  # noqa: F401
    import soundfile
except ImportError:
    lhotse = None


@unittest.skipIf(torch is None or lhotse is None, 'lhotse/torch not installed')
class DataTests(unittest.TestCase):
    def test_manifest_to_icefall_batch(self):
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        from asr_lora.data import dataloader, read_manifests, to_cuts
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for i, seconds in enumerate((1.2, 2.0, 0.9)):
                path = Path(tmp) / f'a{i}.wav'
                soundfile.write(str(path), (0.1 * np.random.randn(int(16000 * seconds))).astype('float32'), 16000)
                records.append({'id': f'u{i}:last', 'audio': path.name, 'text': 'BẬT ĐÈN PHÒNG KHÁCH'})
            manifest = Path(tmp) / 'm.jsonl'
            manifest.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records), encoding='utf-8')
            cuts = to_cuts(read_manifests([manifest], tmp))
            for train, spec in ((True, True), (False, False)):
                batch = next(iter(dataloader(cuts, max_duration=100, train=train, spec_augment_in_dataset=spec, num_workers=0)))
                self.assertEqual(batch['inputs'].shape[-1], 80)
                sup = batch['supervisions']
                self.assertEqual(len(sup['text']), 3)
                for key in ('sequence_idx', 'start_frame', 'num_frames'):  # needed for CR-CTC time warping
                    self.assertIn(key, sup)
                self.assertEqual(int(sup['num_frames'].max()), batch['inputs'].shape[1])


if __name__ == '__main__':
    unittest.main()
