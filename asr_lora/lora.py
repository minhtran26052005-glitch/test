"""LoRA for icefall Zipformer2 encoders, following icefall's own zipformer_lora recipe (pure PyTorch).

Placement, initialisation and merge semantics are those of egs/librispeech/ASR/zipformer_lora:
  RelPositionMultiheadAttentionWeights.in_proj, SelfAttention.in_proj, FeedforwardModule.in_proj/out_proj;
  A ~ kaiming_uniform(a=sqrt(5)), B = 0 (the model starts exactly at the checkpoint), scaling = alpha / r,
  merge W += scaling * B @ A. Only lora_A/lora_B train; every other module stays in eval mode, as there.

Unlike that recipe, LoRA is injected into the *current* zipformer code that the checkpoints run with, by
wrapping modules instead of redefining the encoder, so the same code serves GipFormer and the hynt model and
the merged model is the original architecture again (same parameter count, standard ONNX export).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# icefall zipformer_lora placement; EXTENDED adds the remaining projections of each encoder layer.
DEFAULT_TARGETS = {
    'RelPositionMultiheadAttentionWeights': ('in_proj',),
    'SelfAttention': ('in_proj',),
    'FeedforwardModule': ('in_proj', 'out_proj'),
}
EXTENDED_TARGETS = {
    **DEFAULT_TARGETS,
    'SelfAttention': ('in_proj', 'out_proj'),
    'NonlinAttention': ('in_proj', 'out_proj'),
    'ConvolutionModule': ('in_proj',),
}
# The transducer joint network (icefall joiner.py); with scope ('encoder.', 'joiner.') it adapts the token output
# mapping too, where rare in-domain words (SJC, ZALO) are scored. Opt-in: it moves every output, so check drift.
JOINER_TARGETS = {'Joiner': ('encoder_proj', 'decoder_proj', 'output_linear')}


def lora_targets(config):
    """(targets, scope) of a saved lora_config, so export re-creates exactly the trained wrapping.
    targets 'joiner' adapts the joint network only (second stage of the real-encoder / TTS-joiner recipe)."""
    if config['targets'] == 'joiner':
        return dict(JOINER_TARGETS), ('joiner.',)
    targets = dict(DEFAULT_TARGETS if config['targets'] == 'icefall' else EXTENDED_TARGETS)
    if config.get('joiner'):
        return {**targets, **JOINER_TARGETS}, ('encoder.', 'joiner.')
    return targets, 'encoder.'


def _swoosh_l(x):
    # SwooshL(x) = log(1 + exp(x - 4)) - 0.08 x - 0.035, as icefall SwooshLForward
    return F.softplus(x - 4.0) - 0.08 * x - 0.035


def _swoosh_r(x):
    # SwooshR(x) = log(1 + exp(x - 1)) - 0.08 x - 0.313261687, as icefall SwooshRForward
    return F.softplus(x - 1.0) - 0.08 * x - 0.313261687


ACTIVATIONS = {'SwooshL': _swoosh_l, 'SwooshR': _swoosh_r}


class _LoRA(nn.Module):
    def __init__(self, base: nn.Module, in_features: int, out_features: int, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError('LoRA rank must be positive')
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # Code that reads module.weight / module.bias directly keeps working.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def delta_weight(self):
        return (self.lora_B @ self.lora_A) * self.scaling

    def low_rank(self, x):
        return (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

    def merged(self) -> nn.Module:
        """The base module with the update folded into its weight (original module type, no extra params)."""
        with torch.no_grad():
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
        return self.base


class LoRALinear(_LoRA):
    """Wraps nn.Linear (icefall ScaledLinear returns nn.Linear)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__(base, base.in_features, base.out_features, r, alpha, dropout)

    def forward(self, x):
        return self.base(x) + self.low_rank(x)


class LoRAActivationLinear(_LoRA):
    """Wraps icefall scaling.ActivationDropoutAndLinear: y = W act(x) + b (dropout only in train mode)."""

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0):
        out_features, in_features = base.weight.shape
        super().__init__(base, in_features, out_features, r, alpha, dropout)
        if base.activation not in ACTIVATIONS:
            raise ValueError(f'Unsupported activation {base.activation}')
        self.activation = ACTIVATIONS[base.activation]

    def forward(self, x):
        return self.base(x) + self.low_rank(self.activation(x))


def _wrap(module: nn.Module, r: int, alpha: float, dropout: float) -> nn.Module:
    if isinstance(module, nn.Linear):
        return LoRALinear(module, r, alpha, dropout)
    if type(module).__name__ == 'ActivationDropoutAndLinear':
        return LoRAActivationLinear(module, r, alpha, dropout)
    raise TypeError(f'Cannot add LoRA to {type(module).__name__}')


def add_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.0, targets=None, scope='encoder.'):
    """Freeze the whole model and wrap the target projections under `scope` (a prefix or a tuple of prefixes).
    Returns the wrapped names."""
    targets = DEFAULT_TARGETS if targets is None else targets
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        if not (name + '.').startswith(scope) or type(module).__name__ not in targets:
            continue
        for child in targets[type(module).__name__]:
            original = getattr(module, child, None)
            if original is None or isinstance(original, _LoRA):
                continue
            setattr(module, child, _wrap(original, r, alpha, dropout))
            wrapped.append(f'{name}.{child}')
    if not wrapped:
        raise RuntimeError(f'No LoRA target found under {scope!r}; is this an icefall Zipformer2 model?')
    return wrapped


def lora_modules(model: nn.Module):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, _LoRA)]


def set_lora_train_mode(model: nn.Module):
    """icefall zipformer_lora convention: the frozen network runs in eval mode, only LoRA dropout trains."""
    model.eval()
    for _, module in lora_modules(model):
        module.lora_dropout.train()


def lora_state_dict(model: nn.Module):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if '.lora_A' in name or '.lora_B' in name}


def load_lora_state_dict(model: nn.Module, state):
    expected = set(lora_state_dict(model))
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise ValueError(f'LoRA adapter does not match the model: missing {missing[:3]}, unexpected {extra[:3]}')
    model.load_state_dict(state, strict=False)


def scale_lora(model: nn.Module, scale: float) -> int:
    """Multiply every adapter's update by `scale`: 0 is the original checkpoint, 1 the fine-tuned model.

    Weight-space interpolation between the two (WiSE-FT): with a small in-domain set the update overshoots on
    what the domain data does not cover, and a scale picked on validation (e.g. 0.5-0.8) keeps most of the
    domain gain with less drift. Applies to the forward pass and to merge_lora. Returns how many were scaled."""
    if not 0.0 <= scale <= 2.0:
        raise ValueError('LoRA scale must be in [0, 2]')
    modules = lora_modules(model)
    for _, module in modules:
        module.scaling = module.alpha / module.r * scale
    return len(modules)


def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and restore the original modules; returns how many were merged."""
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LoRA):
                setattr(module, child_name, child.merged())
                merged += 1
    return merged


def trainable_summary(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_parameters': total, 'trainable_parameters': trainable, 'trainable_percent': round(100 * trainable / total, 3)}
