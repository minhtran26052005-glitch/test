"""CPU matrix-fit illustration of LR sensitivity, not evidence of ASR accuracy or its failure causes.

Minimal re-implementation of the ScaledAdam step rule (Adam direction times max(RMS(param), 1e-5); clipping and the
periodic size update are left out), not icefall code. It fits a LoRA update B @ A (B = 0 at start, as in
asr_lora/lora.py) to a low-rank target for 720 illustrative steps (not an ASR epoch count):
  lr 0.0045: rms(B) ~4e-05, ~99% of the wanted update still missing
  lr 0.045:  rms(B) ~6e-04, ~5% missing
Usage: python scripts/scaled_adam_lora_lr_check.py
Use --icefall-dir PATH for the real pinned optimizer instead of the simplified step.
"""
import argparse
import importlib.util
import json
import logging
import math
from pathlib import Path
import torch

def run(lr, steps=720, out=384, inp=384, r=16, alpha=32, seed=0, optimizer_cls=None):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(r, inp, generator=g) * 2 - 1) / math.sqrt(inp)          # kaiming_uniform(a=sqrt(5)) bound
    B = torch.zeros(out, r)
    target = (torch.randn(out, 4, generator=g) @ torch.randn(4, inp, generator=g)) * 1e-3  # ~1e-3 rms delta wanted
    scale = alpha / r
    A.requires_grad_(True); B.requires_grad_(True)
    state = {'A': [torch.zeros_like(A), torch.zeros_like(A)], 'B': [torch.zeros_like(B), torch.zeros_like(B)]}
    optimizer = optimizer_cls([{'named_params': [('A', A), ('B', B)], 'lr': lr}], lr=lr, clipping_scale=2.0) if optimizer_cls else None
    for step in range(1, steps + 1):
        x = torch.randn(64, inp, generator=g)
        loss = ((x @ (scale * B @ A - target).T) ** 2).mean()
        gA, gB = torch.autograd.grad(loss, (A, B))
        if optimizer is not None:
            A.grad, B.grad = gA, gB
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            continue
        with torch.no_grad():
            for name, p, grad in (('A', A, gA), ('B', B, gB)):
                m, v = state[name]
                m.mul_(0.9).add_(grad, alpha=0.1); v.mul_(0.98).add_(grad * grad, alpha=0.02)
                mhat, vhat = m / (1 - 0.9 ** step), v / (1 - 0.98 ** step)
                rms = p.pow(2).mean().sqrt().clamp(min=1e-5)
                p -= lr * rms * mhat / (vhat.sqrt() + 1e-8)
    A, B = A.detach(), B.detach()
    delta = scale * B @ A
    return float(B.pow(2).mean().sqrt()), float((delta - target).pow(2).mean().sqrt() / target.pow(2).mean().sqrt())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--icefall-dir')
    args = parser.parse_args()
    optimizer_cls = None
    if args.icefall_dir:
        path = Path(args.icefall_dir)/'egs/librispeech/ASR/zipformer/optim.py'
        spec = importlib.util.spec_from_file_location('matrix_fit_scaled_adam', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        optimizer_cls = module.ScaledAdam
    logging.basicConfig(level=logging.ERROR)
    result = {'note': 'Matrix-fit experiment, not an ASR accuracy ablation; no Eden schedule.',
              'steps': 720, 'optimizer': 'icefall ScaledAdam' if optimizer_cls else 'simplified', 'results': {}}
    for lr in (0.0045, 0.045):
        b_rms, rel_err = run(lr, optimizer_cls=optimizer_cls)
        result['results'][str(lr)] = {'B_rms': b_rms, 'relative_update_error': rel_err}
    print(json.dumps(result, indent=2))
