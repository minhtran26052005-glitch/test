"""Why train_asr_lora.py uses base lr 0.045 for ScaledAdam: a check that needs only torch (CPU, a few seconds).

Minimal re-implementation of the ScaledAdam step rule (Adam direction times max(RMS(param), 1e-5); clipping and the
periodic size update are left out), not icefall code. It fits a LoRA update B @ A (B = 0 at start, as in
asr_lora/lora.py) to a low-rank target for 720 steps, about 15 epochs of the fit split at --max-duration 300:
  lr 0.0045: rms(B) ~4e-05, ~99% of the wanted update still missing
  lr 0.045:  rms(B) ~6e-04, ~5% missing
Usage: python scripts/scaled_adam_lora_lr_check.py
"""
import math
import torch

def run(lr, steps=720, out=384, inp=384, r=16, alpha=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(r, inp, generator=g) * 2 - 1) / math.sqrt(inp)          # kaiming_uniform(a=sqrt(5)) bound
    B = torch.zeros(out, r)
    target = (torch.randn(out, 4, generator=g) @ torch.randn(4, inp, generator=g)) * 1e-3  # ~1e-3 rms delta wanted
    scale = alpha / r
    A.requires_grad_(True); B.requires_grad_(True)
    state = {'A': [torch.zeros_like(A), torch.zeros_like(A)], 'B': [torch.zeros_like(B), torch.zeros_like(B)]}
    for step in range(1, steps + 1):
        x = torch.randn(64, inp, generator=g)
        loss = ((x @ (scale * B @ A - target).T) ** 2).mean()
        gA, gB = torch.autograd.grad(loss, (A, B))
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

for lr in (0.0045, 0.045):
    b_rms, rel_err = run(lr)
    print(f'lr {lr}: rms(B) {b_rms:.2e}, remaining error of the wanted update {100 * rel_err:.0f}%')
