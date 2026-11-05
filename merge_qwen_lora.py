"""Fold a Qwen3.5 LoRA adapter into the bf16 base and save a drop-in LLM_MODEL_PATH (same parameter count).

  python merge_qwen_lora.py --base models/Qwen3.5-4B --adapter outputs/qwen_lora/adapter --out models/Qwen3.5-4B-lora
  LLM_MODEL_PATH=models/Qwen3.5-4B-lora ./run.sh ...

Merging is done in bf16 on CPU (no quantization reaches the saved weights, also after --load-in-4bit training).
The script checks that the merged model reproduces the adapter model's logits on a short prompt.

--scale s merges s times the adapter update (WiSE-FT interpolation, 0 = base, 1 = trained adapter). With a small
domain set, merge e.g. 1.0 and 0.7, compare them on the validation split of the agent benchmark, then test once.
Plain LoRA only: under DoRA the trained magnitude vector makes scale 0 differ from the base model.
"""
import argparse
import json
from pathlib import Path


def scale_adapter(model, scale):
    """Multiply every LoRA layer's scaling (forward pass and merge) by `scale`; returns how many layers."""
    if not 0.0 <= scale <= 2.0:
        raise ValueError('--scale must be in [0, 2]')
    if scale == 1.0:
        return 0
    if any(getattr(config, 'use_dora', False) for config in model.peft_config.values()):
        raise ValueError('--scale is only an interpolation for plain LoRA, not DoRA')
    layers = 0
    for module in model.modules():
        scaling = getattr(module, 'scaling', None)
        if isinstance(scaling, dict) and hasattr(module, 'lora_A'):
            for name in scaling:
                scaling[name] *= scale
            layers += 1
    if not layers:
        raise RuntimeError('No LoRA layer found to scale')
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', default='models/Qwen3.5-4B')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scale', type=float, default=1.0, help='fraction of the adapter update to merge (0 = base, 1 = trained)')
    parser.add_argument('--tolerance', type=float, default=0.05, help='max |logit difference| allowed (bf16 rounding)')
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from train_qwen_lora import configure_generation
    processor = AutoProcessor.from_pretrained(args.base)
    base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map='cpu')
    stop_ids = configure_generation(base, processor.tokenizer)
    parameters = sum(p.numel() for p in base.parameters())
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    scaled = scale_adapter(model, args.scale)
    probe = processor.tokenizer('Bật điều hòa phòng khách 26 độ.', return_tensors='pt')
    with torch.no_grad():
        with_adapter = model(**probe).logits.float()
    merged = model.merge_and_unload().eval()
    with torch.no_grad():
        merged_logits = merged(**probe).logits.float()
    if not torch.isfinite(with_adapter).all() or not torch.isfinite(merged_logits).all():
        raise RuntimeError('Non-finite logits before/after merging the Qwen adapter')
    difference = float((with_adapter - merged_logits).abs().max())
    if sum(p.numel() for p in merged.parameters()) != parameters:
        raise RuntimeError('Merged model has a different parameter count')
    if difference > args.tolerance:
        raise RuntimeError(f'Merged logits differ from the adapter model by {difference}')
    out = Path(args.out)
    merged.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    report = {'base': args.base, 'adapter': args.adapter, 'scale': args.scale, 'scaled_lora_layers': scaled,
              'parameters': parameters, 'max_logit_difference': difference, 'generation_stop_ids': stop_ids}
    (out / 'merge_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
