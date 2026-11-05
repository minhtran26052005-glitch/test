"""Fold a Qwen3.5 LoRA adapter into the bf16 base and save a drop-in LLM_MODEL_PATH (same parameter count).

  python merge_qwen_lora.py --base models/Qwen3.5-4B --adapter outputs/qwen_lora/adapter --out models/Qwen3.5-4B-lora
  LLM_MODEL_PATH=models/Qwen3.5-4B-lora ./run.sh ...

Merging is done in bf16 on CPU (no quantization reaches the saved weights, also after --load-in-4bit training).
The script checks that the merged model reproduces the adapter model's logits on a short prompt.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', default='models/Qwen3.5-4B')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tolerance', type=float, default=0.05, help='max |logit difference| allowed (bf16 rounding)')
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.base)
    base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map='cpu')
    parameters = sum(p.numel() for p in base.parameters())
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    probe = processor.tokenizer('Bật điều hòa phòng khách 26 độ.', return_tensors='pt')
    with torch.no_grad():
        with_adapter = model(**probe).logits.float()
    merged = model.merge_and_unload().eval()
    with torch.no_grad():
        merged_logits = merged(**probe).logits.float()
    difference = float((with_adapter - merged_logits).abs().max())
    if sum(p.numel() for p in merged.parameters()) != parameters:
        raise RuntimeError('Merged model has a different parameter count')
    if difference > args.tolerance:
        raise RuntimeError(f'Merged logits differ from the adapter model by {difference}')
    out = Path(args.out)
    merged.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    report = {'base': args.base, 'adapter': args.adapter, 'parameters': parameters, 'max_logit_difference': difference}
    (out / 'merge_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
