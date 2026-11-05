"""WER of one or more ASR model directories on a manifest, with the pipeline's own sherpa-onnx decoding.

Use the held-out test split to compare a base model with its LoRA export (same decoding as run.sh):
  python eval_asr_onnx.py --manifest outputs/asr_finetune/test.jsonl \
      --model gipformer=models/gipformer1.5-68M-rnnt --model gipformer=models/gipformer1.5-68M-rnnt-lora
Writes per-utterance hypotheses next to --out and prints WER per model directory.
"""
import argparse
import json
from pathlib import Path

from asr_lora.data import read_manifests
from asr_spoken_text import spoken_tokens


def words(text):
    return ' '.join(spoken_tokens(text)).upper().split()


def edit_distance(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', action='append', required=True)
    parser.add_argument('--model', action='append', required=True, help='kind=dir, kind in gipformer|zipformer')
    parser.add_argument('--audio-root', default='.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='outputs/asr_finetune/eval')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    from asr.gipformer import GipFormer
    from asr.zipformer import ZipFormer
    records = read_manifests(args.manifest, args.audio_root)[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for spec in args.model:
        kind, directory = spec.split('=', 1)
        model = (ZipFormer if kind == 'zipformer' else GipFormer)(directory, args.device)
        errors = total = 0
        name = f'{kind}_{Path(directory).name}'
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as handle:
            for record in records:
                hypothesis = model.transcribe(record['audio'])
                ref, hyp = words(record['text']), words(hypothesis)
                e = edit_distance(ref, hyp)
                errors, total = errors + e, total + len(ref)
                handle.write(json.dumps({'id': record['id'], 'ref': ' '.join(ref), 'hyp': ' '.join(hyp), 'errors': e}, ensure_ascii=False) + '\n')
        summary[spec] = {'utterances': len(records), 'words': total, 'wer_percent': round(100 * errors / max(total, 1), 3)}
        del model
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
