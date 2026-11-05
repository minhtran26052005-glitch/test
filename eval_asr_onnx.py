"""WER of one or more ASR model directories on a manifest, with the pipeline's own sherpa-onnx decoding.

Use the held-out test split to compare a base model with its LoRA export (same decoding as run.sh):
  python eval_asr_onnx.py --manifest outputs/asr_finetune/test.jsonl \
      --model gipformer=models/gipformer1.5-68M-rnnt --model gipformer=models/gipformer1.5-68M-rnnt-lora
Writes per-utterance hypotheses next to --out and prints WER per model directory, plus drift of every model
against the first one: changed transcripts, and on labeled audio how many utterances were fixed / broken.
Records without `text` (e.g. unlabeled public_test audio) are only used for the drift check.
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


def drift(records, outputs, specs):
    """How far each model moved from the first (base) one. On labeled audio also how many utterances it fixed
    and how many it broke (base exact, now wrong): breaking correct transcripts is the sign of learning a bias."""
    base = outputs[specs[0]]
    report = {}
    for spec in specs[1:]:
        changed = moved = base_words = fixed = broken = 0
        for record in records:
            before, after = base[record['id']], outputs[spec][record['id']]
            changed += before != after
            moved += edit_distance(before, after)
            base_words += len(before)
            if record.get('text'):
                ref = words(record['text'])
                was, now = edit_distance(ref, before), edit_distance(ref, after)
                fixed += was > 0 and now == 0
                broken += was == 0 and now > 0
        report[spec] = {'changed_utterances_percent': round(100 * changed / max(len(records), 1), 2),
                        'changed_words_per_100': round(100 * moved / max(base_words, 1), 2),
                        'fixed_utterances': fixed, 'broken_utterances': broken}
    return report


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
    summary, outputs = {}, {}
    for spec in args.model:
        kind, directory = spec.split('=', 1)
        model = (ZipFormer if kind == 'zipformer' else GipFormer)(directory, args.device)
        errors = total = 0
        name = f'{kind}_{Path(directory).name}'
        outputs[spec] = {}
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as handle:
            for record in records:
                hyp = words(model.transcribe(record['audio']))
                outputs[spec][record['id']] = hyp
                row = {'id': record['id'], 'hyp': ' '.join(hyp)}
                if record.get('text'):  # unlabeled audio (e.g. public_test) is only used for the drift check
                    ref = words(record['text'])
                    row.update(ref=' '.join(ref), errors=edit_distance(ref, hyp))
                    errors, total = errors + row['errors'], total + len(ref)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        summary[spec] = {'utterances': len(records), 'labeled_words': total,
                         'wer_percent': round(100 * errors / total, 3) if total else None}
        del model
    summary['drift_vs_first_model'] = drift(records, outputs, args.model)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
