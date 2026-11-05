"""WER of one or more ASR model directories on a manifest, with the pipeline's own sherpa-onnx decoding.

Use the held-out test split to compare a base model with its LoRA export (same decoding as run.sh):
  python eval_asr_onnx.py --manifest outputs/asr_finetune/test.jsonl \
      --model gipformer=models/gipformer1.5-68M-rnnt --model gipformer=models/gipformer1.5-68M-rnnt-lora
A model may carry its own decoding after '@' (keys decoding, beam, hotwords, hotwords_score), e.g.
  --model 'gipformer=models/gipformer1.5-68M-rnnt@decoding=modified_beam_search,hotwords=outputs/asr_hotwords/gipformer.txt'
Writes per-utterance hypotheses next to --out and prints WER per model directory, plus drift of every model
against the first one: changed transcripts, and on labeled audio how many utterances were fixed / broken.
`foreign_words` scores the tokens no Vietnamese syllable spells (SJC, ZALO, SMS, ...), where small in-domain data
and hotwords act: recall of the reference ones and false alarms (such words heard but not said; a misspelling such
as SC for SJC counts as a miss and a false alarm, as in keyword spotting).
Records without `text` (e.g. unlabeled public_test audio) are only used for the drift check.
"""
import argparse
import json
from pathlib import Path

from collections import Counter

from asr_lora.data import read_manifests
from asr_spoken_text import spoken_tokens

DECODING_KEYS = ('decoding', 'beam', 'hotwords', 'hotwords_score')


def words(text):
    return ' '.join(spoken_tokens(text)).upper().split()


def edit_distance(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def parse_spec(spec):
    """kind=dir[@key=value,...] -> (kind, dir, sherpa-onnx decoding options, short label for file names)."""
    from asr.gipformer import decoding_options
    kind, rest = spec.split('=', 1)
    if kind not in ('gipformer', 'zipformer'):
        raise ValueError(f'{spec}: kind must be gipformer or zipformer')
    directory, _, extra = rest.partition('@')
    settings = dict(item.split('=', 1) for item in extra.split(',') if item)
    unknown = sorted(set(settings) - set(DECODING_KEYS))
    if unknown:
        raise ValueError(f'{spec}: unknown decoding keys {unknown}; use {DECODING_KEYS}')
    options = decoding_options(kind.upper(), {f'{kind.upper()}_{k.upper()}': v for k, v in settings.items()})
    label = '_'.join(f'{k}-{Path(v).stem if k == "hotwords" else v}' for k, v in settings.items())
    return kind, directory, options, label


def keyword_scores(reference, hypothesis):
    """(said, recalled, false alarms) of foreign words; counts per word, so a repeated word must be heard twice."""
    from build_hotwords import foreign_words
    said, heard = Counter(foreign_words(reference)), Counter(foreign_words(hypothesis))
    recalled = sum(min(n, heard[w]) for w, n in said.items())
    return sum(said.values()), recalled, sum(max(0, n - said[w]) for w, n in heard.items())


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
    if len(set(args.model)) != len(args.model):
        raise ValueError('The same --model is given twice')
    from asr.gipformer import decoding_identity
    summary, outputs = {}, {}
    for spec in args.model:
        kind, directory, decoding, label = parse_spec(spec)
        model = (ZipFormer if kind == 'zipformer' else GipFormer)(directory, args.device, decoding)
        errors = total = said = recalled = false_alarms = labeled = 0
        name = f'{kind}_{Path(directory).name}' + (f'_{label}' if label else '')
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
                    n, hit, extra = keyword_scores(ref, hyp)
                    said, recalled, false_alarms, labeled = said + n, recalled + hit, false_alarms + extra, labeled + 1
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        summary[spec] = {'utterances': len(records), 'labeled_words': total, 'decoding': decoding_identity(decoding) or 'greedy_search',
                         'wer_percent': round(100 * errors / total, 3) if total else None,
                         'foreign_words': {'said': said, 'recall_percent': round(100 * recalled / said, 2) if said else None,
                                           'false_alarms_per_100_utterances': round(100 * false_alarms / labeled, 2) if labeled else None}}
        del model
    summary['drift_vs_first_model'] = drift(records, outputs, args.model)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
