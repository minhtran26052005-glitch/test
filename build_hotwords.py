"""Domain hotword lists for sherpa-onnx contextual biasing, built from the fit split only (no retraining).

A word is a candidate when it cannot be a Vietnamese syllable (SJC, ZALO, SMS, MEETING, OMEPRAZOLE): Vietnamese is
written one syllable per word, so these are codes, brands, loanwords and foreign names, the tokens a small
fine-tuning set teaches worst. Each model gets the candidates it misses often on the fit split's real recordings
(saved greedy transcripts): occurrences >= --min-count and miss rate >= --min-miss-rate. Validation and test
transcripts are never read, so they stay a fair check.

Output: <out>/<model>.txt (one uppercase word per line, written as the model writes it) and summary.json.
Check on validation before turning it on in run.sh (hotwords only act in modified_beam_search):
  python eval_asr_onnx.py --manifest outputs/asr_finetune/validation.jsonl \\
      --model gipformer=models/gipformer1.5-68M-rnnt \\
      --model 'gipformer=models/gipformer1.5-68M-rnnt@decoding=modified_beam_search' \\
      --model 'gipformer=models/gipformer1.5-68M-rnnt@decoding=modified_beam_search,hotwords=outputs/asr_hotwords/gipformer.txt'
then GIPFORMER_DECODING=modified_beam_search GIPFORMER_HOTWORDS=outputs/asr_hotwords/gipformer.txt ./run.sh ...
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REAL_ASR = Path('outputs/zipformer_150m/asr_comparison.csv')
MANIFEST = Path('outputs/timestamp_word_fusion/split_manifest.json')
TONE_MARKS = {'̀', '́', '̃', '̉', '̣'}
# Onset + vowel nucleus + final of a Vietnamese syllable (tone marks removed, vowel letters ă â ê ô ơ ư kept).
SYLLABLE = re.compile(r'(NGH|NG|NH|CH|GH|GI|KH|PH|QU|TH|TR|B|C|D|Đ|G|H|K|L|M|N|P|R|S|T|V|X)?'
                      r'[AĂÂEÊIOÔƠUƯY]{1,3}(NG|NH|CH|C|M|N|P|T)?')


def is_vietnamese_syllable(word):
    word = unicodedata.normalize('NFD', word.upper())
    word = unicodedata.normalize('NFC', ''.join(c for c in word if c not in TONE_MARKS))
    return bool(SYLLABLE.fullmatch(word))


def foreign_words(tokens):
    """Tokens of at least two letters that no Vietnamese syllable spells: codes, brands, loanwords, foreign names."""
    return [t for t in tokens if sum(c.isalpha() for c in t) >= 2 and not is_vietnamese_syllable(t)]


def miss_counts(rows, model):
    """{word: occurrences}, {word: utterances where the model's transcript lacks it} over (label, hyps) rows."""
    from eval_asr_onnx import words
    seen, missed = Counter(), Counter()
    for label, hyps in rows:
        heard = set(words(hyps[model]))
        for word in set(foreign_words(label.split())):
            seen[word] += 1
            missed[word] += word not in heard
    return seen, missed


def select(seen, missed, min_count=3, min_miss_rate=0.3):
    chosen = [w for w, n in seen.items() if n >= min_count and missed[w] / n >= min_miss_rate]
    return sorted(chosen, key=lambda w: (-missed[w], w))


def fit_rows(path=REAL_ASR, manifest=MANIFEST, models=('gipformer', 'zipformer')):
    from prepare_asr_finetune_data import target_text
    splits = {sid: item['split'] for sid, item in json.loads(Path(manifest).read_text()).items()}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if splits.get(r['id']) == 'fit']
    return [(target_text(r['reference']), {m: r[f'{m}_asr'] for m in models}) for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--asr', default=str(REAL_ASR), help='CSV with reference and <model>_asr columns (train audio)')
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--model', action='append', choices=['gipformer', 'zipformer'])
    parser.add_argument('--min-count', type=int, default=3, help='fit utterances containing the word')
    parser.add_argument('--min-miss-rate', type=float, default=0.3, help='share of those the model transcribes without it')
    parser.add_argument('--out', default='outputs/asr_hotwords')
    args = parser.parse_args()
    models = args.model or ['gipformer', 'zipformer']
    rows = fit_rows(args.asr, args.manifest, models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {'source': args.asr, 'split': 'fit', 'utterances': len(rows), 'min_count': args.min_count,
               'min_miss_rate': args.min_miss_rate, 'models': {}}
    for model in models:
        seen, missed = miss_counts(rows, model)
        chosen = select(seen, missed, args.min_count, args.min_miss_rate)
        (out / f'{model}.txt').write_text(''.join(w + '\n' for w in chosen), encoding='utf-8')
        summary['models'][model] = {
            'hotwords': len(chosen), 'candidate_words': len(seen),
            'missed_occurrences_covered': sum(missed[w] for w in chosen), 'missed_occurrences_total': sum(missed.values()),
            'words': [{'word': w, 'utterances': seen[w], 'missed': missed[w]} for w in chosen]}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({m: {k: v for k, v in s.items() if k != 'words'} for m, s in summary['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
