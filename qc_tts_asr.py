"""Listen to synthesized TTS with both ASRs before any training; keep only audio that says its label.

Both checkpoints act as two independent listeners on every TTS wav (same sherpa-onnx decoding as run.sh):
  tts_failed         every ASR has WER >= --fail-wer: garbled/truncated audio or a different sentence -> drop
  tts_misread        the ASRs agree with each other but not with the label: the TTS most likely read
                     something else (two models rarely make the same mistake) -> drop
  asr_already_right  every ASR is already exact: harmless, but teaches little -> keep (counted)
  asr_learnable      at least one ASR is close (audio is intelligible) and some ASR is wrong -> keep
  uncertain          everything else -> not kept; listen first
The same statistics are computed on the real train audio (outputs/zipformer_150m/asr_comparison.csv) so the
TTS numbers can be judged: much lower WER than real speech = too clean (little to learn, risk of bias toward
TTS speech); much higher shared errors than real speech = TTS mispronunciation.

  python qc_tts_asr.py --manifest outputs/asr_finetune/tts/train.jsonl \
      --model gipformer=models/gipformer1.5-68M-rnnt --model zipformer=models/zipformer-150M-6000h
Writes next to the manifest: <name>_qc/qc.jsonl, summary.json, listen_sample.csv, and <name>_qc.jsonl (kept
records, same format as the manifest) which train_asr_lora.py / run_finetune.sh use instead of the raw TTS.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from prepare_asr_finetune_data import target_text

REAL_ASR = Path('outputs/zipformer_150m/asr_comparison.csv')
KEEP = {'asr_already_right', 'asr_learnable'}


def edits(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def words(text):
    return target_text(text or '').split()


def judge(label, hyps, fail_wer=0.5, close_wer=0.3, misread_wer=0.2):
    """(category, per-model WER) for one utterance; hyps = {model: hypothesis text}."""
    ref = label.split()
    wer = {name: edits(ref, words(hyp)) / max(len(ref), 1) for name, hyp in hyps.items()}
    if all(w >= fail_wer for w in wer.values()):
        return 'tts_failed', wer
    heard = [words(h) for h in hyps.values()]
    if len(heard) > 1 and all(w >= misread_wer for w in wer.values()) and \
            all(edits(heard[0], other) <= 1 for other in heard[1:]):
        return 'tts_misread', wer
    if all(w == 0 for w in wer.values()):
        return 'asr_already_right', wer
    if min(wer.values()) <= close_wer:
        return 'asr_learnable', wer
    return 'uncertain', wer


def profile(pairs):
    """Corpus statistics comparable between TTS and real audio. pairs: [(label, {model: hyp})]."""
    count = Counter()
    errors, total = Counter(), 0
    for label, hyps in pairs:
        ref = label.split()
        total += len(ref)
        heard = {name: words(h) for name, h in hyps.items()}
        wrong = {name: edits(ref, h) for name, h in heard.items()}
        errors.update(wrong)
        values = list(heard.values())
        if all(wrong.values()):
            count['all_wrong'] += 1
            count['all_wrong_identically'] += all(v == values[0] for v in values[1:])
        count['all_wer_ge_50'] += all(w / max(len(ref), 1) >= 0.5 for w in wrong.values())
    n = max(len(pairs), 1)
    return {'utterances': len(pairs), **{f'wer_{name}': round(100 * e / max(total, 1), 2) for name, e in sorted(errors.items())},
            'percent_all_wrong_identically': round(100 * count['all_wrong_identically'] / n, 2),
            'percent_all_wer_ge_50': round(100 * count['all_wer_ge_50'] / n, 2)}


def real_baseline(path=REAL_ASR, names=('gipformer', 'zipformer')):
    if not path.is_file():
        return None
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    pairs = [(target_text(r['reference']), {n: r[f'{n}_asr'] for n in names if f'{n}_asr' in r}) for r in rows]
    result = profile(pairs)
    # The same rules on real speech: how much a perfectly good recording would be dropped anyway.
    categories = Counter(judge(label, hyps)[0] for label, hyps in pairs)
    result['categories_percent'] = {k: round(100 * v / max(len(pairs), 1), 2) for k, v in sorted(categories.items())}
    return result


def transcribe_all(records, kind, directory, cache, device):
    """{id: hypothesis}; reuses <cache> so thresholds can be changed without decoding again."""
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done[row['id']] = row['hyp']
    todo = [r for r in records if r['id'] not in done]
    if todo:
        from asr.gipformer import GipFormer
        from asr.zipformer import ZipFormer
        model = (ZipFormer if kind == 'zipformer' else GipFormer)(directory, device)
        with cache.open('a', encoding='utf-8') as handle:
            for index, record in enumerate(todo, 1):
                done[record['id']] = model.transcribe(record['audio'])
                handle.write(json.dumps({'id': record['id'], 'hyp': done[record['id']]}, ensure_ascii=False) + '\n')
                if index % 200 == 0:
                    print(f'[QC] {kind} {index}/{len(todo)}', flush=True)
        del model
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, help='train.jsonl written by synthesize_tts.py')
    parser.add_argument('--model', action='append', required=True, help='kind=dir; use both ASRs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fail-wer', type=float, default=0.5)
    parser.add_argument('--close-wer', type=float, default=0.3)
    parser.add_argument('--misread-wer', type=float, default=0.2)
    parser.add_argument('--listen-per-category', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    out = manifest.with_name(manifest.stem + '_qc')
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    models = [spec.split('=', 1) for spec in args.model]
    hyps = {kind: transcribe_all(records, kind, directory, out / f'hyps_{kind}.jsonl', args.device) for kind, directory in models}
    rows = []
    for record in records:
        heard = {kind: hyps[kind][record['id']] for kind, _ in models}
        category, wer = judge(record['text'], heard, args.fail_wer, args.close_wer, args.misread_wer)
        rows.append({'id': record['id'], 'category': category, 'wer': {k: round(v, 3) for k, v in wer.items()},
                     'label': record['text'], 'tts_text': record.get('reference', ''), 'hyps': heard,
                     'audio': record['audio'], 'target_model': record.get('target_model'), 'seconds': record.get('seconds')})
    with (out / 'qc.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    category = {row['id']: row['category'] for row in rows}
    kept = [r for r in records if category[r['id']] in KEEP]
    with manifest.with_name(manifest.stem + '_qc.jsonl').open('w', encoding='utf-8') as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    rng = random.Random(args.seed)
    with (out / 'listen_sample.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['category', 'id', 'audio', 'label', 'tts_text', *[f'hyp_{k}' for k, _ in models],
                                                    *[f'wer_{k}' for k, _ in models], 'nghe_dung_nhan', 'ghi_chu'])
        writer.writeheader()
        for name in ('tts_failed', 'tts_misread', 'uncertain', 'asr_learnable', 'asr_already_right'):
            group = [r for r in rows if r['category'] == name]
            for row in rng.sample(group, min(args.listen_per_category, len(group))):
                writer.writerow({'category': name, 'id': row['id'], 'audio': row['audio'], 'label': row['label'], 'tts_text': row['tts_text'],
                                 **{f'hyp_{k}': row['hyps'][k] for k, _ in models}, **{f'wer_{k}': row['wer'][k] for k, _ in models},
                                 'nghe_dung_nhan': '', 'ghi_chu': ''})
    summary = {'manifest': str(manifest), 'records': len(records), 'kept': len(kept),
               'categories': dict(Counter(category.values())),
               'categories_percent': {k: round(100 * v / max(len(records), 1), 2) for k, v in sorted(Counter(category.values()).items())},
               'kept_by_target_model': dict(Counter(r.get('target_model') for r in kept)),
               'tts_profile': profile([(r['text'], {k: hyps[k][r['id']] for k, _ in models}) for r in records]),
               'real_audio_profile': real_baseline(names=tuple(k for k, _ in models)),
               'thresholds': {'fail_wer': args.fail_wer, 'close_wer': args.close_wer, 'misread_wer': args.misread_wer},
               'how_to_read': 'Compare tts_profile with real_audio_profile. Listen to listen_sample.csv (fill nghe_dung_nhan) '
                              'before training; if tts_misread/uncertain samples actually sound right, relax the thresholds.'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ('records', 'kept', 'categories_percent', 'tts_profile', 'real_audio_profile')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
