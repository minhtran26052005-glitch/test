"""Build ASR fine-tuning manifests for GipFormer/ZipFormer from data already in the repo; no model is run.

real audio   every train utterance (final turn + user history, 5,036) -> its reference transcript, written the
             way both models emit text: uppercase syllables, numbers/dates/times read out, no punctuation.
tts requests generated variants that passed audit_finetune_data.py for ASR (asr_use=ok) from seeds in the
             fit split only, with the seed's own train recording as the voice to clone (synthesize_tts.py).

Splits follow the split manifest: fit -> train, validation, test. Never train on test: it is the held-out
check for both the ASR WER and the agent benchmark.

Usage: python prepare_asr_finetune_data.py [--out outputs/asr_finetune]
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from asr_spoken_text import spoken_tokens

UTTERANCES = 'outputs/asr_error_audit/utterances.csv'
VARIANTS = 'outputs/finetune_data_audit/variants.csv'
GENERATED = 'outputs/asr_datagen_api/generated.csv'
MANIFEST = 'outputs/timestamp_word_fusion/split_manifest.json'
SPLIT_FILE = {'fit': 'train', 'validation': 'validation', 'test': 'test'}


def target_text(text):
    """Reference -> model output style (both checkpoints emit uppercase spoken syllables without punctuation)."""
    return ' '.join(spoken_tokens(text)).upper()


def relative_audio(path):
    """Source-machine absolute paths -> path relative to the repo root (data/train+public_test/...)."""
    marker = 'data/train+public_test/'
    return path[path.index(marker):] if marker in path else path


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def real_audio_records(utterances, splits):
    for row in utterances:
        index = row['history_index']
        yield {'id': f"{row['id']}:{row['turn']}{':' + index if index else ''}", 'audio': relative_audio(row['audio']),
               'text': target_text(row['reference']), 'reference': row['reference'], 'sample_id': row['id'],
               'turn': row['turn'], 'source': 'train_audio', 'split': SPLIT_FILE.get(splits.get(row['id']), 'unsplit')}


def tts_requests(variants, generated, references, splits):
    source_audio = {row['variant_id']: row['source_audio'] for row in generated}
    for row in variants:
        if row['asr_use'] != 'ok' or splits.get(row['sample_id']) != 'fit':
            continue
        audio = relative_audio(source_audio.get(row['variant_id'], ''))
        yield {'id': row['variant_id'], 'tts_text': row['spoken_text'], 'text': target_text(row['spoken_text']),
               'speaker_audio': audio, 'speaker_text': references.get(audio, ''), 'sample_id': row['sample_id'],
               'turn': row['turn'], 'target_model': row['target_model'], 'domain': row['domain'], 'source': 'tts', 'split': 'train'}


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='outputs/asr_finetune')
    args = parser.parse_args()
    out = Path(args.out)
    splits = {sid: item['split'] for sid, item in json.loads(Path(MANIFEST).read_text()).items()}
    utterances = rows(UTTERANCES)
    real = list(real_audio_records(utterances, splits))
    by_split = {}
    for record in real:
        by_split.setdefault(record['split'], []).append(record)
    for split, records in by_split.items():
        write(out / f'{split}.jsonl', records)
    references = {relative_audio(row['audio']): row['reference'] for row in utterances}
    tts = list(tts_requests(rows(VARIANTS), rows(GENERATED), references, splits))
    write(out / 'tts_requests.jsonl', tts)
    summary = {'real_audio': dict(Counter(r['split'] for r in real)),
               'tts_requests_fit_only': len(tts), 'tts_by_turn': dict(Counter(r['turn'] for r in tts)),
               'tts_without_speaker_audio': sum(not r['speaker_audio'] for r in tts),
               'target_format': 'uppercase spoken syllables, no punctuation (asr_spoken_text.spoken_tokens)',
               'note': 'Paths are relative to the repo root. tts_requests need synthesize_tts.py before training.'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
