"""Recognize every user audio once, keeping reusable per-audio checkpoints."""
import csv
import json
import os
import time
from pathlib import Path
from asr import model_spec
from main import asr_cache_key, model_fingerprint, save_cache, stage_cache


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_precache(args):
    if not os.environ.get('ASR_CACHE_DIR'):
        raise ValueError('--precache-asr cần ASR_CACHE_DIR để giữ transcript dùng lại cho pipeline')
    split = args.split or os.environ['DATA_SPLIT']
    source = Path(os.environ['DATA_ROOT']).resolve() / split
    items = [json.loads(line) for line in (source / f'{split}.jsonl').read_text().splitlines() if line.strip()]
    tasks = []
    for item in items:
        # References below are exported for inspection; never passed to ASR.
        tasks.append({'id': item['id'], 'turn': 'last', 'history_index': None,
                      'audio': str(source / item['audio']), 'reference': item.get('audio_text', '')})
        for index, turn in enumerate(item.get('history', [])):
            if turn.get('role') == 'user' and turn.get('audio'):
                tasks.append({'id': item['id'], 'turn': 'history', 'history_index': index,
                              'audio': str(source / turn['audio']), 'reference': turn.get('audio_text', '')})
    destination = Path(os.environ['OUTPUT_DIR'])
    destination.mkdir(parents=True, exist_ok=True)
    shared = Path(os.environ['ASR_CACHE_DIR'])
    shared.mkdir(parents=True, exist_ok=True)
    Model, model_path = model_spec(os.environ['ASR_PRIMARY'])
    fingerprint = model_fingerprint(model_path)
    name, device = os.environ['ASR_PRIMARY'], os.environ['ASR_DEVICE']
    records = []
    model = None
    started = time.time()
    failures = 0
    manifest = destination / 'asr_results.jsonl'
    # The canonical checkpoints are atomic .cache.json files. Rebuild this view
    # on resume from those files, without another model call for completed audio.
    with manifest.open('w', encoding='utf-8') as output:
        for index, task in enumerate(tasks, 1):
            record = dict(task, model=model_path, asr_name=name)
            try:
                key = asr_cache_key(name, fingerprint, task['audio'], device)
                value = stage_cache(shared, key, key, True)
                record.update(cache_key=key, from_cache=value is not None)
                if value is None:
                    if model is None:
                        model = Model(model_path, device)
                    start = time.monotonic()
                    value = model.transcribe(task['audio'])
                    save_cache(shared, key, key, value)
                    record['latency_seconds'] = time.monotonic() - start
                record['asr_text'] = value
            except Exception as exc:
                failures += 1
                record['error'] = f'{type(exc).__name__}: {exc}'
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            atomic_json(destination / 'asr_progress.json', {
                'stage': 'asr', 'model': model_path, 'completed': index, 'total': len(tasks),
                'errors': failures, 'current_id': task['id'], 'current_turn': task['turn'],
                'elapsed_seconds': round(time.time() - started, 2), 'finished': index == len(tasks),
            })
            if index == 1 or index % 25 == 0 or index == len(tasks):
                print(f'[ASR PRECACHE] {index}/{len(tasks)} errors={failures}', flush=True)
    with (destination / 'asr_transcripts.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        columns = ['id', 'turn', 'history_index', 'audio', 'reference', 'asr_text', 'error', 'cache_key']
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    if failures:
        raise RuntimeError(f'{failures} audio ASR lỗi; xem {manifest}; chạy tiếp sẽ sử dụng lại cache đã hoàn thành')
