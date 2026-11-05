"""One resumable CPU pass over both ASRs; no Qwen or simulator execution."""
import csv
import json
import time
from pathlib import Path

from asr.gipformer import GipFormer
from asr.zipformer import ZipFormer
from asr.timed import atomic_json, cache_identity, to_words
from main import model_fingerprint

ROOT = Path('outputs/asr_timed_v1')
MODELS = {'gipformer': (GipFormer, 'models/gipformer1.5-68M-rnnt'),
          'zipformer': (ZipFormer, 'models/zipformer-150M-6000h')}


def main():
    import sherpa_onnx
    ROOT.mkdir(parents=True, exist_ok=True)
    with Path('outputs/zipformer_150m/asr_comparison.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5036
    started = time.time()
    failures = drift = count = 0
    manifest = []
    with (ROOT / 'manifest.jsonl').open('w', encoding='utf-8') as output:
        for name, (Model, path) in MODELS.items():
            model = None
            fingerprint = model_fingerprint(path)
            for row in rows:
                count += 1
                entry = {key: row[key] for key in ('id', 'turn', 'history_index', 'audio')}
                entry['model'] = name
                try:
                    key, identity = cache_identity(row['audio'], fingerprint, sherpa_onnx.__version__)
                    destination = ROOT / name / (key + '.json')
                    if destination.exists():
                        cached = json.loads(destination.read_text())
                        assert cached['key'] == key
                        entry['cache_hit'] = True
                    else:
                        if model is None:
                            model = Model(path, 'cpu')
                        start = time.monotonic()
                        detail = model.transcribe_detailed(row['audio'])
                        detail['words'] = to_words(detail)
                        cached = {'key': key, 'identity': identity, **detail,
                                  'latency_seconds': time.monotonic()-start}
                        atomic_json(destination, cached)
                        entry['cache_hit'] = False
                    entry.update(cache=str(destination), cache_key=key,
                                 text_matches_previous=cached['text'] == row[name + '_asr'],
                                 words=len(cached['words']))
                    drift += not entry['text_matches_previous']
                except Exception as exc:
                    failures += 1
                    entry['error'] = f'{type(exc).__name__}: {exc}'
                    print('[TIMED ASR ERROR]', json.dumps(entry, ensure_ascii=False), flush=True)
                manifest.append(entry)
                output.write(json.dumps(entry, ensure_ascii=False)+'\n')
                output.flush()
                atomic_json(ROOT / 'progress.json', {'processed': count, 'total': 2*len(rows),
                    'model': name, 'current_id': row['id'], 'errors': failures,
                    'text_drift_from_previous': drift, 'elapsed_seconds': round(time.time()-started, 2),
                    'finished': count == 2*len(rows)})
                if count == 1 or count % 100 == 0 or count == 2*len(rows):
                    print(f'[TIMED ASR] {count}/{2*len(rows)} errors={failures} text_drift={drift}', flush=True)
            del model
    if failures:
        raise RuntimeError(f'{failures} timed ASR failures; completed cache entries are reusable')


if __name__ == '__main__':
    main()
