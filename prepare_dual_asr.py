"""Validate/reuse both completed ASR runs; never import or invoke a recognizer."""
import json
import os
import shutil
from pathlib import Path
from main import asr_cache_key, model_fingerprint, save_cache


def main():
    destination = Path(os.environ['OUTPUT_DIR'])
    shared = Path(os.environ['ASR_CACHE_DIR'])
    shared.mkdir(parents=True, exist_ok=True)
    zip_root = Path('outputs/zipformer_150m')
    old_root = Path('outputs/benchmark_seed42_n1545/samples')
    model_paths = {'gipformer': os.environ['GIPFORMER_MODEL_PATH'], 'zipformer': os.environ['ZIPFORMER_MODEL_PATH']}
    fingerprints = {name: model_fingerprint(path) for name, path in model_paths.items()}
    records = [json.loads(line) for line in (zip_root / 'asr_results.jsonl').read_text().splitlines() if line.strip()]
    seen = set()
    sources = []
    for row in records:
        uid = (row['id'], row['turn'], row['history_index'])
        if uid in seen:
            raise ValueError(f'Duplicate audio: {uid}')
        seen.add(uid)
        for name in ('gipformer', 'zipformer'):
            if name == 'gipformer':
                stage = 'asr_gipformer' if row['turn'] == 'last' else f"history_{row['history_index']}_gipformer"
                source = old_root / row['id'] / f'{stage}.cache.json'
            else:
                source = zip_root / 'asr_cache' / (row['cache_key'] + '.cache.json')
            value = json.loads(source.read_text())
            key = asr_cache_key(name, fingerprints[name], row['audio'], 'cpu')
            if value['key'] != key:
                raise ValueError(f'Cache does not match model/audio: {source}')
            target = shared / (key + '.cache.json')
            if not target.exists():
                shutil.copyfile(source, target)
            else:
                assert json.loads(target.read_text()) == value, target
            sources.append({'id': row['id'], 'turn': row['turn'], 'history_index': row['history_index'],
                            'model': name, 'key': key, 'source_cache': str(source), 'audio': row['audio']})
    if len(seen) != 5036:
        raise ValueError(f'Expected 5036 train audios; got {len(seen)}')
    with (destination / 'asr_cache_sources.jsonl').open('w', encoding='utf-8') as handle:
        for row in sources:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'audios': len(seen), 'verified_transcripts': len(sources), 'model_paths': model_paths,
                'asr_model_calls': 0, 'cache_validation': 'Every key matches current model fingerprint, audio SHA256, decoder device cpu, and ASR name.',
                'fusion': 'Qwen reviews both saved transcripts for last turn and user audio history; agent consumes the merged normalized text.',
                'no_reference_input': True}
    (destination / 'asr_cache_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
