"""Export ASR, normalizer and raw Qwen decisions for a train benchmark."""
import csv
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/benchmark_seed42_n1545')
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'asr_qwen_debug.csv'
    dataset = Path('data/train+public_test/train/train.jsonl')
    with (root / 'results.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = {row['id'] for row in rows}
    with dataset.open(encoding='utf-8') as handle:
        references = {item['id']: item for line in handle if line.strip() and (item := json.loads(line))['id'] in ids}
    if len(ids) != len(rows) or ids != references.keys():
        raise ValueError('Danh sách mẫu bị trùng hoặc thiếu nhãn transcript')
    missing = {'asr': 0, 'normalized': 0, 'decisions': 0}
    with destination.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['reference_asr', 'asr_predict', 'qwen_decisions', 'qwen_normalized_text', 'expected_tool_calls'])
        for row in rows:
            sid = row['id']
            folder = root / 'samples' / sid
            def cache(stage):
                path = folder / (stage + '.cache.json')
                return json.loads(path.read_text(encoding='utf-8'))['value'] if path.exists() else ''
            asr = row.get('asr_text')
            if asr is None:
                paths = sorted(folder.glob('asr_*.cache.json'))
                if len(paths) == 1:
                    asr = json.loads(paths[0].read_text(encoding='utf-8'))['value']
                else:
                    asr = ''
            normalized = cache('normalize')
            tracefile = folder / 'trace.jsonl'
            traces = [json.loads(line) for line in tracefile.read_text(encoding='utf-8').splitlines() if line.strip()] if tracefile.exists() else []
            decisions = [event.get('output', '') for event in traces if event.get('stage') == 'agent']
            missing['asr'] += not bool(asr)
            missing['normalized'] += not bool(normalized)
            missing['decisions'] += not bool(decisions)
            qwen_text = '\n'.join([f'ID: {sid}'] + decisions)
            expected = json.dumps(references[sid]['label']['tool_calls'], ensure_ascii=False)
            writer.writerow([references[sid]['audio_text'], asr, qwen_text, normalized, expected])
    with destination.open(encoding='utf-8-sig', newline='') as handle:
        check = csv.reader(handle)
        header = next(check)
        count = sum(1 for row in check if len(row) == len(header))
    if count != len(rows):
        raise RuntimeError(f'CSV không đủ số hàng: {count}/{len(rows)}')
    print(json.dumps({'path': str(destination), 'rows': count, 'columns': header, 'missing': missing}, ensure_ascii=False))

if __name__ == '__main__':
    main()
