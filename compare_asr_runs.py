"""Compare saved GipFormer/ZipFormer transcripts and calling, without inference."""
import csv
import json
import sys
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/zipformer_150m')
    new_root = root / 'benchmark_seed42_n1545'
    old_root = Path('outputs/benchmark_seed42_n1545')
    old = {row['id']: row for row in jsonl(old_root / 'results.jsonl')}
    new = {row['id']: row for row in jsonl(new_root / 'results.jsonl')}
    fields = ['id', 'reference', 'gipformer_asr', 'zipformer_asr', 'gipformer_normalized',
              'zipformer_normalized', 'expected_tool_calls', 'gipformer_tool_calls',
              'zipformer_tool_calls', 'gipformer_state_match', 'zipformer_state_match',
              'gipformer_exact_calls', 'zipformer_exact_calls', 'state_change',
              'gipformer_error', 'zipformer_error']
    counts = dict(improved=0, regressed=0, both_correct=0, both_failed=0)
    def cache(folder, stage):
        path = folder / f'{stage}.cache.json'
        return json.loads(path.read_text())['value'] if path.exists() else ''
    def calls(entry, folder):
        if 'predicted_tool_calls' in entry:
            return entry['predicted_tool_calls']
        trace = folder / 'trace.jsonl'
        # Preserve calls executed before a step-limit/error as well.
        return [{'name': event['name'], 'arguments': event['arguments']}
                for event in jsonl(trace) if event.get('stage') == 'tool'] if trace.exists() else []
    with (root / 'calling_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sid, right in new.items():
            left = old[sid]
            lf, rf = old_root / 'samples' / sid, new_root / 'samples' / sid
            before, after = bool(left['state_match']), bool(right['state_match'])
            change = ('improved' if after else 'regressed') if before != after else ('both_correct' if after else 'both_failed')
            counts[change] += 1
            writer.writerow({'id': sid, 'reference': right['reference_text'],
                'gipformer_asr': left.get('asr_text', cache(lf, 'asr_gipformer')),
                'zipformer_asr': right.get('asr_text', cache(rf, 'asr_zipformer')),
                'gipformer_normalized': left.get('normalized_text', cache(lf, 'normalize')),
                'zipformer_normalized': right.get('normalized_text', cache(rf, 'normalize')),
                'expected_tool_calls': json.dumps(right['expected_tool_calls'], ensure_ascii=False),
                'gipformer_tool_calls': json.dumps(calls(left, lf), ensure_ascii=False),
                'zipformer_tool_calls': json.dumps(calls(right, rf), ensure_ascii=False),
                'gipformer_state_match': before, 'zipformer_state_match': after,
                'gipformer_exact_calls': left['exact_calls'], 'zipformer_exact_calls': right['exact_calls'],
                'state_change': change, 'gipformer_error': left.get('error', ''), 'zipformer_error': right.get('error', '')})
    with Path('outputs/asr_spoken_eval/utterances.csv').open(encoding='utf-8-sig', newline='') as handle:
        baseline = {(row['id'], row['turn'], row['history_index']): row for row in csv.DictReader(handle)}
    with (root / 'asr_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['id', 'turn', 'history_index', 'reference', 'gipformer_asr', 'zipformer_asr', 'audio']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in jsonl(root / 'asr_results.jsonl'):
            index = '' if row['history_index'] is None else str(row['history_index'])
            previous = baseline[(row['id'], row['turn'], index)]
            writer.writerow({'id': row['id'], 'turn': row['turn'], 'history_index': index,
                'reference': row['reference'], 'gipformer_asr': previous['asr'],
                'zipformer_asr': row.get('asr_text', ''), 'audio': row['audio']})
    summary = {'paired_samples': len(new), 'state_outcome_changes': counts,
               'baseline': json.loads((old_root / 'summary.json').read_text()),
               'zipformer': json.loads((new_root / 'summary.json').read_text()),
               'note': 'Paired saved outputs; state-hash/exact-call metrics are proxies, not official score or proof that each difference was caused by ASR.'}
    (root / 'comparison_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({'paired_samples': len(new), 'state_outcome_changes': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
