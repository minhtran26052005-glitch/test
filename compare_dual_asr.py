"""Offline three-way comparison; keep both original ASRs and Qwen fusion."""
import csv
import json
from pathlib import Path


def main():
    roots = {'gipformer': Path('outputs/benchmark_seed42_n1545'),
             'zipformer': Path('outputs/zipformer_150m/benchmark_seed42_n1545'),
             'dual': Path('outputs/dual_asr_qwen/benchmark_seed42_n1545')}
    runs = {name: {r['id']: r for line in (root / 'results.jsonl').read_text().splitlines() if (r := json.loads(line))}
            for name, root in roots.items()}
    target = Path('outputs/dual_asr_qwen')
    counts = {}
    rows = []
    def cache(model, sid, stage):
        path = roots[model] / 'samples' / sid / (stage + '.cache.json')
        return json.loads(path.read_text())['value'] if path.exists() else ''
    def calls(model, sid):
        entry = runs[model][sid]
        value = entry.get('predicted_tool_calls')
        if value is None:
            trace = roots[model] / 'samples' / sid / 'trace.jsonl'
            events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()] if trace.exists() else []
            value = [{'name': event['name'], 'arguments': event['arguments']} for event in events if event.get('stage') == 'tool']
        return json.dumps(value, ensure_ascii=False)
    for sid, result in runs['dual'].items():
        row = {'id': sid, 'reference': result['reference_text'],
               'gipformer_asr': cache('dual', sid, 'asr_gipformer'),
               'zipformer_asr': cache('dual', sid, 'asr_zipformer'),
               'expected_calls': json.dumps(result['expected_tool_calls'], ensure_ascii=False)}
        for model in roots:
            entry = runs[model][sid]
            row[model + '_normalized'] = entry.get('normalized_text', cache(model, sid, 'normalize'))
            row[model + '_actual_calls'] = calls(model, sid)
            row[model + '_state_match'] = entry['state_match']
            row[model + '_exact_calls'] = entry['exact_calls']
            row[model + '_error'] = entry.get('error', '')
        rows.append(row)
    for model in roots:
        selected = [runs[model][sid] for sid in runs['dual']]
        counts[model] = {'samples': len(selected), 'state_match': sum(r['state_match'] for r in selected),
                         'exact_calls': sum(r['exact_calls'] for r in selected), 'errors': sum('error' in r for r in selected)}
        for metric in ('state_match', 'exact_calls'):
            counts[model][metric + '_percent'] = round(100*counts[model][metric]/len(selected), 2)
    with (target / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'models': counts, 'note': 'Qwen dual-ASR review includes user history audio. Same tool agent and prompts. Hash/exact-call metrics are local proxies, not an official semantic score.'}
    (target / 'comparison_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
