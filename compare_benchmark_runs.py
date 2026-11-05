"""Paired comparison of two saved benchmark runs on the same samples; reads outputs only, no inference.

Usage: python compare_benchmark_runs.py BASE_RUN NEW_RUN OUT_DIR [KEYWORD_EVIDENCE_CSV]

Samples are grouped by ASR keyword preservation (from keyword_evidence.csv) to show whether a change
helps where one ASR kept a keyword the other lost, where both lost it, or where ASR kept every keyword.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

KEYWORD_EVIDENCE = 'outputs/timestamp_word_fusion/comparison/keyword_evidence.csv'
GROUPS = ('mot_asr_giu_duoc', 'ca_hai_asr_lech', 'khong_lech', 'khong_ro')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def keyword_groups(path):
    """Per sample, over final turn and history: one_asr_kept > both_lost > none_lost."""
    flags = defaultdict(set)
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            gip, zip_ = row['gipformer_flagged'] == 'True', row['zipformer_flagged'] == 'True'
            flags[row['id']].add('mot_asr_giu_duoc' if gip != zip_ else 'ca_hai_asr_lech' if gip else 'khong_lech')
    return {sid: next(group for group in GROUPS if group in found) for sid, found in flags.items()}


def outcome(before, after):
    if before == after:
        return 'both_correct' if after else 'both_failed'
    return 'improved' if after else 'regressed'


def compare(base, new, groups):
    base = {row['id']: row for row in base}
    rows, changes = [], Counter()
    by_group, by_calls = defaultdict(Counter), defaultdict(Counter)
    for item in new:
        sid = item['id']
        if sid not in base:
            raise ValueError(f'{sid} không có trong run gốc; hai run phải cùng tập mẫu')
        left = base[sid]
        change = outcome(bool(left['state_match']), bool(item['state_match']))
        group = groups.get(sid, 'khong_ro')
        expected = item.get('expected_tool_calls') or left.get('expected_tool_calls') or []
        calls = str(min(len(expected), 4)) + ('+' if len(expected) >= 4 else '')
        changes[change] += 1
        for bucket in (by_group[group], by_calls[calls]):
            bucket['samples'] += 1
            bucket['base_state_match'] += bool(left['state_match'])
            bucket['new_state_match'] += bool(item['state_match'])
            bucket[change] += 1
        rows.append({'id': sid, 'state_change': change, 'keyword_group': group, 'gold_calls': len(expected),
                     'reference': item.get('reference_text', ''), 'asr_text': item.get('asr_text', ''),
                     'normalized_text': item.get('normalized_text', ''),
                     'expected_tool_calls': json.dumps(expected, ensure_ascii=False),
                     'base_tool_calls': json.dumps(left.get('predicted_tool_calls', []), ensure_ascii=False),
                     'new_tool_calls': json.dumps(item.get('predicted_tool_calls', []), ensure_ascii=False),
                     'base_state_match': bool(left['state_match']), 'new_state_match': bool(item['state_match']),
                     'base_exact_calls': bool(left['exact_calls']), 'new_exact_calls': bool(item['exact_calls']),
                     'base_error': left.get('error', ''), 'new_error': item.get('error', '')})
    def totals(selected):
        return {'samples': len(selected), 'state_match': sum(bool(r['state_match']) for r in selected),
                'exact_calls': sum(bool(r['exact_calls']) for r in selected), 'errors': sum('error' in r for r in selected)}
    summary = {'base': totals([base[r['id']] for r in new]), 'new': totals(new), 'state_outcome_changes': dict(changes),
               'by_keyword_group': {k: dict(v) for k, v in sorted(by_group.items())},
               'by_gold_calls': {k: dict(v) for k, v in sorted(by_calls.items())},
               'note': 'Paired local proxy (state hash / exact calls), not an official score. keyword_group comes from automatic keyword flags, not manual review.'}
    return rows, summary


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    base_root, new_root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    evidence = Path(sys.argv[4] if len(sys.argv) == 5 else KEYWORD_EVIDENCE)
    groups = keyword_groups(evidence) if evidence.exists() else {}
    rows, summary = compare(jsonl(base_root / 'results.jsonl'), jsonl(new_root / 'results.jsonl'), groups)
    summary.update(base_run=str(base_root), new_run=str(new_root), keyword_evidence=str(evidence) if groups else None)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['id'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ('base', 'new', 'state_outcome_changes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
