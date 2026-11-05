"""Paired comparison of two saved benchmark runs on the same samples; reads outputs only, no inference.

Usage: python compare_benchmark_runs.py BASE_RUN NEW_RUN OUT_DIR [KEYWORD_EVIDENCE_CSV]

Samples are grouped by ASR keyword preservation (from keyword_evidence.csv) to show whether a change
helps where one ASR kept a keyword the other lost, where both lost it, or where ASR kept every keyword.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

KEYWORD_EVIDENCE = 'outputs/timestamp_word_fusion/comparison/keyword_evidence.csv'
SPLIT_MANIFEST = 'outputs/timestamp_word_fusion/split_manifest.json'  # fit/validation/test; only test is fair after fine-tuning on fit
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


def mcnemar_exact(improved, regressed):
    """Two-sided exact McNemar p-value on the samples whose outcome changed (both runs on the same samples).

    Only flips carry evidence: with 4 improved and 1 regressed, p = 0.375, i.e. the change is still noise. On a
    test split of a few hundred samples, small fine-tuning gains need this check before they are believed."""
    flips = improved + regressed
    if not flips:
        return 1.0
    tail = sum(comb(flips, k) for k in range(min(improved, regressed) + 1)) / 2 ** flips
    return round(min(1.0, 2 * tail), 4)


def outcome(before, after):
    if before == after:
        return 'both_correct' if after else 'both_failed'
    return 'improved' if after else 'regressed'


def compare(base, new, groups, splits=None):
    base = {row['id']: row for row in base}
    rows, changes = [], Counter()
    splits = splits or {}
    by_group, by_calls, by_split = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
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
        for bucket in (by_group[group], by_calls[calls], by_split[splits.get(sid, 'khong_ro')]):
            bucket['samples'] += 1
            bucket['base_state_match'] += bool(left['state_match'])
            bucket['new_state_match'] += bool(item['state_match'])
            bucket[change] += 1
        rows.append({'id': sid, 'state_change': change, 'keyword_group': group, 'split': splits.get(sid, ''), 'gold_calls': len(expected),
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
    for bucket in by_split.values():
        bucket['mcnemar_exact_p'] = mcnemar_exact(bucket['improved'], bucket['regressed'])
    summary = {'base': totals([base[r['id']] for r in new]), 'new': totals(new), 'state_outcome_changes': dict(changes),
               'mcnemar_exact_p': mcnemar_exact(changes['improved'], changes['regressed']),
               'by_keyword_group': {k: dict(v) for k, v in sorted(by_group.items())},
               'by_gold_calls': {k: dict(v) for k, v in sorted(by_calls.items())},
               'by_split': {k: dict(v) for k, v in sorted(by_split.items())},
               'note': 'Paired local proxy (state hash / exact calls), not an official score. keyword_group comes from automatic keyword flags, not manual review.'}
    return rows, summary


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    base_root, new_root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    evidence = Path(sys.argv[4] if len(sys.argv) == 5 else KEYWORD_EVIDENCE)
    groups = keyword_groups(evidence) if evidence.exists() else {}
    manifest = Path(SPLIT_MANIFEST)
    splits = {sid: item['split'] for sid, item in json.loads(manifest.read_text()).items()} if manifest.is_file() else {}
    rows, summary = compare(jsonl(base_root / 'results.jsonl'), jsonl(new_root / 'results.jsonl'), groups, splits)
    summary.update(base_run=str(base_root), new_run=str(new_root), keyword_evidence=str(evidence) if groups else None)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['id'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ('base', 'new', 'state_outcome_changes', 'mcnemar_exact_p')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
