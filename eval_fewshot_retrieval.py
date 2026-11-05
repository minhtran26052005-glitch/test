"""Offline proxy for few-shot quality, from a saved benchmark; no model or devkit needed.

Builds a temporary pool from the run's reference transcripts and gold calls, queries it with each sample's
own ASR/normalized text (excluding the sample and its split group), and reports how often the retrieved
examples carry the same tool-name sequence / app set as the sample's gold. Compare with names_match of the agent.

Usage: python eval_fewshot_retrieval.py [RUN_DIR] [SPLIT_MANIFEST] [K]
"""
import json
import random
import sys
from pathlib import Path
from fewshot import FewShotIndex


def names(calls):
    return [call['name'] for call in calls]


def apps(calls):
    return sorted({call['name'].split('.')[0] for call in calls if not call['name'].startswith('sys.')})


def evaluate(rows, groups, k=3, seed=0, max_similarity=None):
    rows = [row for row in rows if 'expected_tool_calls' in row]
    pool = [{'id': row['id'], 'text': row['reference_text'], 'tools': [], 'tool_calls': row['expected_tool_calls']} for row in rows]
    index = FewShotIndex(pool)
    rng = random.Random(seed)
    counts = dict(top1_same_tools=0, any_same_tools=0, top1_same_apps=0, any_same_apps=0, random_same_tools=0)
    for row in rows:
        group = groups.get(row['id'])
        _, examples = index.select([row.get('normalized_text'), row.get('asr_text')], None, k,
                                   exclude_ids={row['id']}, exclude_groups={group} if group else set(), groups=groups,
                                   max_similarity=max_similarity)
        gold = row['expected_tool_calls']
        if examples:
            counts['top1_same_tools'] += names(examples[0]['tool_calls']) == names(gold)
            counts['top1_same_apps'] += apps(examples[0]['tool_calls']) == apps(gold)
        counts['any_same_tools'] += any(names(e['tool_calls']) == names(gold) for e in examples)
        counts['any_same_apps'] += any(apps(e['tool_calls']) == apps(gold) for e in examples)
        other = rng.choice([p for p in pool if p['id'] != row['id'] and (not group or groups.get(p['id']) != group)])
        counts['random_same_tools'] += names(other['tool_calls']) == names(gold)
    return {'samples': len(rows), 'k': k, **{key: round(100 * value / len(rows), 1) for key, value in counts.items()},
            'agent_names_match': round(100 * sum(bool(r.get('names_match')) for r in rows) / len(rows), 1),
            'unit': 'percent of samples with gold labels'}


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/benchmark_seed42_n1545')
    manifest = Path(sys.argv[2] if len(sys.argv) > 2 else 'outputs/timestamp_word_fusion/split_manifest.json')
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    groups = {sid: item['group'] for sid, item in json.loads(manifest.read_text()).items()} if manifest.is_file() else {}
    report = {'all_examples': evaluate(rows, groups, k),
              'without_near_copies_of_query': evaluate(rows, groups, k, max_similarity=0.8)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
