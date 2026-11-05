"""Build the few-shot pool (outputs/fewshot/pool.jsonl) from labeled train data; no model is run.

Each record keeps the final user request (reference transcript), the last history turns, the sample's
available tools and its gold tool calls. These are training labels used as examples for *other* samples:
fewshot.select_examples always excludes the sample itself, and its group when a split manifest is given.

Usage: python build_fewshot_pool.py [--data-root data/train+public_test] [--out outputs/fewshot/pool.jsonl]
"""
import argparse
import json
import os
from pathlib import Path

HISTORY_TAIL = 2


def turn_text(turn):
    return turn.get('text') or turn.get('audio_text') or ''


def pool_record(item, tools):
    tail = [{'role': turn.get('role'), 'text': turn_text(turn)} for turn in item.get('history', [])[-HISTORY_TAIL:] if turn_text(turn)]
    return {'id': item['id'], 'text': item['audio_text'], 'history_tail': tail, 'tools': tools,
            'tool_calls': item['label']['tool_calls']}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT', 'data/train+public_test'))
    parser.add_argument('--out', default='outputs/fewshot/pool.jsonl')
    args = parser.parse_args()
    source = Path(args.data_root) / 'train'
    items = [json.loads(line) for line in (source / 'train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    records = []
    for item in items:
        state = source / 'train_environment' / item['id'] / 'state.json'
        tools = json.loads(state.read_text(encoding='utf-8'))['_meta']['tools'] if state.is_file() else []
        records.append(pool_record(item, tools))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({'records': len(records), 'out': str(out),
                      'with_tools': sum(bool(r['tools']) for r in records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
