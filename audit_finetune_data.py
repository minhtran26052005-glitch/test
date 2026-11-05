"""Audit generated text variants for the two fine-tuning uses; reads saved outputs only, no model or API.

ASR use (text that still needs TTS): rejected if manually rejected, confirmed spoken corruption, foreign script,
or empty spoken text. Agent use (reusing the seed's gold tool calls as label): only final-turn variants without
slot changes whose spoken tokens keep every "anchor" token, i.e. a token of a gold argument that the user
actually said in the reference. Variants that swap entities need new labels and environments.

Usage: python audit_finetune_data.py [OUT_DIR]
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from asr_spoken_text import spoken_tokens

GENERATED = Path('outputs/asr_datagen_api/generated.csv')
QUALITY = Path('outputs/asr_datagen_quality_20261104')
RUNS = ['outputs/benchmark_seed42_n1545', 'outputs/zipformer_150m/benchmark_seed42_n1545', 'outputs/dual_asr_qwen/benchmark_seed42_n1545']
FREE_TEXT = {'content', 'message', 'text', 'note', 'title', 'subject', 'body', 'description', 'label', 'question'}
FOREIGN = re.compile(r'[Ѐ-ӿ؀-ۿ֐-׿぀-ヿ一-鿿฀-๿]')
# Unit words come from spoken timestamps ("09:00" -> "chín giờ") and match the reference by accident.
STOP = set('anh em chị cho tôi mình của và với là thì có không cái này đó nhé nha giúp vào ra lên ở đi rồi được từ giờ phút giây ngày tháng năm'.split())


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def gold_labels(runs=RUNS):
    gold = {}
    for run in runs:
        path = Path(run) / 'results.jsonl'
        if path.is_file():
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line) if line.strip() else {}
                if 'expected_tool_calls' in row:
                    gold[row['id']] = row['expected_tool_calls']
    return gold


def argument_values(value, key=None):
    """(key, value) leaves of tool-call arguments."""
    if isinstance(value, dict):
        for name, item in value.items():
            yield from argument_values(item, name)
    elif isinstance(value, list):
        for item in value:
            yield from argument_values(item, key)
    elif value is not None and not isinstance(value, bool):
        yield key, value


def anchors(calls, reference):
    said = set(spoken_tokens(reference))
    tokens = set()
    for call in calls:
        for key, value in argument_values(call['arguments']):
            if key in FREE_TEXT:
                continue  # paraphrased message/reminder wording is expected; flagged separately as label_safe_free_text
            text = value if isinstance(value, str) else str(value)
            tokens |= {t for t in spoken_tokens(text) if t in said and t not in STOP and len(t) > 1}
    return tokens


def has_free_text(calls):
    return any(key in FREE_TEXT for call in calls for key, value in argument_values(call['arguments']) if isinstance(value, str))


def assess(row, gold, manual, corrupt):
    reasons = []
    spoken = row['spoken_text'] or ''
    verdict = manual.get(row['variant_id'])
    if not spoken.strip():
        reasons.append('empty_spoken_text')
    if FOREIGN.search(spoken) or FOREIGN.search(row['text']):
        reasons.append('foreign_script')
    if row['variant_id'] in corrupt:
        reasons.append('confirmed_spoken_corruption')
    if verdict == 'reject_text':
        reasons.append('manual_reject')
    asr = 'reject' if reasons else ('edit' if verdict == 'edit_text' else 'ok')
    slot_changes = json.loads(row['slot_changes'] or '[]')
    calls = gold.get(row['sample_id'])
    missing = []
    if row['turn'] != 'last':
        agent = 'history_paraphrase'  # usable as a paraphrased history turn; the final label is unchanged
    elif asr == 'reject' or verdict == 'edit_text':
        agent = 'reject'
    elif calls is None:
        agent = 'no_gold_label'
    elif slot_changes:
        agent = 'needs_new_label'  # an entity was swapped: the seed's environment and calls no longer apply
    else:
        kept = set(spoken_tokens(row['text'])) | set(spoken_tokens(spoken))  # same tokenizer as the reference side
        missing = sorted(anchors(calls, row['reference']) - kept)
        agent = 'needs_new_label' if missing else ('label_safe_free_text' if has_free_text(calls) else 'label_safe')
    return {'variant_id': row['variant_id'], 'seed_id': row['seed_id'], 'sample_id': row['sample_id'], 'turn': row['turn'],
            'split': row['source_original_split'], 'target_model': row['target_model'], 'domain': row['domain'],
            'asr_use': asr, 'asr_reasons': ';'.join(reasons), 'agent_use': agent, 'missing_anchor_tokens': ' '.join(missing),
            'manual_decision': verdict or '', 'gold_calls': len(calls) if calls is not None else '',
            'reference': row['reference'], 'text': row['text'], 'spoken_text': spoken}


def call_profile(calls):
    names = [call['name'] for call in calls]
    if not names:
        return 'khong_goi_tool'
    if 'sys.ask_missing_info' in names:
        return 'hoi_lai'
    return f'{min(len(names), 4)}{"+" if len(names) >= 4 else ""}_loi_goi'


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/finetune_data_audit')
    generated = csv_rows(GENERATED)
    gold = gold_labels()
    manual = {row['variant_id']: row['decision'] for row in csv_rows(QUALITY / 'reviewed_examples.csv')}
    corrupt = {row['variant_id'] for row in csv_rows(QUALITY / 'confirmed_spoken_corruption.csv')}
    rows = [assess(row, gold, manual, corrupt) for row in generated]
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'variants.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final = [row for row in rows if row['turn'] == 'last']
    safe_seeds = {row['sample_id'] for row in final if row['agent_use'].startswith('label_safe')}
    train_safe = [row for row in final if row['agent_use'].startswith('label_safe') and row['split'] == 'fit']
    profile_all = Counter(call_profile(calls) for calls in gold.values())
    profile_safe = Counter(call_profile(gold[sid]) for sid in safe_seeds)
    apps_all = Counter(app for calls in gold.values() for app in {c['name'].split('.')[0] for c in calls})
    apps_safe = Counter(app for sid in safe_seeds for app in {c['name'].split('.')[0] for c in gold[sid]})
    summary = {
        'variants': len(rows), 'seeds': len({row['seed_id'] for row in rows}),
        'asr_use': dict(Counter(row['asr_use'] for row in rows)),
        'asr_reject_reasons': dict(Counter(reason for row in rows for reason in row['asr_reasons'].split(';') if reason)),
        'agent_use_final_turn': dict(Counter(row['agent_use'] for row in final)),
        'history_paraphrase_variants': sum(row['agent_use'] == 'history_paraphrase' for row in rows),
        'final_turn_samples_total': len(gold), 'final_turn_samples_covered': len({row['sample_id'] for row in final}),
        'final_turn_samples_label_safe': len(safe_seeds),
        'label_safe_variants_in_fit_split': len(train_safe),
        'gold_profile_all_samples': dict(profile_all), 'gold_profile_label_safe_samples': dict(profile_safe),
        'apps_without_label_safe_variant': sorted(app for app in apps_all if app not in apps_safe and app != 'sys'),
        'note': 'Automatic screen. label_safe means the anchor tokens survived, not that meaning was verified; '
                'label_safe_free_text keeps the seed free-text arguments (message/reminder content), which may not match the paraphrase.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
