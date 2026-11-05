"""Assemble the hand-written synthetic agent data (outputs/synthetic_agent/authored/) into labeled records + TTS requests.

Three kinds, each with a label rule that needs no new environment (seeds are fit-split train samples, so the
replay in build_agent_sft.py runs every call in that sample's own devkit environment):
  paraphrase  same request said differently, every name/number/condition kept -> the seed's gold tool calls
              (replay can also verify the label's final state hash)
  missing     single-intent request with one required value removed -> sys.ask_missing_info(tool, [param]),
              the convention of the train labels (e.g. "Nhắc mình gọi cho mẹ nhé" -> ask reminders.create time)
  cancel      complete request followed by a withdrawal ("À mà thôi, khỏi làm gì hết") -> no tool call

Automatic checks on every authored sentence: the anchor tokens of the gold arguments (names, queries, places,
numbers the user said) must survive, and the numeric values said in the sentence must equal the source's.
Flagged records are written with check_ok=false and left out of the default SFT source.

Usage: python build_synthetic_agent.py [--out outputs/synthetic_agent]
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from asr_spoken_text import spoken_text, spoken_tokens
from audit_finetune_data import anchors, gold_labels
from prepare_asr_finetune_data import relative_audio, tts_label

AUTHORED = Path('outputs/synthetic_agent/authored')
MANIFEST = 'outputs/timestamp_word_fusion/split_manifest.json'
RUNS = ['outputs/benchmark_seed42_n1545', 'outputs/dual_asr_qwen/benchmark_seed42_n1545']
DIGITS = {'không': 0, 'một': 1, 'mốt': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'tư': 4, 'năm': 5, 'lăm': 5, 'sáu': 6,
          'bảy': 7, 'bẩy': 7, 'tám': 8, 'chín': 9}
SCALES = {'nghìn': 1000, 'ngàn': 1000, 'triệu': 10 ** 6, 'tỷ': 10 ** 9, 'tỉ': 10 ** 9}
NUMBER_WORDS = set(DIGITS) | set(SCALES) | {'mười', 'mươi', 'chục', 'trăm', 'linh', 'lẻ', 'rưỡi'}
# Two accepted tone-mark placements are the same word ('hủy'/'huỷ', 'hòa'/'hoà').
TONE_PLACEMENT = {'uỷ': 'ủy', 'uý': 'úy', 'uỳ': 'ùy', 'uỹ': 'ũy', 'uỵ': 'ụy', 'oà': 'òa', 'oá': 'óa', 'oả': 'ỏa', 'oã': 'õa',
                  'oạ': 'ọa', 'oè': 'òe', 'oé': 'óe', 'oẻ': 'ỏe', 'oẽ': 'õe', 'oẹ': 'ọe'}


def same_spelling(token):
    for variant, canonical in TONE_PLACEMENT.items():
        token = token.replace(variant, canonical)
    return token


def number_value(run):
    """Value of one run of Vietnamese number words, incl. colloquial forms:
    'hai lăm' = 25, 'hai triệu tám' = 2.8 million, 'nghìn rưỡi' = 1500, digit strings 'một tám năm sáu' = 1856."""
    total, group, num, previous, last_scale = 0, 0, 0, None, 1
    for token in run:
        if token in DIGITS:
            digit = DIGITS[token]
            if previous in ('mươi', 'mười'):
                num += digit
            elif previous in DIGITS:
                num = num * 10 + digit
            else:
                num = digit
        elif token == 'mười':
            num = 10
        elif token == 'mươi':
            num *= 10
        elif token == 'chục':
            num = num * 10 if num else 10  # 'năm chục' = 50, 'chục hột gà' = 10
        elif token == 'trăm':
            group, num = group + (num or 1) * 100, 0
        elif token in SCALES:
            last_scale = SCALES[token]
            total, group, num = total + ((group + num) or 1) * last_scale, 0, 0
        elif token == 'rưỡi':
            if last_scale > 1:
                total += last_scale // 2
            else:
                num += 0.5
        previous = token
    if len(run) >= 2 and run[-1] in DIGITS and run[-2] in SCALES:
        num = DIGITS[run[-1]] * SCALES[run[-2]] // 10  # 'hai triệu tám': the digit is a tenth of the scale
    return total + group + num


def numbers(text):
    """Counter of numeric values said in the text; punctuation separates runs ('thứ bảy, ba rưỡi' is 7 and 3.5)."""
    values = Counter()
    # split on punctuation followed by a space before spelling out, so '900.000' and '8:30' stay whole
    for segment in re.split(r'[,.;:!?…]+(?=\s|$)', text):
        tokens = spoken_tokens(segment)
        tokens = [t for i, t in enumerate(tokens)
                  if not (t == 'giờ' and i + 1 < len(tokens) and tokens[i + 1] == 'rưỡi')
                  and not (t in ('tỷ', 'tỉ') and i + 1 < len(tokens) and tokens[i + 1] == 'giá')]  # 'tỷ giá' = rate
        run = []
        for token in tokens + ['.']:
            if token in NUMBER_WORDS:
                run.append(token)
            elif run:
                # a lone 'không' is the negation word, not zero
                if any(t in DIGITS and t != 'không' or t in ('mười', 'trăm', 'chục') or t in SCALES for t in run):
                    values[number_value(run)] += 1
                run = []
    return values


def references(runs=RUNS):
    found = {}
    for run in runs:
        path = Path(run) / 'results.jsonl'
        if path.is_file():
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line) if line.strip() else {}
                if 'reference_text' in row:
                    found[row['id']] = row['reference_text']
    return found


def authored_lines(path):
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            index, *variants = line.split('|')
            yield int(index), [v.strip() for v in variants if v.strip()]


def check(text, source, calls, compare_numbers=True):
    said = {same_spelling(t) for t in spoken_tokens(text)}
    expected = {same_spelling(t) for t in anchors(calls, source) if t not in NUMBER_WORDS}  # numbers are compared by value
    missing = sorted(expected - said)
    number_diff = []
    if compare_numbers:
        a, b = numbers(source), numbers(text)
        difference = (a - b) + (b - a)
        difference.pop(1, None)  # 'một' is also the article ('một cái', 'một ghi chú'): adding/dropping it is not a value change
        number_diff = sorted(map(str, difference))
    return {'check_ok': not missing and not number_diff, 'missing_anchor_tokens': missing, 'number_mismatch': number_diff}


def build(gold, refs, splits):
    records = []
    seeds = json.loads((AUTHORED / 'paraphrase_seeds.json').read_text())['seeds']
    for path in sorted(AUTHORED.glob('paraphrase_*.txt')):
        for index, variants in authored_lines(path):
            sid = seeds[index]
            for k, text in enumerate(variants):
                records.append({'id': f'{sid}:paraphrase:{k}', 'sample_id': sid, 'kind': 'paraphrase', 'text': text,
                                'source_text': refs[sid], 'tool_calls': gold[sid], 'verify_state': True,
                                **check(text, refs[sid], gold[sid])})
    missing_seeds = json.loads((AUTHORED / 'missing_info_seeds.json').read_text())['seeds']
    for index, variants in authored_lines(AUTHORED / 'missing_info.txt'):
        sid, tool, param = missing_seeds[index]
        calls = [{'name': 'sys.ask_missing_info', 'arguments': {'tool_name': tool, 'params': [param]}}]
        # Train labels ask straight away for a missing scalar ("chuyển ít tiền qua ví cho Huy" -> only the ask): reads are dropped too.
        kept = [c for c in gold[sid] if c['name'] != tool and c['name'] not in ('contacts.search', 'smart_home.get_status')]
        for k, text in enumerate(variants):
            # The removed value must really be gone (no number left over from the source sentence for time/amount).
            leftover = sorted(map(str, numbers(text))) if param in ('time', 'amount', 'start', 'volume') else []
            records.append({'id': f'{sid}:missing:{k}', 'sample_id': sid, 'kind': 'missing', 'text': text,
                            'source_text': refs[sid], 'tool_calls': calls, 'verify_state': False, 'removed': param,
                            'check_ok': not kept and not leftover, 'missing_anchor_tokens': [], 'number_mismatch': leftover})
    phrases = [line for line in (AUTHORED / 'cancel_phrases.txt').read_text(encoding='utf-8').splitlines()
               if line.strip() and not line.startswith('#')]
    cancel_seeds = [s for s in sorted(gold) if splits.get(s) == 'fit' and s in refs and len(gold[s]) >= 2
                    and gold[s][-1]['name'] != 'sys.ask_missing_info'][:200:]
    for n, sid in enumerate(cancel_seeds[:150]):
        text = refs[sid].rstrip() + ' ' + phrases[n % len(phrases)]
        records.append({'id': f'{sid}:cancel:0', 'sample_id': sid, 'kind': 'cancel', 'text': text, 'source_text': refs[sid],
                        'tool_calls': [], 'verify_state': False, 'check_ok': True, 'missing_anchor_tokens': [], 'number_mismatch': []})
    for record in records:
        record['split'] = splits.get(record['sample_id'])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='outputs/synthetic_agent')
    parser.add_argument('--tts-target-model', choices=['gipformer', 'zipformer', 'both'], default='gipformer',
                        help='Vietnamese/domain replay defaults to Gip; English-focused requests are generated separately')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {sid: item['split'] for sid, item in json.loads(Path(MANIFEST).read_text()).items()}
    records = build(gold_labels(), references(), splits)
    if any(r['split'] != 'fit' for r in records):
        raise ValueError('Synthetic data must only use fit-split seeds')
    with (out / 'synthetic.jsonl').open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    with Path('outputs/asr_error_audit/utterances.csv').open(encoding='utf-8-sig', newline='') as handle:
        speaker = {row['id']: relative_audio(row['audio']) for row in csv.DictReader(handle) if row['turn'] == 'last'}
    with (out / 'tts_requests.jsonl').open('w', encoding='utf-8') as handle:
        for record in records:
            if record['check_ok']:
                spoken = spoken_text(record['text'])
                handle.write(json.dumps({'id': record['id'].replace(':', '_'), 'tts_text': spoken, 'text': tts_label(spoken, record['text']),
                                         'speaker_audio': speaker[record['sample_id']], 'speaker_text': record['source_text'],
                                         'sample_id': record['sample_id'], 'turn': 'last', 'source': 'tts_synthetic_agent',
                                         'kind': record['kind'], 'target_model': args.tts_target_model, 'split': 'train'}, ensure_ascii=False) + '\n')
    kept = [r for r in records if r['check_ok']]
    summary = {'records': len(records), 'check_ok': len(kept), 'by_kind': dict(Counter(r['kind'] for r in records)),
               'check_ok_by_kind': dict(Counter(r['kind'] for r in kept)), 'seed_samples': len({r['sample_id'] for r in records}),
               'apps_in_gold': len({c['name'].split('.')[0] for r in kept for c in r['tool_calls'] if not c['name'].startswith('sys.')}),
               'gold_call_profile': dict(Counter(min(len(r['tool_calls']), 4) for r in kept)),
               'flagged': [{k: r[k] for k in ('id', 'text', 'missing_anchor_tokens', 'number_mismatch')} for r in records if not r['check_ok']]}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'flagged'}, ensure_ascii=False, indent=2))
    for item in summary['flagged']:
        print('FLAG', item['id'], item['missing_anchor_tokens'], item['number_mismatch'], '|', item['text'])


if __name__ == '__main__':
    main()
