"""Re-score cached ASR with symmetric spoken normalization, without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from asr_error_audit import align, get_calls, names
from asr_spoken_text import VERSION, raw_tokens, spoken_text


def score(reference: str, prediction: str) -> tuple[dict, list[dict]]:
    r, h, ops, errors = align(reference, prediction)
    counts = Counter(op for op, _, _ in ops)
    stats = {'utterances': 1, 'reference_words': len(r), 'substitutions': counts['substitution'],
             'deletions': counts['deletion'], 'insertions': counts['insertion'],
             'word_errors': errors, 'exact_utterances': int(errors == 0)}
    blocks = []
    current = []
    for op, i, j in ops:
        if op == 'equal':
            if current:
                blocks.append(current)
                current = []
        else:
            current.append((op, i, j))
    if current:
        blocks.append(current)
    result = []
    for block in blocks:
        ref = ' '.join(r[i][0] for op, i, j in block if op != 'insertion')
        hyp = ' '.join(h[j][0] for op, i, j in block if op != 'deletion')
        result.append({'reference_span': ref, 'asr_span': hyp,
                       'error_type': '+'.join(key for key in ('substitution', 'deletion', 'insertion') if any(op == key for op, _, _ in block))})
    return stats, result


def finish(counter: Counter) -> dict:
    result = dict(counter)
    n = counter['reference_words']
    result['wer_percent'] = round(100 * counter['word_errors'] / n, 4) if n else None
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path('outputs/benchmark_seed42_n1545'))
    parser.add_argument('--data', type=Path, default=Path('data/train+public_test/train'))
    parser.add_argument('--output', type=Path, default=Path('outputs/asr_spoken_eval'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {item['id']: item for item in map(json.loads, (args.data / 'train.jsonl').open(encoding='utf-8'))}
    runs = [json.loads(line) for line in (args.benchmark / 'results.jsonl').read_text().splitlines() if line.strip()]
    ids = [item['id'] for item in runs]
    if len(set(ids)) != len(ids) or any(sid not in cases for sid in ids):
        raise ValueError('Benchmark has duplicate/unknown ids')
    catalog = {app['id']: app.get('group', 'unknown') for app in json.loads((args.data / 'train_tools_catalog.json').read_text())['apps']}
    stats = defaultdict(Counter)
    domains = defaultdict(Counter)
    rows, errors, missing = [], [], []
    profiles = {'raw': lambda s: ' '.join(raw_tokens(s)),
                'spoken_numbers': spoken_text,
                'spoken_context_aliases': lambda s: spoken_text(s, contextual_aliases=True)}
    for sid in ids:
        case = cases[sid]
        turns = [('last', None, case['audio'], case['audio_text'], 'asr_gipformer')]
        turns += [('history', index, turn['audio'], turn['audio_text'], f'history_{index}_gipformer')
                  for index, turn in enumerate(case.get('history', []))
                  if turn.get('role') == 'user' and turn.get('audio') and 'audio_text' in turn]
        for turn, index, audio, reference, cache in turns:
            cache_path = args.benchmark / 'samples' / sid / (cache + '.cache.json')
            if not cache_path.is_file():
                missing.append(str(cache_path))
                continue
            prediction = json.loads(cache_path.read_text())['value']
            apps, groups = names(get_calls(case, index), catalog)
            row = {'id': sid, 'turn': turn, 'history_index': '' if index is None else index,
                   'audio': str(args.data / audio), 'apps': '; '.join(apps), 'domains': '; '.join(groups),
                   'reference': reference, 'asr': prediction}
            for profile, normalize in profiles.items():
                ref, hyp = normalize(reference), normalize(prediction)
                counts, spans = score(ref, hyp)
                stats[(turn, profile)].update(counts)
                stats[('all', profile)].update(counts)
                for dimension, values in [('app', apps or ['no_labeled_tool']), ('domain', groups or ['no_labeled_tool'])]:
                    for value in values:
                        domains[(turn, profile, dimension, value)].update(counts)
                row[profile + '_reference_words'] = counts['reference_words']
                row[profile + '_word_errors'] = counts['word_errors']
                row[profile + '_wer_percent'] = round(100 * counts['word_errors'] / counts['reference_words'], 4) if counts['reference_words'] else None
                if profile != 'raw':
                    row[profile + '_reference'] = ref
                    row[profile + '_asr'] = hyp
                if profile == 'spoken_numbers':
                    for span in spans:
                        errors.append({'id': sid, 'turn': turn, 'history_index': row['history_index'],
                                       'audio': row['audio'], 'apps': row['apps'], 'domains': row['domains'],
                                       **span, 'reference_spoken': ref, 'asr_spoken': hyp,
                                       'reference_original': reference, 'asr_original': prediction})
            row['format_only_difference'] = row['raw_word_errors'] > 0 and row['spoken_numbers_word_errors'] == 0
            rows.append(row)
    summary = {
        'version': VERSION, 'num2words_version': version('num2words'),
        'normalizer_sha256': hashlib.sha256(Path('asr_spoken_text.py').read_bytes()).hexdigest(),
        'benchmark': str(args.benchmark), 'sample_count': len(ids), 'transcript_count': len(rows), 'missing': missing,
        'metrics': {turn: {profile: finish(stats[(turn, profile)]) for profile in profiles} for turn in ('last', 'history', 'all')},
        'format_only_utterances': {turn: sum(row['format_only_difference'] for row in rows if turn == 'all' or row['turn'] == turn) for turn in ('last', 'history', 'all')},
        'word_error_count_changes': {
            'decreased': sum(row['spoken_numbers_word_errors'] < row['raw_word_errors'] for row in rows),
            'equal': sum(row['spoken_numbers_word_errors'] == row['raw_word_errors'] for row in rows),
            'increased': sum(row['spoken_numbers_word_errors'] > row['raw_word_errors'] for row in rows)
        },
        'normalization_changed_utterances': sum(row['spoken_numbers_reference'] != ' '.join(raw_tokens(row['reference'])) or row['spoken_numbers_asr'] != ' '.join(raw_tokens(row['asr'])) for row in rows),
        'note': 'Symmetric text rules; no model inference or hypothesis-selected reference. Raw baseline preserved. Spoken WER is a local diagnostic, not an official competition score. Context aliases are reported separately. No dialect/English/name/negation corrections. Numeric words and the WER denominator may increase after expansion.'
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(args.output / 'utterances.csv', rows)
    write_csv(args.output / 'normalization_changed.csv', [row for row in rows if row['spoken_numbers_reference'] != ' '.join(raw_tokens(row['reference'])) or row['spoken_numbers_asr'] != ' '.join(raw_tokens(row['asr']))])
    write_csv(args.output / 'format_only_utterances.csv', [row for row in rows if row['format_only_difference']])
    write_csv(args.output / 'error_spans.csv', errors)
    domain_rows = [{'turn': turn, 'profile': profile, 'dimension': dim, 'name': name, **finish(count)}
                   for (turn, profile, dim, name), count in sorted(domains.items())]
    write_csv(args.output / 'domain_summary.csv', domain_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
