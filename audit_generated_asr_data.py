#!/usr/bin/env python3
"""Freeze a local generated-text snapshot and prepare semantic review; no API calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re

from asr_datagen_audit import semantic_text
from generate_asr_data_api import norm, validate


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})


def canonical(value):
    text = semantic_text(value)
    for pattern, replacement in [(r'\bs m s\b|\bét em ét\b|\bét mờ ét\b|\bés em és\b', 'sms'),
                                  (r'\be mail\b', 'email'), (r'\bki lô mét\b', 'km')]:
        text = re.sub(pattern, replacement, text)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('outputs/asr_datagen_api'))
    p.add_argument('--output', type=Path, default=Path('outputs/asr_datagen_quality_20261104'))
    p.add_argument('--random-n', type=int, default=120)
    p.add_argument('--targeted-n', type=int, default=80)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Output snapshot already exists; choose a new --output')
    args.output.mkdir(parents=True)
    files = sorted((args.input / 'results').glob('*.json'))
    rows, invalid, hashes, prompts = [], [], {}, {}
    group = defaultdict(list)
    strategies = Counter()
    for file in files:
        data = file.read_bytes()
        hashes[file.name] = hashlib.sha256(data).hexdigest()
        result = json.loads(data)
        sid = result['seed_id']
        prompt = json.loads((args.input / 'prompts' / f'{sid}.json').read_text())
        prompts[sid] = prompt['input']
        strategies[prompt['input']['zip_english_plan']['strategy']] += 1
        try:
            validate(json.dumps({'variants': result['variants']}, ensure_ascii=False), 10, prompt)
        except Exception as exc:
            invalid.append({'seed_id': sid, 'error': str(exc)})
        for i, v in enumerate(result['variants'], 1):
            ref = result['reference']
            text, speech = canonical(v['text']), canonical(v['spoken_text'])
            ratio = SequenceMatcher(None, text.split(), speech.split(), autojunk=False).ratio()
            missing = []
            for h in prompt['input']['keyword_risk_hints']:
                term = canonical(h['correct_term'])
                if term not in canonical(ref):
                    continue
                if any(norm(h['correct_term']) == norm(c['from']) for c in v['slot_changes']):
                    continue
                if term not in text and term not in speech:
                    missing.append({'category': h['category'], 'term': h['correct_term']})
            changes = [dict(operation=tag, text=' '.join(text.split()[a:b]),
                            spoken=' '.join(speech.split()[c:d]))
                       for tag,a,b,c,d in SequenceMatcher(None,text.split(),speech.split(),autojunk=False).get_opcodes()
                       if tag != 'equal']
            row = {'variant_id': f'{sid}_v{i:02d}', 'seed_id': sid, 'generator_model': result['model'],
                   'target_model': v['target_model'], 'domain': v['domain'], 'reference': ref,
                   'text': v['text'], 'spoken_text': v['spoken_text'], 'focus_terms': v['focus_terms'],
                   'slot_changes': v['slot_changes'], 'semantic_warnings': v['semantic_warnings'],
                   'missing_keyword_candidates': missing,
                   'text_spoken_exact_after_format_normalization': text == speech,
                   'text_spoken_similarity_screen': round(ratio, 4), 'text_spoken_diffs': changes,
                   'reference_similarity_screen': round(SequenceMatcher(None,canonical(ref).split(),text.split(),autojunk=False).ratio(),4),
                   'zip_strategy': prompt['input']['zip_english_plan']['strategy'],
                   'context_file': str(args.input / 'prompts' / f'{sid}.json')}
            rows.append(row)
            group[sid].append(row)
    spoken_duplicates = []
    for sid, values in group.items():
        by_speech = defaultdict(list)
        for r in values:
            by_speech[canonical(r['spoken_text'])].append(r['variant_id'])
        for speech, ids in by_speech.items():
            if len(ids) > 1:
                spoken_duplicates.append({'seed_id': sid, 'variant_ids': ids, 'canonical_spoken': speech})
    rng = random.Random(20261104)
    representative = rng.sample(rows, min(args.random_n, len(rows)))
    chosen = {r['variant_id'] for r in representative}
    selected = []

    def pick(candidates, count, reason):
        candidates = [r for r in candidates if r['variant_id'] not in chosen]
        for r in candidates[:count]:
            chosen.add(r['variant_id'])
            selected.append({**r, 'sample_type': 'targeted', 'selection_reason': reason})

    # Keep all current medical examples, then inspect conversion/number/negation/model/name risks.
    pick([r for r in rows if r['domain'] == 'thuoc_y_te'], 40, 'medical_terms_and_taken_status')
    pick(sorted([r for r in rows if not r['text_spoken_exact_after_format_normalization']],
                key=lambda r:r['text_spoken_similarity_screen']), 12, 'text_spoken_conversion_risk')
    pick([r for r in rows if any(m['category']=='gia_tri_so' for m in r['missing_keyword_candidates'])],8,'numeric_value_risk')
    pick([r for r in rows if any(m['category']=='phu_dinh_dieu_kien' for m in r['missing_keyword_candidates'])],8,'negation_condition_risk')
    gemma = [r for r in rows if r['generator_model'].startswith('gemma')]
    rng.shuffle(gemma)
    pick(gemma,8,'fallback_model_coverage')
    pick([r for r in rows if r['slot_changes']],max(0,args.targeted_n-len(selected)),'foreign_name_slot_changes')
    selected = selected[:args.targeted_n]
    review = [{**r, 'sample_type':'simple_random', 'selection_reason':'uniform_random_variant_seed_20261104'}
              for r in representative] + selected
    for i,r in enumerate(review,1):
        r.update(review_index=i, decision='', severity='', issue_types='', reviewer_notes='')
    write_csv(args.output/'all_variants_screen.csv',rows)
    write_csv(args.output/'review.csv',review)
    write_csv(args.output/'text_spoken_candidates.csv',[r for r in rows if not r['text_spoken_exact_after_format_normalization']],list(rows[0]))
    write_csv(args.output/'keyword_candidates.csv',[r for r in rows if r['missing_keyword_candidates']],list(rows[0]))
    write_csv(args.output/'spoken_duplicates.csv',spoken_duplicates,['seed_id','variant_ids','canonical_spoken'])
    # Freeze the exact rows and context used in this review while generation continues.
    (args.output/'snapshot.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (args.output/'review_input.jsonl').write_text(''.join(json.dumps({**r,'preceding_history':prompts[r['seed_id']]['preceding_history'],
                   'zip_plan':prompts[r['seed_id']]['zip_english_plan']},ensure_ascii=False)+'\n' for r in review))
    stats = {'created_at':datetime.now(timezone.utc).isoformat(),'snapshot_seeds':len(files),'snapshot_variants':len(rows),
             'seed_result_hashes':hashes,'structural_validation_failures':invalid,
             'by_generator':dict(Counter(r['generator_model'] for r in rows)),
             'by_target':dict(Counter(r['target_model'] for r in rows)),
             'by_domain':dict(Counter(r['domain'] for r in rows)),
             'zip_strategy_seeds':dict(strategies),
             'variants_with_old_warnings':sum(bool(r['semantic_warnings']) for r in rows),
             'text_spoken_conversion_candidates':sum(not r['text_spoken_exact_after_format_normalization'] for r in rows),
             'keyword_preservation_candidates':sum(bool(r['missing_keyword_candidates']) for r in rows),
             'spoken_duplicate_groups_within_seed':len(spoken_duplicates),
             'random_review_n':len(representative),'targeted_review_n':len(selected),
             'limitations':['Screen mismatches are review candidates, not verified semantic errors.',
                'This is a partial prefix of a still-running generation run, not the full planned corpus.',
                'The random review is uniform over variants in this snapshot; targeted review is deliberately biased.',
                'No audio/TTS or downstream calling evaluation is performed.']}
    (args.output/'snapshot_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in stats.items() if k!='seed_result_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
