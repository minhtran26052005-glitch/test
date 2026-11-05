#!/usr/bin/env python3
"""Publish the language review already recorded for a frozen generation snapshot."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from audit_generated_asr_data import canonical, write_csv

OUT = Path('outputs/asr_datagen_quality_20261104')
DECISIONS = {'pass_text':'Đạt ở mức văn bản', 'edit_text':'Cần sửa nhẹ', 'reject_text':'Cần loại hoặc viết lại trước TTS'}
FIXES = {
    3: 'Hãy đặt báo thức vào sáu rưỡi sáng mai, rồi gọi cho Minh giúp tôi.',
    9: 'Mai có mưa thì huỷ lịch họp team rồi nhắn SMS báo cô giáo Mai; còn không mưa thì đặt ô tô bốn chỗ từ nhà tới chỗ hẹn đó, đón trước giờ hẹn nửa tiếng.',
    13: 'Bật tivi lên kênh VTV6 giúp tôi.',
    26: 'Quanh đây có chỗ nào bán bún bò không?',
    29: 'Em thanh toán tiền mạng bằng ví điện tử giúp chị; nếu ví thiếu thì dùng ngân hàng, xong nhắn SMS cho Lan báo chị đã đóng rồi nhé.',
    46: 'Nếu mai mưa thì huỷ lịch đá bóng và nhắn SMS báo Minh đồng nghiệp; nếu không mưa thì gọi ô tô bốn chỗ từ nhà tới chỗ hẹn đó, đón trước giờ hẹn nửa tiếng.',
    70: 'Chuyển cho Alice Thompson khoản tiền đúng bằng hoá đơn mạng tháng này từ tài khoản thanh toán; không đủ thì lấy từ tiết kiệm nhé.',
    78: 'Em ơi, đặt báo thức năm giờ sáng mai và chuyển ví cho anh Tuấn chút tiền giúp tôi.',
    83: 'Bạn trả tiền mạng bằng ví điện tử giúp tôi, ví không đủ thì trừ ngân hàng; xong nhắn SMS cho Sếp Dũng báo tôi đã đóng rồi nhé.',
    89: 'Nạp điện thoại năm mươi nghìn vào số không chín năm không năm ba sáu không ba sáu bảy hộ em Linh nha.',
    92: 'Nếu mai mưa thì huỷ lịch khám răng rồi nhắn SMS báo bố; không mưa thì gọi ô tô bốn chỗ từ nhà đến chỗ hẹn đó, đón trước giờ hẹn nửa tiếng.',
    108: 'Nếu giá bán vàng SJC hôm nay trên một trăm hai mươi tám triệu thì chín giờ sáng mai nhắc tôi đi bán; chưa tới thì chuyển hai triệu từ tài khoản thanh toán cho Trần Văn Tuấn để góp mua vàng.',
    170: 'Mở tivi lên kênh SCTV9 giúp tôi.',
    172: 'Chỉ khi ví đủ tiền thì trả tiền điện bằng ví và nhắn Bố báo đã đóng; nếu ví thiếu thì trả bằng ngân hàng, rồi đặt nhắc tôi tám giờ tối nay nạp tiền vào ví.',
    185: 'Nếu mai mưa thì huỷ lịch đá bóng... À thôi, cứ để lịch đó, chỉ xem giúp mình mai có mưa không nhé.',
}


def main():
    original = [json.loads(l) for l in (OUT/'review_input.jsonl').read_text().splitlines()]
    extras = [json.loads(l) for l in (OUT/'extra_candidates.jsonl').read_text().splitlines()]
    medical = [json.loads(l) for l in (OUT/'extra_medical.jsonl').read_text().splitlines()]
    annotations = json.loads((OUT/'manual_annotations.json').read_text())
    rows = original + [{**r,'review_index':i,'sample_type':'targeted_extra',
                         'selection_reason':'unexpected_script_or_damaged_acronym'} for i,r in enumerate(extras,201)]
    rows += [{**r,'review_index':i,'sample_type':'targeted_extra',
              'selection_reason':'complete_medical_keyword_coverage'} for i,r in enumerate(medical,233)]
    assert len({r['variant_id'] for r in rows}) == len(rows)
    assert set(annotations) == {str(r['review_index']) for r in rows}
    for r in rows:
        i = r['review_index']
        r.update(annotations[str(i)])
        r['decision_vi'] = DECISIONS[r['decision']]
        r['suggested_text'] = FIXES.get(i, '')
        r['suggested_spoken_text'] = ''
        if 201 <= i <= 232:
            s = r['spoken_text']
            s = re.sub('Если', 'Nếu', s, flags=re.I)
            s = re.sub(r'\bE-M-S\b|\bét em ếch\b', 'ét em ét', s, flags=re.I)
            s = re.sub(r'\bV vê\b', 'vê tê vê', s)
            s = s.replace('ét جي xi', 'SJC')
            assert s != r['spoken_text']
            r['suggested_spoken_text'] = s
        r['suggestions_status'] = 'proposal_only_not_applied_to_raw_results'
        r['source_task_tools'] = json.loads(Path(r['context_file']).read_text())['input']['source_task_tools']
    fields = ['review_index','sample_type','selection_reason','variant_id','seed_id','generator_model','target_model',
              'domain','reference','text','spoken_text','slot_changes','decision','decision_vi','severity','issue_types',
              'reviewer_notes','suggested_text','suggested_spoken_text','suggestions_status','semantic_warnings',
              'context_file','source_task_tools']
    write_csv(OUT/'reviewed_examples.csv',rows,fields)
    for filename, decision in [('passed_reviewed.csv','pass_text'),('needs_edit_reviewed.csv','edit_text'),
                               ('rejected_reviewed.csv','reject_text')]:
        write_csv(OUT/filename,[r for r in rows if r['decision']==decision],fields)
    write_csv(OUT/'confirmed_spoken_corruption.csv',[r for r in rows if 201<=r['review_index']<=232],fields)
    random_rows = [r for r in rows if r['sample_type']=='simple_random']
    targeted = [r for r in rows if r['sample_type']!='simple_random']
    snapshot = [json.loads(l) for l in (OUT/'snapshot.jsonl').read_text().splitlines()]
    input_stats = json.loads((OUT/'snapshot_summary.json').read_text())
    spoken_groups = defaultdict(list)
    for r in snapshot:
        spoken_groups[canonical(r['spoken_text'])].append(r)
    duplicates = [{'canonical_spoken':s, 'variant_ids':[r['variant_id'] for r in rs], 'copies':len(rs)}
                  for s,rs in spoken_groups.items() if len(rs)>1]
    write_csv(OUT/'global_spoken_duplicates.csv',duplicates,['canonical_spoken','variant_ids','copies'])
    swaps = [c['to'] for r in snapshot for c in r['slot_changes']]
    med_rows = [r for r in rows if re.search(r'\b(?:thuốc|uống|viên sắt|tảo|dầu cá)\b',r['reference'],re.I)]
    by_model = {}
    for model in sorted({r['generator_model'] for r in random_rows}):
        ss=[r for r in random_rows if r['generator_model']==model]
        by_model[model]={'n':len(ss),'decisions':dict(Counter(r['decision'] for r in ss))}
    summary = {'snapshot_seeds':input_stats['snapshot_seeds'],'snapshot_variants':len(snapshot),
               'manual_review_total':len(rows),'random_review_n':len(random_rows),
               'random_review_decisions':dict(Counter(r['decision'] for r in random_rows)),
               'random_review_percentages':{d:round(sum(r['decision']==d for r in random_rows)/len(random_rows)*100,2) for d in DECISIONS},
               'targeted_review_n':len(targeted),'targeted_review_decisions':dict(Counter(r['decision'] for r in targeted)),
               'medical_review_n':len(med_rows),'medical_review_decisions':dict(Counter(r['decision'] for r in med_rows)),
               'medical_source_seeds':sorted({r['seed_id'] for r in med_rows}),
               'confirmed_spoken_corruption':32,'unexpected_script_rows':11,
               'damaged_acronym_rows':21,'global_unique_spoken':len(spoken_groups),
               'global_duplicate_excess':len(snapshot)-len(spoken_groups),
               'name_swap_variants':len(swaps),'distinct_replacement_names':len(set(swaps)),
               'zip_strategy_seeds':input_stats['zip_strategy_seeds'],
               'random_review_by_generator':by_model,
               'old_warning_on_random_rejected':sum(bool(r['semantic_warnings']) for r in random_rows if r['decision']=='reject_text'),
               'verified_issue_counts_in_review':dict(Counter(t for r in rows for t in r['issue_types'])),
               'limits':['120 uniform random variants describe the reviewed snapshot, not a verified pass rate over all 7270 rows.',
                         '132 targeted variants deliberately over-sample known risks; never pool their percentages with the random sample.',
                         'Ratings are a language review by the assistant, without a second independent annotator.',
                         'No audio or TTS pronunciation was evaluated; equivalent number formats are not errors.',
                         'Generation continues; all counts above use the frozen snapshot.',
                         'Generator sample sizes and domain mixes differ; model ranking is not supported.']}
    (OUT/'quality_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
