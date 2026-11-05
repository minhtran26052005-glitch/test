"""Compare keyword preservation of two ASRs and actual time-word fusion."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

from asr_datagen_audit import CATEGORIES,medical_group,semantic_text
from asr_error_audit import align,get_calls
from compare_asr_domains import (LABELS,candidate_occurrences,aligned_target,matches,
                                numeric_difference,write_csv)

ROOT=Path('outputs/timestamp_word_fusion')
MODELS=('gipformer','zipformer','fusion')


def main():
    original=list(csv.DictReader(Path('outputs/zipformer_150m/asr_comparison.csv').open(encoding='utf-8-sig')))
    fused={}
    for line in (ROOT/'results.jsonl').read_text().splitlines():
        row=json.loads(line);key=(row['id'],row['turn'],row['history_index'])
        assert key not in fused
        fused[key]=row
    assert len(fused)==len(original)==5036
    cases={r['id']:r for line in Path('data/train+public_test/train/train.jsonl').read_text().splitlines() if (r:=json.loads(line))}
    lexicon=json.loads(Path('outputs/asr_datagen_targets/lexicon.json').read_text())
    by_first=defaultdict(list)
    for key,spec in lexicon.items():by_first[spec['tokens'][0]].append((key,spec))
    evidence=[];terms=defaultdict(list);categories=defaultdict(dict);transcripts=[];decisions=[]
    for row in original:
        uid=(row['id'],row['turn'],row['history_index']);result=fused[uid]
        texts={'gipformer':row['gipformer_asr'],'zipformer':row['zipformer_asr'],'fusion':result['fusion_text']}
        detailed=json.loads(Path(result['fusion_detail']).read_text())
        sources=Counter(d['chosen_source'] for d in detailed['decisions'] if not d['agreement'] and d['chosen_words'])
        transcripts.append({**{k:row[k] for k in ('id','turn','history_index','reference','audio')},
            'split':result['split'],'gipformer':texts['gipformer'],'zipformer':texts['zipformer'],
            'fusion':texts['fusion'],'disagreement_regions':sum(not d['agreement'] for d in detailed['decisions']),
            'selected_gipformer_regions':sources['gipformer'],'selected_zipformer_regions':sources['zipformer'],
            'mixed_sources_in_audio':bool(sources['gipformer'] and sources['zipformer']),
            'uncertain_regions':sum(d['uncertain'] for d in detailed['decisions']),'decision_file':result['fusion_detail']})
        for decision in detailed['decisions']:
            decisions.append({'id':row['id'],'turn':row['turn'],'history_index':row['history_index'],'split':result['split'],
                **{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in decision.items()}})
        ref=semantic_text(row['reference']);alignments={}
        for model,text in texts.items():
            r,h,ops,_=align(ref,semantic_text(text));alignments[model]=(h,ops)
        local=defaultdict(lambda:{name:False for name in MODELS})
        for start,end,key,spec in candidate_occurrences([w for _,w in r],by_first):
            heard={name:aligned_target(h,ops,start,end) for name,(h,ops) in alignments.items()}
            bad={name:not matches(key,text) for name,text in heard.items()}
            gold=get_calls(cases[row['id']],int(row['history_index']) if row['turn']=='history' else None)
            entry={'id':row['id'],'turn':row['turn'],'history_index':row['history_index'],'split':result['split'],
                'category':spec['category'],'category_vi':LABELS[spec['category']],
                'keyword':spec['term'],'source':spec['source'],'reference_position':start,
                'reference_context':' '.join(w for _,w in r[max(0,start-5):end+5]),
                **{name+'_target':heard[name] for name in MODELS},
                **{name+'_flagged':bad[name] for name in MODELS},
                'fixed_vs_gipformer':bad['gipformer'] and not bad['fusion'],
                'fixed_vs_zipformer':bad['zipformer'] and not bad['fusion'],
                'regressed_vs_gipformer':not bad['gipformer'] and bad['fusion'],
                'regressed_vs_zipformer':not bad['zipformer'] and bad['fusion'],
                'one_source_preserved':bad['gipformer']!=bad['zipformer'],
                'recovered_available_keyword':bad['gipformer']!=bad['zipformer'] and not bad['fusion'],
                'fusion_new_flag_when_both_preserved':not bad['gipformer'] and not bad['zipformer'] and bad['fusion'],
                'fusion_preserved_when_both_flagged':bad['gipformer'] and bad['zipformer'] and not bad['fusion'],
                'error_bucket':('both_asr_need_domain_review' if bad['gipformer'] and bad['zipformer'] else 'fusion_selector_chose_wrong_available_source') if bad['fusion'] else 'keyword_preserved',
                'numeric_fusion_assessment':numeric_difference(key,heard['fusion'],spec['source']) if spec['category']=='gia_tri_so' else '',
                'reference':row['reference'],'gipformer':texts['gipformer'],'zipformer':texts['zipformer'],'fusion':texts['fusion'],
                'expected_calls_this_turn':json.dumps(gold,ensure_ascii=False),
                'decision_file':result['fusion_detail'],'audio':row['audio'],
                'assessment':'semantic_risk_screen_not_full_manual_accuracy_or_calling_causality'}
            evidence.append(entry);terms[key].append(entry)
            for name in MODELS:local[spec['category']][name]|=bad[name]
        for category,flags in local.items():categories[category][uid]=(result['split'],flags)
    out=ROOT/'comparison';out.mkdir(exist_ok=True)
    write_csv(out/'keyword_evidence.csv',evidence)
    write_csv(out/'transcripts.csv',transcripts)
    write_csv(out/'word_decisions.csv',decisions)
    fields=list(evidence[0])
    write_csv(out/'fixed_keywords.csv',[r for r in evidence if r['fixed_vs_gipformer'] or r['fixed_vs_zipformer']],fields)
    write_csv(out/'new_regressions.csv',[r for r in evidence if r['regressed_vs_gipformer'] or r['regressed_vs_zipformer']],fields)
    write_csv(out/'remaining_errors.csv',[r for r in evidence if r['fusion_flagged']],fields)
    write_csv(out/'needs_selector_fix.csv',[r for r in evidence if r['error_bucket']=='fusion_selector_chose_wrong_available_source'],fields)
    write_csv(out/'both_asr_data_candidates.csv',[r for r in evidence if r['error_bucket']=='both_asr_need_domain_review'],fields)
    targets=[]
    for key,rows in terms.items():
        for split in ['all','test']:
            selected=[r for r in rows if split=='all' or r['split']==split]
            if not selected:continue
            n=len(selected);spec=lexicon[key]
            item={'category':spec['category'],'category_vi':LABELS[spec['category']],'keyword':spec['term'],
                'source':spec['source'],'split':split,'mentions':n,
                **{name+'_flagged_mentions':sum(r[name+'_flagged'] for r in selected) for name in MODELS},
                **{name+'_flagged_percent':round(100*sum(r[name+'_flagged'] for r in selected)/n,2) for name in MODELS},
                **{metric:sum(r[metric] for r in selected) for metric in ['fixed_vs_gipformer','fixed_vs_zipformer','regressed_vs_gipformer','regressed_vs_zipformer','one_source_preserved','recovered_available_keyword','fusion_new_flag_when_both_preserved','fusion_preserved_when_both_flagged']},
                **{name+'_error_variants':json.dumps(Counter(r[name+'_target'] or '[mất cụm]' for r in selected if r[name+'_flagged']).most_common(6),ensure_ascii=False) for name in MODELS},
                'generation_focus':CATEGORIES[spec['category']]['gen']}
            targets.append(item)
    targets.sort(key=lambda r:(r['split'],r['category'],-r['fusion_flagged_mentions'],r['keyword']))
    write_csv(out/'keyword_summary.csv',targets)
    write_csv(out/'medical_keywords.csv',[{'medical_group':medical_group(r['keyword']),**r} for r in targets if r['category']=='thuoc_y_te'])
    category_rows=[]
    for category in CATEGORIES:
        for split in ['all','fit','validation','test']:
            for turn in ['all','last','history']:
                selected=[flags for uid,(part,flags) in categories[category].items() if (split=='all' or split==part) and (turn=='all' or uid[1]==turn)]
                if not selected:continue
                n=len(selected)
                category_rows.append({'category':category,'category_vi':LABELS[category],'split':split,'turn':turn,
                    'audio_with_keyword':n,
                    **{name+'_flagged_audio':sum(r[name] for r in selected) for name in MODELS},
                    **{name+'_flagged_percent':round(100*sum(r[name] for r in selected)/n,2) for name in MODELS},
                    'fixed_vs_gipformer_audio':sum(r['gipformer'] and not r['fusion'] for r in selected),
                    'fixed_vs_zipformer_audio':sum(r['zipformer'] and not r['fusion'] for r in selected),
                    'new_vs_gipformer_audio':sum(not r['gipformer'] and r['fusion'] for r in selected),
                    'new_vs_zipformer_audio':sum(not r['zipformer'] and r['fusion'] for r in selected)})
    write_csv(out/'category_summary.csv',category_rows)
    reviewed=list(csv.DictReader(Path('outputs/zipformer_150m/domain_comparison/reviewed_examples.csv').open(encoding='utf-8-sig')))
    review_output=[]
    for item in reviewed:
        result=fused[item['id'],item['turn'],item['history_index']]
        review_output.append({**item,'fusion':result['fusion_text'],'split':result['split'],
            'fusion_decision_file':result['fusion_detail'],'fusion_review_status':'actual_fusion_text_available_for_field_review'})
    write_csv(out/'reviewed_examples.csv',review_output)
    # Field-by-field language findings checked against the actual fused text.
    annotations=[
        ('train_0001','last','','một tiếng','sửa từ Gip','Ghép giữ một tiếng của Zip, thay một chuyến của Gip; vẫn còn lỗi dời ở câu này.'),
        ('train_0001','last','','Sếp Dũng','sửa từ Zip','Cùng câu đó, ghép giữ Sếp Dũng của Gip thay Xích Dũng của Zip: minh chứng bù lỗi ở từng vị trí.'),
        ('train_0029','last','','SJC','sửa từ Zip','Giữ mã SJC từ Gip, tránh SGC của Zip.'),
        ('train_0069','history','4','SCTV9','sửa từ Gip','Giữ SCTV chín từ Zip thay SCTV chính.'),
        ('train_0086','history','0','chín giờ','sửa từ Zip','Giữ giờ chín từ Gip thay đúng giờ không có con số.'),
        ('train_0138','last','','điều hòa','sửa từ Gip','Giữ điều hòa từ Zip thay đều vào.'),
        ('train_0039','last','','Minh','sửa từ Gip','Giữ tên Minh từ Zip. Phần nội dung tin nhắn trong cùng câu vẫn cần kiểm tra; không kết luận cả câu đúng.'),
        ('train_0280','last','','nhắn SMS','bù giữa các từ','Gip giữ nhắn nhưng thiếu SMS; Zip có SMS nhưng nhánh. Ghép giữ được nhắn SMS.'),
        ('train_0084','history','4','RON95','bù giữa các từ','Ghép giữ RON chín lăm; tương đương RON95, không coi thiếu chữ mươi là sai.'),
        ('train_0006','last','','một trăm hai mươi ba triệu','sửa từ Zip','Giữ đủ 123 triệu từ Gip thay một trăm hai triệu của Zip. Mẫu thuộc fit, không dùng làm bằng chứng holdout.'),
        ('train_0124','last','','sáu mươi phút','sửa từ Zip','Giữ sáu mươi phút của Gip, tránh mất mươi thành sáu phút. Mẫu thuộc fit.'),
        ('train_0144','last','','bốn mươi phần trăm','sửa từ Zip','Giữ bốn mươi phần trăm, tránh thành bốn phần trăm. Mẫu thuộc fit.'),
        ('train_1695','last','','vitamin D','sửa từ Gip','Giữ vitamin D của Zip thay vitamin T; lịch thuốc và các từ khác vẫn sai.'),
        ('train_1510','last','','chưa uống','sửa từ Zip','Giữ chưa uống của Gip, tránh chú uống của Zip. Tên thuốc ở cùng câu chưa được sửa đủ.'),
        ('train_1125','last','','thuốc bao tử','sửa từ Zip','Giữ thuốc bao tử của Gip; đoạn lịch thuốc sau đó vẫn sai.'),
        ('train_1474','last','','Omeprazole','cả hai sai, ghép chưa sửa','Chọn AMERAZON; không có chuỗi Omeprazole đúng trong hai nguồn để bộ chọn giữ lại.'),
        ('train_1343','last','','Clearwater Pharma','cả hai sai, ghép chưa sửa','Hai nguồn cùng CIA WATER PHẠM MA; ghép không phục hồi tên gốc.'),
        ('train_1430','last','','kẽm Brightpath','cả hai sai, ghép chưa sửa','Tên Brightpath vẫn mất/méo; cần data tên ngoại trong câu y tế.'),
        ('train_1510','last','','viên sắt','chọn sai nguồn','Zip đã giữ viên sắt nhưng bộ ghép vẫn chọn viên sạch từ Gip. Đây là lỗi bộ chọn, không phải thiếu đáp án trong hai ASR.'),
        ('train_0485','last','','viên sắt','chọn sai nguồn','Gip đúng viên sắt, ghép chọn men sắt của Zip. Cùng thuật ngữ sai ở cả hai hướng: chưa ổn định domain này.'),
        ('train_1703','last','','thuốc huyết áp','chọn sai nguồn','Zip có huyết áp nhưng bộ ghép bỏ thông tin này theo Gip. Cần cải thiện bộ chọn cho cụm y tế hiếm.'),
        ('train_0083','last','','ba mươi phần trăm','chọn sai nguồn','Zip đúng ba mươi, ghép vẫn chọn năm mươi từ Gip. Bù được nhiệt độ ở cuối câu nhưng chưa chọn đúng độ sáng.'),
        ('train_1715','last','','đừng','cả hai sai, ghép chưa sửa','Hai ASR đều đổi đừng thành được; ghép vẫn không giữ đúng phủ định.'),
        ('train_0963','last','','à không','cờ tự động cần rà','Bộ so sánh bỏ từ đệm à nên đánh dấu phục hồi khi chỉ còn không. Câu sửa ý vẫn méo: không coi đây là ca sửa ngữ nghĩa chắc chắn.'),
    ]
    reviewed_actual=[]
    for sid,turn,index,keyword,verdict,note in annotations:
        found=[r for r in evidence if (r['id'],r['turn'],r['history_index'])==(sid,turn,index) and semantic_text(r['keyword'])==semantic_text(keyword)]
        if not found:raise ValueError(f'Missing reviewed keyword: {(sid,turn,index,keyword)}')
        reviewed_actual.append({**found[0],'language_review_verdict':verdict,'language_review_note':note,
            'review_method':'purposive_language_review_of_actual_saved_text_no_audio_no_causal_calling_claim'})
    write_csv(out/'language_review.csv',reviewed_actual)
    summary={'audios':len(transcripts),'audio_splits':dict(Counter(r['split'] for r in transcripts)),
        'keyword_occurrences':len(evidence),'distinct_keywords':len(terms),
        'mixed_disagreement_sources_audio':sum(r['mixed_sources_in_audio'] for r in transcripts),
        'word_regions':len(decisions),'uncertain_regions':sum(r['uncertain'] for r in decisions),
        'language_review_keywords':len(reviewed_actual),
        'metrics':{},'categories_all':[r for r in category_rows if r['split']=='all' and r['turn']=='all'],
        'categories_test':[r for r in category_rows if r['split']=='test' and r['turn']=='all'],
        'note':'Same reference keyword inventory and equivalent-spoken matching as earlier audit. Counts are field-preservation flags; not WER, not verified causal calling errors. Fit/validation/test are retrospective engineering partitions; no reference enters fusion inference.'}
    for split in ['all','test']:
        selected=[r for r in evidence if split=='all' or r['split']==split]
        metrics={name+'_flagged':sum(r[name+'_flagged'] for r in selected) for name in MODELS}
        metrics.update({metric:sum(r[metric] for r in selected) for metric in ['fixed_vs_gipformer','fixed_vs_zipformer','regressed_vs_gipformer','regressed_vs_zipformer','one_source_preserved','recovered_available_keyword','fusion_new_flag_when_both_preserved','fusion_preserved_when_both_flagged']})
        metrics['occurrences']=len(selected)
        metrics['selector_error_candidates']=sum(r['error_bucket']=='fusion_selector_chose_wrong_available_source' for r in selected)
        metrics['both_asr_remaining_candidates']=sum(r['error_bucket']=='both_asr_need_domain_review' for r in selected)
        summary['metrics'][split]=metrics
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# Kết quả ghép theo timestamp ở mức từ','',
        f"Đã ghép đủ {len(transcripts)} audio từ token/timestamp/log-probability gốc; cả hai lần nhận dạng khớp transcript benchmark cũ. Bộ ghép chọn từ/cụm tại từng vùng thời gian, có đổi nguồn trong cùng câu; không đưa hai câu cho Qwen viết lại.",'',
        'Bộ chọn gồm hiệu chỉnh điểm từng model, điểm ngữ cảnh n-gram từ phần fit và chọn đường qua các vùng căn thời gian. Các cụm số, tên Latin, thuốc, phủ định và ngày tương đối được ràng buộc để không trộn mảnh tùy ý. Các lựa chọn đều có nguồn trong hai bản ASR.','',
        '## Phạm vi và cách đọc','',
        f"Audio theo phân vùng: {summary['audio_splits']}. Chỉ fit/validation dùng để hiệu chỉnh/chọn tham số; test được báo riêng. Tập train từng được khảo sát từ trước, nên đây là kiểm tra kỹ thuật có giữ lại dữ liệu, không phải benchmark mới chưa từng thấy.",'',
        'Bảng đếm audio có cụm từ khóa sai/nghi sai; mẫu số chỉ gồm audio chứa từ khóa đã khảo sát. Loại khác hoa/thường, số/chữ tương đương và các alias đã công bố. Nhóm/cụm có thể chồng lấp. Tên phiên âm/căn biên còn cần rà bằng ngôn ngữ; không coi đây là % lỗi calling đã chứng minh.','']
    for section,key in [('Toàn bộ audio','categories_all'),('Phần test giữ lại','categories_test')]:
        lines += ['## '+section,'','| Nhóm | Audio | Gip có cờ | Zip có cờ | Ghép có cờ |','|---|---:|---:|---:|---:|']
        for r in summary[key]:
            if 'phat_hien_them' in r['category']:continue
            lines.append(f"| {r['category_vi']} | {r['audio_with_keyword']} | {r['gipformer_flagged_audio']} | {r['zipformer_flagged_audio']} | {r['fusion_flagged_audio']} |")
        lines.append('')
    lines+=['## Từng từ khóa và quyết định','',
        '- keyword_summary.csv: từng cụm, số lượt giữ/sai của ba nhánh, số sửa và lỗi mới so với từng ASR; có toàn tập và test.',
        '- keyword_evidence.csv: từng lần xuất hiện, reference và đoạn cả ba nhánh, câu đầy đủ, tool kỳ vọng và file quyết định.',
        '- medical_keywords.csv: toàn bộ cụm thuốc/y tế, không chỉ những tên đã có lỗi.',
        '- fixed_keywords.csv / new_regressions.csv / remaining_errors.csv: lỗi được sửa / lỗi mới so với một ASR / lỗi còn lại.',
        '- needs_selector_fix.csv: một ASR đã giữ đúng nhưng bộ ghép chọn sai; ưu tiên sửa bộ chọn, không tự quy thành thiếu data ASR.',
        '- both_asr_data_candidates.csv: cả hai ASR còn có cờ, dùng rà/chọn domain cần gen data; vẫn cần kiểm tra biến thể phiên âm/căn biên.',
        '- word_decisions.csv: timestamp, hai ứng viên, nguồn được chọn, điểm chọn, cờ chưa chắc và ràng buộc từng vùng.',
        '- reviewed_examples.csv: 60 trường đã rà trước đây, thêm transcript ghép thực tế để kiểm tra.',
        '- language_review.csv: các kết luận ngôn ngữ đã kiểm tra trực tiếp trên từ khóa và transcript ghép thực tế, gồm sửa được và chọn sai nguồn.',
        '- transcripts.csv: đủ 5.036 audio, reference và ba transcript, phân vùng và nguồn từ.',
        '',f"Có {summary['mixed_disagreement_sources_audio']} audio lấy các vùng bất đồng từ cả hai model trong cùng audio. Số này chỉ chứng minh có ghép từ, không phải số audio được sửa đúng.",
        '', '## Kết luận ở các ca đã rà', '',
        '- train_0001 thuộc test: ghép lấy một tiếng từ Zip (Gip nghe một chuyến), đồng thời lấy Sếp Dũng từ Gip (Zip nghe Xích Dũng). Đây là bù lỗi ở hai vị trí trong cùng một audio.',
        '- train_0280 thuộc test: Gip giữ nhắn, Zip giữ SMS, ghép được nhắn SMS; tên Minh trong câu đó vẫn sai thành mình.',
        '- Các ví dụ số được giữ từ Gip: 123 triệu, 60 phút, 40 phần trăm; các mẫu minh họa này thuộc phần fit. Hiệu quả trên test được thể hiện ở bảng riêng phía trên.',
        '- Chưa giải quyết tên Omeprazole, Clearwater Pharma, Brightpath khi cả hai nguồn sai.',
        '- Bộ chọn vẫn sai: train_1510 và train_0485 chọn sai viên sắt theo hai hướng khác nhau; train_1703 bỏ huyết áp dù Zip giữ được; train_0083 chọn sai 50 dù Zip giữ 30.',
        '- Thuốc trên phần test chỉ có 13 audio chứa cụm mục tiêu: fusion còn 9 audio có cờ so với Zip 8. Không kết luận cải thiện thuốc nói chung từ con số toàn tập 19/18/17.',
        '- Sàng lọc có dương tính giả: train_0963 bị coi là sửa à không do quy tắc bỏ từ đệm, nhưng chưa xác minh giữ đúng ý sửa câu. Xem ghi chú ngôn ngữ trong language_review.csv.',
        '- Trong quá trình rà đã sửa lỗi cấu trúc căn từ: timestamp phát token có thể lệch hơn một giây dù cùng từ. Bản trước ngưỡng cứng tạo lặp từ; bản v2 dùng mốc từ giống nhau với chi phí thời gian mềm. Bản chẩn đoán cũ lưu riêng tại outputs/timestamp_word_fusion_v1/.',
        '', 'Kết quả dừng ở ASR ghép trước normalization. Chưa chạy lại tool calling cho nhánh timestamp; không suy tỷ lệ calling từ cờ từ khóa.']
    (out/'BAO_CAO_GHEP_TU.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if not k.startswith('categories')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
