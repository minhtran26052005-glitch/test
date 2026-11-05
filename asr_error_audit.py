"""Audit every cached train ASR transcript; tags are review hints, not ground truth."""
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from num2words import num2words
from pathlib import Path

ROOT = Path('data/train+public_test/train')
BENCH = Path('outputs/benchmark_seed42_n1545')
OUT = Path('outputs/asr_error_audit')
TOKEN_RE = re.compile(r'[^\W_]+', re.UNICODE)
NUMBER_WORDS = set('không một hai ba bốn tư năm lăm sáu bảy bẩy tám chín mười mươi trăm nghìn ngàn triệu tỷ tỉ rưỡi nửa kém giờ phút giây sáng trưa chiều tối khuya mai nay hôm qua ngày tháng năm tuần thứ số độ phần trăm mét phân lít kwh kg gram đồng tiền'.split())
NEGATION = set('không chẳng chưa đừng chớ khỏi thôi hủy huỷ bỏ ngưng dừng vẫn trừ ngoại'.split())
ENGLISH = set('work from home team meeting meet zoom google microsoft excel powerpoint python design check cancel confirm booking book club speaking english email mail chat app video workshop online offline wifi internet cloud storage laptop smartphone marketing trading foods hotel hotels resort cafe coffee service bank banking tax invoice delivery voucher cashback point points reward sale sales project manager managerment data analyst science learning fitness yoga studio premium standard business economy silver gold sport smartphone netflix youtube tiktok facebook instagram zalo spotify talkai chatgpt tiktok paypal youtube grab uber spotify youtube chrome windows android ios ipad iphone samsung lg rong viettel mobifone vinaphone omeprazole loratadine paracetamol panadol amoxicillin ibuprofen metformin insulin vitamin collagen aspirin'.split())
SIMPLE_NUMBER_RE = re.compile(r'(?<![\w.,:/-])\d{1,3}(?![\w.,:/-])')

@lru_cache(maxsize=1000)
def vietnamese_number(value):
    return num2words(int(value), lang='vi')

def expand_simple_numbers(text):
    def replace(match):
        value=match.group(0)
        if len(value)>1 and value.startswith('0'):
            return value
        return vietnamese_number(value)
    return SIMPLE_NUMBER_RE.sub(replace,text)

PRONOUN_CAPITALS = set('anh chị cô chú bác em con tôi mình ông bà bố mẹ dì cậu mợ thầy người trừ nếu khi rồi còn thì và như tới bữa hôm mai năm nhà đó đang được không đi quận thành phố đường phường'.split())


def tokenise(text):
    text=unicodedata.normalize('NFC',text)
    return [(m.group(0),m.group(0).casefold()) for m in TOKEN_RE.finditer(text)]


def align(reference,hypothesis):
    r=tokenise(reference);h=tokenise(hypothesis);n=len(r);m=len(h)
    d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0]=i
    for j in range(m+1): d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(r[i-1][1]!=h[j-1][1]))
    ops=[];i=n;j=m
    while i or j:
        if i and j and r[i-1][1]==h[j-1][1] and d[i][j]==d[i-1][j-1]:
            ops.append(('equal',i-1,j-1));i-=1;j-=1
        elif i and j and d[i][j]==d[i-1][j-1]+1:
            ops.append(('substitution',i-1,j-1));i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:
            ops.append(('deletion',i-1,j));i-=1
        else:
            ops.append(('insertion',i,j-1));j-=1
    return r,h,list(reversed(ops)),d[n][m]


def entities(text,calls):
    """Reference arguments are only hints and are not used as ASR input."""
    med=set()
    for call in calls:
        if call.get('name')=='medication.add':
            value=call.get('arguments',{}).get('name','')
            med.update(word for _,word in tokenise(value) if len(word)>1 and word not in {'thuốc'})
    for m in re.finditer(r'\bthuốc\s+([A-Z][a-zA-Z0-9-]{3,})',text):
        med.add(m.group(1).casefold())
    return med


def reference_tags(r,h,blocks,med):
    tags=set();refwords=[r[i] for op,i,j in blocks if op in ('substitution','deletion')]
    hypwords=[h[j] for op,i,j in blocks if op in ('substitution','insertion')]
    ref_lower={w for _,w in refwords}; hyp_lower={w for _,w in hypwords}
    if any(any(ch.isdigit() for ch in surface) or lower in NUMBER_WORDS for surface,lower in refwords+hypwords): tags.add('số_thời_gian_đơn_vị_nghi_vấn')
    if (ref_lower|hyp_lower)&NEGATION: tags.add('phủ_định_điều_kiện_nghi_vấn')
    if ref_lower&med: tags.add('tên_thuốc_nghi_vấn')
    if ref_lower&ENGLISH: tags.add('tiếng_anh_thương_hiệu_nghi_vấn')
    if any((len(surface)>=2 and surface.isascii() and surface.isupper() and any(c.isalpha() for c in surface)) or (any(c.isdigit() for c in surface) and any(c.isalpha() for c in surface)) for surface,lower in refwords): tags.add('viết_tắt_mã_nghi_vấn')
    if any(i>0 and r[i][0][:1].isupper() and r[i][1] not in PRONOUN_CAPITALS for op,i,j in blocks if op in ('substitution','deletion')): tags.add('tên_riêng_địa_danh_nghi_vấn')
    if not tags: tags.add('từ_khác')
    return sorted(tags)


def get_calls(case,index=None):
    if index is None: return case['label']['tool_calls']
    hist=case.get('history',[])
    for turn in hist[index+1:]:
        if turn.get('role')=='user': break
        if turn.get('role')=='assistant': return turn.get('tool_calls',[])
    return []


def names(calls, catalog):
    apps=[]
    for call in calls:
        name=call.get('name','')
        if name=='sys.ask_missing_info': name=call.get('arguments',{}).get('tool_name',name)
        app=name.split('.',1)[0]
        if app and app!='sys' and app not in apps: apps.append(app)
    groups=sorted({catalog[a] for a in apps if a in catalog})
    return apps,groups


def context(tokens,first,last):
    return ' '.join(surface for surface,_ in tokens[max(0,first-4):min(len(tokens),last+5)])


def records():
    rows=[json.loads(x) for x in (BENCH/'results.jsonl').read_text().splitlines() if x.strip()]
    cases={x['id']:x for x in map(json.loads,(ROOT/'train.jsonl').open(encoding='utf-8'))}
    for row in rows:
        sid=row['id'];case=cases[sid];folder=BENCH/'samples'/sid
        def cache(stage):
            p=folder/(stage+'.cache.json')
            return json.loads(p.read_text(encoding='utf-8'))['value'] if p.is_file() else None
        yield sid,'last',None,case['audio'],case['audio_text'],cache('asr_gipformer'),get_calls(case)
        for index,turn in enumerate(case.get('history',[])):
            if turn.get('role')=='user' and turn.get('audio') and 'audio_text' in turn:
                yield sid,'history',index,turn['audio'],turn['audio_text'],cache(f'history_{index}_gipformer'),get_calls(case,index)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    catalog={a['id']:a.get('group','không_xác_định') for a in json.loads((ROOT/'train_tools_catalog.json').read_text())['apps']}
    utterances=[];errors=[];stats=defaultdict(Counter);tagclips=defaultdict(set);top=Counter();missing=[]
    for sid,stage,index,audio,ref,hyp,calls in records():
        if hyp is None:
            missing.append({'id':sid,'stage':stage,'index':index,'audio':audio});continue
        apps,groups=names(calls,catalog); med=entities(ref,calls)
        r,h,ops,dist=align(ref,hyp);counts=Counter(x[0] for x in ops); key=f'{sid}:{stage}:{index if index is not None else "last"}'
        blocks=[];current=[]
        for op in ops:
            if op[0]=='equal':
                if current: blocks.append(current); current=[]
            else: current.append(op)
        if current: blocks.append(current)
        nr,nh,nops,ndist=align(expand_simple_numbers(ref),expand_simple_numbers(hyp))
        alltags=set(); error_phrases=[]
        for block in blocks:
            tags=reference_tags(r,h,block,med);alltags.update(tags)
            rt=[r[i][0] for op,i,j in block if op in ('substitution','deletion')]
            ht=[h[j][0] for op,i,j in block if op in ('substitution','insertion')]
            typ=Counter(op for op,_,_ in block); ref_span=' '.join(rt);hyp_span=' '.join(ht)
            refpos=[i for op,i,j in block if op in ('substitution','deletion')];hyppos=[j for op,i,j in block if op in ('substitution','insertion')]
            if ref_span and hyp_span and any(char.isdigit() for char in ref_span) and [w for _,w in tokenise(expand_simple_numbers(ref_span))]==[w for _,w in tokenise(expand_simple_numbers(hyp_span))]:
                tags=sorted(set(tags)|{'có_thể_chỉ_khác_cách_viết_số'})
                alltags.update(tags)
            ref_at=refpos[0] if refpos else min(block[0][1],max(len(r)-1,0));hyp_at=hyppos[0] if hyppos else min(block[0][2],max(len(h)-1,0))
            err={'id':sid,'turn':stage,'history_index': '' if index is None else index,'audio':str(ROOT/audio),'apps':'; '.join(apps),'domains':'; '.join(groups),'error_type':'+'.join(x for x in ('substitution','deletion','insertion') if typ[x]),'reference_span':ref_span,'asr_span':hyp_span,'reference_context':context(r,ref_at,refpos[-1] if refpos else ref_at),'asr_context':context(h,hyp_at,hyppos[-1] if hyppos else hyp_at),'suspected_tags':'; '.join(tags),'substitutions':typ['substitution'],'deletions':typ['deletion'],'insertions':typ['insertion']}
            errors.append(err);error_phrases.append(f'{ref_span or "∅"} → {hyp_span or "∅"}')
            if len(rt)==1 and len(ht)==1: top[(rt[0].casefold(),ht[0].casefold())]+=1
        for tag in alltags: tagclips[tag].add(key)
        utterances.append({'id':sid,'turn':stage,'history_index':'' if index is None else index,'audio':str(ROOT/audio),'apps':'; '.join(apps),'domains':'; '.join(groups),'reference':ref,'asr':hyp,'reference_tokens':len(r),'substitutions':counts['substitution'],'deletions':counts['deletion'],'insertions':counts['insertion'],'word_errors':dist,'wer_percent':round(dist*100/len(r),2) if r else '', 'number_expanded_ref_words':len(nr),'number_expanded_word_errors':ndist,'number_expanded_wer_percent':round(ndist*100/len(nr),2) if nr else '', 'suspected_tags':'; '.join(sorted(alltags)),'error_spans':' | '.join(error_phrases)})
        for dimension,values in [('turn',[stage]),('app',apps or ['không_có_nhãn_tool']),('domain',groups or ['không_có_nhãn_tool']),(f'app_{stage}',apps or ['không_có_nhãn_tool']),(f'domain_{stage}',groups or ['không_có_nhãn_tool'])]:
            for value in values:
                s=stats[(dimension,value)];s['clips']+=1;s['clips_with_error']+=bool(dist);s['reference_words']+=len(r);s['substitutions']+=counts['substitution'];s['deletions']+=counts['deletion'];s['insertions']+=counts['insertion'];s['word_errors']+=dist;s['number_expanded_reference_words']+=len(nr);s['number_expanded_word_errors']+=ndist
    def write(name, rows, fields):
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write('utterances.csv',utterances,list(utterances[0]))
    write('error_spans.csv',errors,list(errors[0]))
    domain_rows=[]
    for (dimension,value),s in sorted(stats.items()):
        domain_rows.append({'dimension':dimension,'name':value,**dict(s),'wer_percent':round(100*s['word_errors']/s['reference_words'],2) if s['reference_words'] else None, 'number_expanded_wer_percent':round(100*s['number_expanded_word_errors']/s['number_expanded_reference_words'],2) if s['number_expanded_reference_words'] else None})
    write('domain_summary.csv',domain_rows,['dimension','name','clips','clips_with_error','reference_words','substitutions','deletions','insertions','word_errors','number_expanded_reference_words','number_expanded_word_errors','wer_percent','number_expanded_wer_percent'])
    for tag,filename in {'tiếng_anh_thương_hiệu_nghi_vấn':'review_tieng_anh_thuong_hieu.csv','viết_tắt_mã_nghi_vấn':'review_viet_tat_ma.csv','tên_thuốc_nghi_vấn':'review_ten_thuoc.csv','tên_riêng_địa_danh_nghi_vấn':'review_ten_rieng_dia_danh.csv','số_thời_gian_đơn_vị_nghi_vấn':'review_so_thoi_gian_don_vi.csv','phủ_định_điều_kiện_nghi_vấn':'review_phu_dinh_dieu_kien.csv','có_thể_chỉ_khác_cách_viết_số':'review_dinh_dang_so.csv'}.items():
        write(filename,[e for e in errors if tag in e['suspected_tags'].split('; ')],list(errors[0]))
    tag_rows=[]
    for tag in sorted(tagclips):
        for turn in ('last','history','all'):
            selected=[u for u in utterances if tag in u['suspected_tags'].split('; ') and (turn=='all' or u['turn']==turn)]
            tag_rows.append({'suspected_tag':tag,'turn':turn,'utterances':len(selected)})
    write('tag_summary.csv',tag_rows,['suspected_tag','turn','utterances'])
    top_rows=[{'reference':a,'asr':b,'count':n} for (a,b),n in top.most_common()]
    write('frequent_substitutions.csv',top_rows,['reference','asr','count'])
    result={'total_transcripts':len(utterances),'last':sum(u['turn']=='last' for u in utterances),'history':sum(u['turn']=='history' for u in utterances),'missing':missing,'error_spans':len(errors),'suspected_tag_utterance_counts':{k:len(v) for k,v in sorted(tagclips.items())},'note':'WER theo token: NFC, chữ thường, bỏ dấu câu; chưa chuẩn hóa ITN; number_expanded_wer chỉ thử đổi số nguyên 0–999 đơn lẻ thành chữ, bỏ qua số đầu 0, số điện thoại, ngày/giờ dạng dấu và số thập phân, không phải WER chính thức; nhãn nhóm là heuristic, có thể chồng lấp và chứa dương tính giả; domain dựa trên tool nhãn, không đánh giá ASR theo giọng hay nhân quả.'}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
