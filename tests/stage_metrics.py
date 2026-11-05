import json, re, sys, unicodedata
from pathlib import Path

def tokens(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold(), flags=re.UNICODE)

def distance(ref,hyp):
    prev=list(range(len(hyp)+1))
    for i,r in enumerate(ref,1):
        cur=[i]
        for j,h in enumerate(hyp,1):
            cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(r!=h)))
        prev=cur
    return prev[-1]

def compute(pairs):
    edit=ref_count=char_edits=chars=exact=0
    for ref,hyp in pairs:
        r,h=tokens(ref),tokens(hyp)
        edit+=distance(r,h);ref_count+=len(r);exact+=r==h
        rc,hc=' '.join(r),' '.join(h)
        char_edits+=distance(rc,hc);chars+=len(rc)
    return {'n':len(pairs),'word_errors':edit,'reference_words':ref_count,'wer_percent':round(100*edit/ref_count,2) if ref_count else None, 'normalized_text_exact':exact,'normalized_text_exact_percent':round(100*exact/len(pairs),2) if pairs else None, 'character_errors':char_edits,'reference_characters':chars,'cer_percent':round(100*char_edits/chars,2) if chars else None}

b=Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/benchmark_seed42_n100'); rows=[json.loads(x) for x in (b/'results.jsonl').read_text().splitlines()]; ids={x['id'] for x in rows}; data={x['id']:x for x in map(json.loads,open('data/train+public_test/train/train.jsonl')) if x['id'] in ids}
asr=[]; norm=[]; hist=[]; detail=[]
for row in rows:
 sid=row['id']; item=data[sid]; path=b/'samples'/sid
 read=lambda filename:json.loads((path/filename).read_text())['value']
 reference=item['audio_text']
 rawfile=path/'asr_gipformer.cache.json'; normfile=path/'normalize.cache.json'
 raw=read(rawfile.name) if rawfile.is_file() else None
 normalized=read(normfile.name) if normfile.is_file() else None
 rawerr=distance(tokens(reference),tokens(raw)) if raw is not None else None
 normerr=distance(tokens(reference),tokens(normalized)) if normalized is not None else None
 if raw is not None: asr.append((reference,raw))
 if normalized is not None: norm.append((reference,normalized))
 detail.append({'id':sid,'reference_word_count':len(tokens(reference)),'asr_word_errors':rawerr,'normalized_word_errors':normerr,'normalized_exact_calls':row['exact_calls'],'state_match':row['state_match'],'names_match':row['names_match']})
 for i,t in enumerate(item.get('history',[])):
  filename=f'history_{i}_gipformer.cache.json'
  if t.get('role')=='user' and t.get('audio') and 'audio_text' in t and (path/filename).is_file():
   hist.append((t['audio_text'],read(filename)))

tool_calls=0
rejected=0
done_false=0
agent_turns=0
invalid_decisions=0
cases_invalid=0
for row in rows:
    trace_path=b/'samples'/row['id']/'trace.jsonl'
    trace=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()] if trace_path.is_file() else []
    errors=[entry for entry in trace if entry['stage']=='agent_error']
    invalid_decisions+=len(errors)
    cases_invalid+=bool(errors)
    agent_turns+=sum(entry['stage']=='agent' for entry in trace)
    for entry in trace:
        if entry['stage']=='tool':
            tool_calls+=1
            observation=entry.get('observation',{})
            rejected+=observation.get('ok') is False
            done_false+=observation.get('done') is False
agent={'samples':len(rows), 'completed':sum('error' not in row for row in rows),
       'step_limit_errors':sum('Agent chưa hoàn tất sau 12 bước' in row.get('error','') for row in rows),
       'agent_model_turns':agent_turns, 'invalid_decisions':invalid_decisions, 'samples_with_invalid_decision':cases_invalid,
       'executed_calls':tool_calls, 'official_schema_rejections':rejected, 'tool_observations_done_false':done_false,
       'exact_tool_names':sum(row['names_match'] for row in rows), 'exact_calls':sum(row['exact_calls'] for row in rows),
       'state_match':sum(row['state_match'] for row in rows)}
report={'description':'WER/CER vi mô theo token từ Unicode; chuẩn hóa NFC, chữ thường, bỏ dấu câu; không chuyển số chữ sang chữ số. Điểm normalizer so sánh với transcript gốc, không phải nhãn chuẩn hóa.', 'asr_last_utterance':compute(asr),'normalized_last_utterance':compute(norm),'asr_history':compute(hist),'tool_agent':agent,'normalizer_comparison':{'lower_word_errors':sum(d['normalized_word_errors']<d['asr_word_errors'] for d in detail if d['normalized_word_errors'] is not None and d['asr_word_errors'] is not None),'equal_word_errors':sum(d['normalized_word_errors']==d['asr_word_errors'] for d in detail if d['normalized_word_errors'] is not None and d['asr_word_errors'] is not None),'higher_word_errors':sum(d['normalized_word_errors']>d['asr_word_errors'] for d in detail if d['normalized_word_errors'] is not None and d['asr_word_errors'] is not None)},'samples':detail}
(b/'stage_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False,indent=2))
