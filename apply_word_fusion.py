"""Replay the frozen selector on timed cache; no fitting or neural inference."""
import csv
import hashlib
import json
import time
from pathlib import Path
from asr.timed import atomic_json
from asr.word_fusion import VERSION,LanguageCounts,prepare_regions,select_path


def main():
    root=Path('outputs/timestamp_word_fusion')
    artifact=json.loads((root/'selector.json').read_text())
    if artifact['version']!=VERSION:raise ValueError('Selector version differs; refit explicitly')
    splits=json.loads((root/'split_manifest.json').read_text())
    rows=list(csv.DictReader(Path('outputs/zipformer_150m/asr_comparison.csv').open(encoding='utf-8-sig')))
    paths={}
    for line in Path('outputs/asr_timed_v1/manifest.jsonl').read_text().splitlines():
        row=json.loads(line)
        if 'error' in row:raise ValueError('Timed cache has errors')
        paths[row['id'],row['turn'],row['history_index'],row['model']]=row['cache']
    assert len(paths)==2*len(rows)
    language=LanguageCounts(artifact['language_counts']);started=time.time()
    temporary=root/'results.tmp'
    with temporary.open('w',encoding='utf-8') as output:
        for i,row in enumerate(rows,1):
            uid=(row['id'],row['turn'],row['history_index'])
            details={name:json.loads(Path(paths[uid+(name,)]).read_text()) for name in ('gipformer','zipformer')}
            g,z=details['gipformer']['words'],details['zipformer']['words']
            result=select_path(prepare_regions(g,z,artifact,language),g,z,artifact['switch_penalty'])
            key=hashlib.sha256('|'.join(uid).encode()).hexdigest()
            entry={**{k:row[k] for k in ('id','turn','history_index','audio')},'split':splits[row['id']]['split'],
                'fusion_text':result['text'],'gipformer_cache':paths[uid+('gipformer',)],
                'zipformer_cache':paths[uid+('zipformer',)],'fusion_detail':str(root/'decisions'/(key+'.json'))}
            atomic_json(entry['fusion_detail'],{**entry,'decisions':result['decisions']})
            output.write(json.dumps(entry,ensure_ascii=False)+'\n')
            if i%500==0 or i==len(rows):print('[FROZEN WORD FUSION]',i,'/',len(rows),flush=True)
    temporary.replace(root/'results.jsonl')
    atomic_json(root/'progress.json',{'processed':len(rows),'total':len(rows),'finished':True,
        'elapsed_seconds':round(time.time()-started,2),'mode':'frozen_selector_no_refit_no_neural_inference'})


if __name__=='__main__':main()
