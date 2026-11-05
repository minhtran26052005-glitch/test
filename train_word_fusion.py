"""Fit small calibration/selection models, then freeze and fuse cached audio.

References are confined to split/training/evaluation code in this file. The
inference module asr.word_fusion receives only the two ASR results and artifact.
"""
import csv
import hashlib
import json
import re
import time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from asr.word_fusion import (VERSION, NUMBER, LanguageCounts, align_words,
    normal, word_features, region_features, probability, prepare_regions, select_path)
from asr.timed import atomic_json
from asr_datagen_audit import semantic_text
from asr_error_audit import align
from compare_asr_domains import candidate_occurrences, aligned_target, matches

ROOT=Path('outputs/timestamp_word_fusion')
SOURCE=Path('outputs/zipformer_150m/asr_comparison.csv')


def fit_logistic(features,labels,regularization=4.):
    x=np.asarray(features,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    mean=x.mean(axis=0);scale=x.std(axis=0)
    mean[0]=0;scale[0]=1;scale[scale<1e-6]=1
    z=(x-mean)/scale;coef=np.zeros(z.shape[1]);coef[0]=np.log((y.sum()+1)/(len(y)-y.sum()+1))
    ridge=np.eye(z.shape[1])*regularization;ridge[0,0]=.01
    for _ in range(35):
        pred=1/(1+np.exp(-np.clip(z@coef,-30,30)))
        weight=np.maximum(pred*(1-pred),1e-5)
        gradient=z.T@(pred-y)+ridge@coef
        hessian=z.T@(z*weight[:,None])+ridge
        delta=np.linalg.solve(hessian,gradient)
        coef-=delta
        if np.linalg.norm(delta)<1e-5:break
    raw=coef/scale;raw[0]-=float(np.dot(coef,mean/scale))
    return raw.tolist()


def split_cases(rows):
    # Group conversations and numeric variants of sufficiently long final
    # commands. Short generic turns do not join unrelated conversations.
    groups={};last={r['id']:r for r in rows if r['turn']=='last'}
    for sid,row in last.items():
        tokens=semantic_text(row['reference']).split()
        template=re.sub(r'(?:\b(?:'+ '|'.join(sorted(NUMBER,key=len,reverse=True))+r')\b\s*)+', '<num> ', ' '.join(tokens)).strip()
        group=template if len(tokens)>=8 else sid
        value=int(hashlib.sha256(('timed-fusion-seed42|'+group).encode()).hexdigest()[:8],16)%100
        groups[sid]={'group':hashlib.sha256(group.encode()).hexdigest()[:16],
                     'split':'fit' if value<70 else ('validation' if value<85 else 'test')}
    return groups


def word_reference_map(reference,words):
    # Keep an explicit map from normalized hypothesis token to source word.
    tokens=[];owners=[]
    for index,word in enumerate(words):
        pieces=semantic_text(word['text']).split()
        tokens.extend(pieces);owners.extend([index]*len(pieces))
    r,h,ops,_=align(semantic_text(reference),' '.join(tokens))
    positions=defaultdict(set);equal=Counter();totals=Counter(owners)
    for op,i,j in ops:
        if op!='deletion':
            owner=owners[j]
            if op!='insertion':positions[owner].add(i)
            if op=='equal':equal[owner]+=1
    labels=[float(equal[index]==totals[index] and totals[index]>0) for index in range(len(words))]
    return [w for _,w in r],positions,labels


def expected_region(column,maps,gip,zip_):
    positions=set()
    for name in ('gipformer','zipformer'):
        for index in column[name]:positions.update(maps[name][1][index])
    ref=maps['gipformer'][0]
    return ' '.join(ref[min(positions):max(positions)+1]) if positions else ''


def reference_cost(expected,heard):
    r,h,ops,_=align(expected,semantic_text(heard))
    cost=0.
    for op,i,j in ops:
        if op=='equal':continue
        word=r[i][1] if op!='insertion' else h[j][1]
        cost+=3. if word in NUMBER|{'đừng','chưa','không','đã','trước','sau','kém'} else 1.
    return cost


def keyword_evaluator(lexicon):
    by_first=defaultdict(list)
    for key,spec in lexicon.items():by_first[spec['tokens'][0]].append((key,spec))
    def errors(reference,text):
        ref=semantic_text(reference);r,h,ops,_=align(ref,semantic_text(text))
        count=total=0
        for start,end,key,spec in candidate_occurrences([word for _,word in r],by_first):
            total+=1
            if not matches(key,aligned_target(h,ops,start,end)):
                count+=3 if spec['category'] in {'thuoc_y_te','gia_tri_so','phu_dinh_dieu_kien','thoi_gian_quan_he'} else 1
        return count,total
    return errors


def main():
    ROOT.mkdir(parents=True,exist_ok=True);started=time.time()
    rows=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
    manifest=[json.loads(line) for line in Path('outputs/asr_timed_v1/manifest.jsonl').read_text().splitlines() if line.strip()]
    if len(manifest)!=2*len(rows) or any('error' in r for r in manifest):
        raise ValueError('Timed cache is incomplete; finish cache_timed_asr.py first')
    paths={(r['id'],r['turn'],r['history_index'],r['model']):r['cache'] for r in manifest}
    splits=split_cases(rows);atomic_json(ROOT/'split_manifest.json',splits)
    records=[]
    for row in rows:
        uid=(row['id'],row['turn'],row['history_index'])
        record={'row':row,'split':splits[row['id']]['split']}
        for name in ('gipformer','zipformer'):
            record[name]=json.loads(Path(paths[uid+(name,)]).read_text())
        records.append(record)
    fit=[r for r in records if r['split']=='fit'];validation=[r for r in records if r['split']=='validation']
    print('[FUSION SPLIT]',dict(Counter(r['split'] for r in records)),flush=True)
    calibrators={};training_counts={}
    for name in ('gipformer','zipformer'):
        xs=[];ys=[]
        for record in fit:
            words=record[name]['words'];_,_,labels=word_reference_map(record['row']['reference'],words)
            xs.extend(word_features(word) for word in words);ys.extend(labels)
        calibrators[name]=fit_logistic(xs,ys)
        training_counts[name]={'words':len(ys),'fraction_exact_reference_units':sum(ys)/len(ys)}
    language=LanguageCounts()
    for record in fit:language.add(semantic_text(record['row']['reference']).split())
    artifact={'version':VERSION,'calibrators':calibrators,'language_counts':language.export(),
        'timestamp_tolerance':.2,'switch_penalty':.1,'selector':None,
        'fit_metadata':{'split_method':'Conversation + normalized numeric final-command template; sha256 seed42, 70/15/15',
            'split_counts_audio':dict(Counter(r['split'] for r in records)),
            'word_calibration':training_counts,'references_used_only_in_fit_and_validation':True,
            'retrospective_holdout_note':'This train dataset was previously inspected; split is an engineering holdout, not a new unseen benchmark.'}}
    xs=[];ys=[];ties=0
    for index,record in enumerate(fit):
        g,z=record['gipformer']['words'],record['zipformer']['words']
        columns=align_words(g,z)
        maps={name:word_reference_map(record['row']['reference'],record[name]['words']) for name in ('gipformer','zipformer')}
        for i,column in enumerate(columns):
            ga=' '.join(g[k]['text'] for k in column['gipformer']);za=' '.join(z[k]['text'] for k in column['zipformer'])
            if normal(ga).replace(' ','')==normal(za).replace(' ',''):continue
            expected=expected_region(column,maps,g,z)
            gc,zc=reference_cost(expected,ga),reference_cost(expected,za)
            if gc==zc:ties+=1;continue
            before=[g[k]['text'] for c in columns[max(0,i-2):i] for k in c['gipformer']][-3:]
            after=[z[k]['text'] for c in columns[i+1:i+3] for k in c['zipformer']][:3]
            xs.append(region_features(column,g,z,calibrators,language,before,after));ys.append(float(gc<zc))
        if index%500==0:print('[FUSION FIT]',index,'regions',len(ys),flush=True)
    artifact['selector']=fit_logistic(xs,ys,regularization=12.)
    artifact['fit_metadata'].update(selector_training_regions=len(ys),tied_training_regions_skipped=ties)
    print('[FUSION VALIDATION] preparing',len(validation),'audio',flush=True)
    prepared=[prepare_regions(r['gipformer']['words'],r['zipformer']['words'],artifact,language) for r in validation]
    lexicon=json.loads(Path('outputs/asr_datagen_targets/lexicon.json').read_text())
    evaluate=keyword_evaluator(lexicon);scores=[]
    for penalty in [0.,.10,.30]:
        loss=total=0
        for record,columns in zip(validation,prepared):
            text=select_path(columns,record['gipformer']['words'],record['zipformer']['words'],penalty)['text']
            bad,n=evaluate(record['row']['reference'],text);loss+=bad;total+=n
        scores.append({'switch_penalty':penalty,'weighted_keyword_flags':loss,'target_occurrences':total})
        print('[FUSION VALIDATION]',scores[-1],flush=True)
    artifact['switch_penalty']=min(scores,key=lambda r:r['weighted_keyword_flags'])['switch_penalty']
    artifact['fit_metadata']['validation_candidates']=scores
    artifact['fit_metadata']['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    atomic_json(ROOT/'selector.json',artifact)
    print('[FUSION SELECTOR FROZEN]',artifact['switch_penalty'],flush=True)
    with (ROOT/'results.jsonl').open('w',encoding='utf-8') as output:
        for index,record in enumerate(records,1):
            row=record['row'];g,z=record['gipformer'],record['zipformer']
            result=select_path(prepare_regions(g['words'],z['words'],artifact,language),g['words'],z['words'],artifact['switch_penalty'])
            key=hashlib.sha256((row['id']+'|'+row['turn']+'|'+row['history_index']).encode()).hexdigest()
            detail={'id':row['id'],'turn':row['turn'],'history_index':row['history_index'],
                'audio':row['audio'],'split':record['split'],'fusion_text':result['text'],
                'gipformer_cache':paths[(row['id'],row['turn'],row['history_index'],'gipformer')],
                'zipformer_cache':paths[(row['id'],row['turn'],row['history_index'],'zipformer')],
                'fusion_detail':str(ROOT/'decisions'/(key+'.json'))}
            atomic_json(detail['fusion_detail'],{**detail,'decisions':result['decisions']})
            output.write(json.dumps(detail,ensure_ascii=False)+'\n');output.flush()
            if index%100==0 or index==len(records):
                atomic_json(ROOT/'progress.json',{'processed':index,'total':len(records),'finished':index==len(records),'elapsed_seconds':round(time.time()-started,2)})
                print('[FUSION OUTPUT]',index,'/',len(records),flush=True)
    print('[FUSION COMPLETE]',round(time.time()-started,2),'seconds',flush=True)


if __name__=='__main__':main()
