"""Time-aware word alignment and calibrated, reference-free path selection."""
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

VERSION = 'timed-word-fusion-v2'
NUMBER = set('không một mốt hai ba bốn tư năm lăm sáu bảy bẩy tám chín mười mươi trăm nghìn ngàn triệu tỷ tỉ lẻ linh phẩy'.split())
UNITS = set('giờ phút giây độ phần đồng đô lít tiếng'.split())
NEGATION = set('chưa đã đừng không khỏi chớ'.split())
ACTIONS = set('uống bật tắt trả giao cần gửi đặt chuyển nhắn nhắc hủy xóa dời'.split())
STOP = set('anh em con cho toi trong nhanh nha nghe nhe hen mai nay ba hai nam la co va di den tu voi thi khi roi neu can xem giup dung sau truoc so cung o noi ben ve hom cai cua chi ong me bo gi chua moi lam day ay xin mot nhan tat bat mo len de rat qua qua'.split())


def normal(text):
    text = unicodedata.normalize('NFC', text).casefold().replace('hoà', 'hòa').replace('huỷ', 'hủy').replace('xoá', 'xóa')
    return ' '.join(re.findall(r'[^\W_]+', text))


def join_key(words):
    return ''.join(normal(w['text']).replace(' ', '') for w in words)


def latin(word):
    text = normal(word)
    return bool(re.fullmatch('[a-z][a-z0-9]*', text)) and text not in STOP and text not in NUMBER


def protect_spans(words):
    """Group only short semantic units, using hypotheses, never reference names."""
    spans = []
    tokens = [normal(w['text']) for w in words]
    index = 0
    while index < len(tokens):
        end = index
        while end < len(tokens) and tokens[end] in NUMBER | UNITS:
            end += 1
        if end-index >= 2 and any(w in NUMBER for w in tokens[index:end]):
            spans.append((index, end, 'numeric_value'))
        index = max(index+1, end)
    index = 0
    while index < len(tokens):
        end = index
        while end < len(tokens) and latin(words[end]['text']):
            end += 1
        if end-index >= 2 and any(len(w) >= 4 for w in tokens[index:end]):
            spans.append((index, min(end, index+6), 'latin_identifier'))
        index = max(index+1, end)
    for index, token in enumerate(tokens):
        if index+1 < len(tokens) and token in NEGATION and tokens[index+1] in ACTIONS:
            spans.append((index, index+2, 'negation_action'))
        if index+1 < len(tokens) and token in {'sáng', 'trưa', 'chiều', 'tối', 'hôm', 'ngày'} and tokens[index+1] in {'mai', 'nay', 'qua', 'kia'}:
            spans.append((index, index+2, 'relative_time'))
        if token in {'thuốc', 'viên', 'vitamin', 'cữ'}:
            end = index+1
            while end < min(len(tokens), index+4) and tokens[end] not in {'cho', 'của', 'tôi', 'anh', 'em', 'chị', 'cô', 'thì', 'để', 'rồi', 'với', 'và', 'là', 'mẹ', 'đó', 'ấy', 'đi', 'uống', 'ngày', 'mỗi', 'đúng', 'vô'}:
                end += 1
            if end > index+1:
                spans.append((index, end, 'medical_phrase'))
    return spans


def align_words(gip, zip_, tolerance=.20):
    n, m = len(gip), len(zip_)
    costs = [[float('inf')]*(m+1) for _ in range(n+1)]
    back = {}
    costs[0][0] = 0.
    def relax(i, j, ni, nj, extra):
        value = costs[i][j]+extra
        if value < costs[ni][nj]:
            costs[ni][nj] = value
            back[ni, nj] = (i, j)
    for i in range(n+1):
        for j in range(m+1):
            if i < n:
                relax(i,j,i+1,j,.78)
            if j < m:
                relax(i,j,i,j+1,.78)
            if i == n or j == m:
                continue
            dt = abs(gip[i]['start']-zip_[j]['start'])
            a, b = join_key(gip[i:i+1]), join_key(zip_[j:j+1])
            # RNN-T emission times can differ by >1 s near a pause even for
            # identical words. Keep lexical anchors available; time is a soft
            # cost, otherwise two gap columns duplicate a word during fusion.
            if dt > .85 and a != b:
                continue
            similarity = 1. if a == b else SequenceMatcher(None,a,b).ratio()
            relax(i,j,i+1,j+1,(1-similarity)*.9 + min(dt/tolerance,3)*.20)
            for ga, za in ((1,2),(2,1),(1,3),(3,1)):
                if i+ga>n or j+za>m:
                    continue
                a,b = join_key(gip[i:i+ga]),join_key(zip_[j:j+za])
                if a != b:
                    continue
                last = abs(gip[i+ga-1]['last_token_time']-zip_[j+za-1]['last_token_time'])
                if last < .4:
                    relax(i,j,i+ga,j+za,.08*(ga+za-2)+min(dt/tolerance,3)*.20)
    i,j=n,m
    columns=[]
    while i or j:
        pi,pj=back[i,j]
        columns.append({'gipformer':list(range(pi,i)), 'zipformer':list(range(pj,j)), 'protected':[]})
        i,j=pi,pj
    columns.reverse()
    # Map protected intervals from either hypothesis onto aligned columns.
    intervals=[]
    for name,words in [('gipformer',gip),('zipformer',zip_)]:
        mapping={index:c for c,column in enumerate(columns) for index in column[name]}
        for start,end,kind in protect_spans(words):
            positions=[mapping[k] for k in range(start,end)]
            intervals.append((min(positions),max(positions)+1,kind))
    groups=[]
    for start,end,kind in sorted(intervals):
        if groups and start<groups[-1][1]:
            groups[-1][1]=max(end,groups[-1][1]);groups[-1][2].add(kind)
        else:
            groups.append([start,end,{kind}])
    result=[];index=0;by_start={a:(b,k) for a,b,k in groups}
    while index<len(columns):
        if index in by_start:
            end,kinds=by_start[index]
            result.append({name:[w for c in columns[index:end] for w in c[name]] for name in ('gipformer','zipformer')})
            result[-1]['protected']=sorted(kinds)
            index=end
        else:
            result.append(columns[index]);index+=1
    for name,length in [('gipformer',n),('zipformer',m)]:
        if [k for c in result for k in c[name]]!=list(range(length)):
            raise ValueError('Alignment lost or duplicated source words')
    return result


def word_features(word):
    token=normal(word['text'])
    return [1.,max(-8.,word['mean_logp']),max(-12.,word['min_logp']),
            math.log1p(len(word['token_indices'])),min(len(token),20)/10,
            float(token in NUMBER),float(latin(word['text'])),float(token in NEGATION),
            float(token in ACTIONS),float(token in UNITS)]


def probability(features, coefficients):
    value=sum(a*b for a,b in zip(features,coefficients))
    return 1/(1+math.exp(-max(-30.,min(30.,value))))


class LanguageCounts:
    def __init__(self,data=None):
        self.unigram=Counter();self.bigram=Counter();self.trigram=Counter()
        if data:
            self.unigram.update(data['unigram'])
            self.bigram.update({tuple(k.split('\t')):v for k,v in data['bigram'].items()})
            self.trigram.update({tuple(k.split('\t')):v for k,v in data['trigram'].items()})
        self.total=sum(self.unigram.values())

    def add(self,tokens):
        words=['<s>','<s>']+tokens+['</s>']
        self.unigram.update(words)
        self.bigram.update(zip(words,words[1:]))
        self.trigram.update(zip(words,words[1:],words[2:]))
        self.total=sum(self.unigram.values())

    def score(self,words):
        if not words:
            return 0.
        tokens=['<s>','<s>']+[normal(w) for w in words]+['</s>']
        vocab=max(100,len(self.unigram))
        result=[]
        for a,b,c in zip(tokens,tokens[1:],tokens[2:]):
            uni=(self.unigram[c]+.1)/(self.total+.1*vocab)
            bi=(self.bigram[b,c]+.5*uni)/(self.unigram[b]+.5)
            tri=(self.trigram[a,b,c]+1.*bi)/(self.bigram[a,b]+1.)
            result.append(math.log(max(1e-12,.2*uni+.3*bi+.5*tri)))
        return sum(result)/len(result)

    def export(self):
        return {'unigram':dict(self.unigram),
                'bigram':{'\t'.join(k):v for k,v in self.bigram.items()},
                'trigram':{'\t'.join(k):v for k,v in self.trigram.items()}}


def region_features(column, gip, zip_, calibrators, language, before=(), after=()):
    features=[];stats=[]
    for name,words in [('gipformer',gip),('zipformer',zip_)]:
        selected=[words[i] for i in column[name]]
        probs=[probability(word_features(w),calibrators[name]) for w in selected]
        raw=[normal(w['text']) for w in selected]
        stats.append([sum(probs)/len(probs) if probs else .5,
                      min(probs) if probs else .5,
                      sum(max(-8,w['mean_logp']) for w in selected)/len(selected) if selected else -2.,
                      language.score(list(before)+[w['text'] for w in selected]+list(after)),
                      min(len(selected),12)/6,
                      float(not selected),
                      sum(w in NUMBER for w in raw)/max(1,len(raw)),
                      sum(latin(w['text']) for w in selected)/max(1,len(raw)),
                      sum(w in NEGATION for w in raw)/max(1,len(raw)),
                      sum(w in ACTIONS for w in raw)/max(1,len(raw)),
                      min(sum(len(w) for w in raw),60)/30])
    difference=[a-b for a,b in zip(*stats)]
    features=[1.]+difference
    for kind in ['numeric_value','latin_identifier','negation_action','relative_time','medical_phrase']:
        flag=float(kind in column['protected'])
        features += [flag,flag*difference[0],flag*difference[3],flag*difference[2]]
    return features


def prepare_regions(gip, zip_, artifact, language=None):
    columns=align_words(gip,zip_,artifact.get('timestamp_tolerance',.2))
    if language is None:
        language=LanguageCounts(artifact['language_counts'])
    for index,column in enumerate(columns):
        left=[gip[k]['text'] for c in columns[max(0,index-2):index] for k in c['gipformer']][-3:]
        right=[zip_[k]['text'] for c in columns[index+1:index+3] for k in c['zipformer']][:3]
        column['features']=region_features(column,gip,zip_,artifact['calibrators'],language,left,right)
        a=[gip[k] for k in column['gipformer']];b=[zip_[k] for k in column['zipformer']]
        column['agreement']=join_key(a)==join_key(b)
        column['p_gipformer']=probability(column['features'],artifact['selector']) if not column['agreement'] else .5
        column['calibrated_word_scores']={name:[probability(word_features(words[k]),artifact['calibrators'][name])
            for k in column[name]] for name,words in [('gipformer',gip),('zipformer',zip_)]}
    return columns


def select_path(columns,gip,zip_,penalty=.10):
    if not columns:
        return {'text':'','decisions':[]}
    scores=[[float('-inf')]*2 for _ in columns];back=[]
    for index,column in enumerate(columns):
        p=column['p_gipformer'];unary=[math.log(max(p,1e-8)),math.log(max(1-p,1e-8))]
        prev=[]
        for source in (0,1):
            if index==0:
                scores[index][source]=unary[source];prev.append(-1)
            else:
                options=[scores[index-1][s]-(penalty if s!=source and not column['agreement'] and not columns[index-1]['agreement'] else 0) for s in (0,1)]
                chosen=max((0,1),key=lambda s:options[s])
                scores[index][source]=unary[source]+options[chosen];prev.append(chosen)
        back.append(prev)
    path=[];source=max((0,1),key=lambda s:scores[-1][s])
    for index in range(len(columns)-1,-1,-1):
        path.append(source);source=back[index][source]
    path.reverse();decisions=[];output=[]
    for index,(column,source) in enumerate(zip(columns,path)):
        name=('gipformer','zipformer')[source]
        words=(gip,zip_)[source];selected=[words[k] for k in column[name]]
        all_words=[gip[k] for k in column['gipformer']]+[zip_[k] for k in column['zipformer']]
        text=' '.join(w['text'] for w in selected)
        output.extend(w['text'] for w in selected)
        decisions.append({'region':index,'start':min(w['start'] for w in all_words),
            'end_estimated':max(w['end_estimated'] for w in all_words),'end_is_estimated':True,
            'gipformer_words':' '.join(gip[k]['text'] for k in column['gipformer']),
            'zipformer_words':' '.join(zip_[k]['text'] for k in column['zipformer']),
            'chosen_words':text,'chosen_source':name,'agreement':column['agreement'],
            'p_gipformer':column['p_gipformer'],'uncertain':not column['agreement'] and abs(column['p_gipformer']-.5)<.12,
            'protected':column['protected'],
            'decision_reason':'same_normalized_word' if column['agreement'] else 'calibrated_word_scores_and_fit_only_ngram_selector',
            'calibrated_word_scores':column.get('calibrated_word_scores',{}),
            'source_word_indices':{n:column[n] for n in ('gipformer','zipformer')},
            'source_token_indices':{name:[i for k in column[name] for i in words[k]['token_indices']]
                for name,words in [('gipformer',gip),('zipformer',zip_)]},
            'gipformer_mean_logp':[gip[k]['mean_logp'] for k in column['gipformer']],
            'zipformer_mean_logp':[zip_[k]['mean_logp'] for k in column['zipformer']]})
    return {'text':' '.join(output),'decisions':decisions}


def fuse_detailed(gip,zip_,artifact):
    return select_path(prepare_regions(gip['words'],zip_['words'],artifact),gip['words'],zip_['words'],artifact.get('switch_penalty',.1))


def transcribe_fused(audio_path,selector_path,timed_root,paths,cache_only=False):
    """Runtime entry point: only audio, checkpoints and frozen selector; no gold."""
    import json
    from pathlib import Path
    import sherpa_onnx
    from main import model_fingerprint
    from .timed import atomic_json,cache_identity,to_words
    from .gipformer import GipFormer
    from .zipformer import ZipFormer
    artifact=json.loads(Path(selector_path).read_text())
    if artifact['version']!=VERSION:
        raise ValueError('Unsupported fusion selector version')
    details={}
    for name,Model in [('gipformer',GipFormer),('zipformer',ZipFormer)]:
        key,identity=cache_identity(audio_path,model_fingerprint(paths[name]),sherpa_onnx.__version__)
        destination=Path(timed_root)/name/(key+'.json')
        if destination.exists():
            details[name]=json.loads(destination.read_text())
            if details[name]['key']!=key:raise ValueError('Timed cache identity mismatch')
        else:
            if cache_only:raise RuntimeError(f'Thiếu timed ASR cache: {destination}')
            model=Model(paths[name],'cpu')
            try:detail=model.transcribe_detailed(str(audio_path))
            finally:del model
            detail['words']=to_words(detail)
            details[name]={'key':key,'identity':identity,**detail}
            atomic_json(destination,details[name])
    result=fuse_detailed(details['gipformer'],details['zipformer'],artifact)
    result['transcripts']={name:details[name]['text'] for name in details}
    result['selector_version']=artifact['version']
    return result
