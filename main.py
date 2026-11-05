import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from agent import normalize, run_agent
from env_adapter import make_environment


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stage_cache(folder, stage, key, resume):
    path = folder / (stage + '.cache.json')
    if resume and path.exists():
        data = json.loads(path.read_text())
        if data.get('key') == key:
            return data['value']
    return None


def save_cache(folder, stage, key, value):
    path = folder / (stage + '.cache.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'key': key, 'value': value}, ensure_ascii=False, indent=2))
    temporary.replace(path)


def model_fingerprint(path):
    p = Path(path)
    if not p.exists():
        # Resolve cached HF revisions; an unpinned moving model must not reuse cache.
        try:
            from huggingface_hub import model_info
            return [path, model_info(path).sha]
        except Exception:
            return [path, None, time.time_ns()]
    files = [p] if p.is_file() else sorted(f for f in p.rglob('*') if f.is_file())
    return [str(p.resolve()), [(str(f.relative_to(p)) if p.is_dir() else f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files]]


def comparable(text):
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold()))


def asr_evidence(hypotheses, chosen):
    """Raw ASR hypotheses for the agent, only when they differ from the chosen text beyond case/punctuation."""
    cleaned = {name: text.strip() for name, text in hypotheses.items() if isinstance(text, str) and text.strip()}
    if not cleaned or all(comparable(text) == comparable(chosen or '') for text in cleaned.values()):
        return None
    return cleaned


def asr_cache_key(name, fingerprint, audio, device, decoding=None):
    # decoding=None (greedy) keeps the key of every cache saved before beam search / hotwords existed.
    parts = [name, fingerprint, hashlib.sha256(Path(audio).read_bytes()).hexdigest(), device]
    return digest(parts + [decoding] if decoding else parts)


def recognize(audio, name, folder, stage, trace, resume, save, history=False):
    from asr import decoding_spec, model_spec
    from asr.gipformer import decoding_identity
    Model, path = model_spec(name)
    decoding = decoding_spec(name)
    key = asr_cache_key(name, model_fingerprint(path), audio, os.environ['ASR_DEVICE'], decoding_identity(decoding))
    value = stage_cache(folder, stage, key, resume)
    shared = Path(os.environ['ASR_CACHE_DIR']) if os.environ.get('ASR_CACHE_DIR') else None
    if value is None and shared:
        value = stage_cache(shared, key, key, True)
        if value is not None:
            trace(stage='asr_cache_hit', model=path, input=str(audio), output=value, cache_key=key)
    if value is None:
        if os.environ.get('ASR_CACHE_ONLY') == '1':
            raise RuntimeError(f'Thiếu ASR cache đã xác minh cho {name}: {audio}; ASR_CACHE_ONLY=1 chặn chạy lại model')
        start = time.monotonic()
        model = Model(path, os.environ['ASR_DEVICE'], decoding) if decoding else Model(path, os.environ['ASR_DEVICE'])
        try:
            value = model.transcribe(str(audio))
        finally:
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except ImportError:
                pass
        trace(stage='history_asr' if history else 'asr', model=path, input=str(audio), output=value, latency=time.monotonic()-start)
        if save and shared:
            shared.mkdir(parents=True, exist_ok=True)
            save_cache(shared, key, key, value)
    if save:
        save_cache(folder, stage, key, value)
    return value


def recognize_fused(audio,folder,stage,trace,resume,save):
    if os.environ['ASR_MODE']!='dual' or {os.environ['ASR_PRIMARY'],os.environ['ASR_SECONDARY']}!={'gipformer','zipformer'}:
        raise ValueError('timestamp_word cần dual GipFormer + ZipFormer')
    if os.environ['ASR_DEVICE']!='cpu':
        raise ValueError('timestamp_word hiện dùng cache/decoder CPU đã hiệu chỉnh')
    from asr import decoding_spec
    if decoding_spec('gipformer') or decoding_spec('zipformer'):
        raise ValueError('timestamp_word selector được hiệu chỉnh trên greedy_search; bỏ *_DECODING/*_HOTWORDS hoặc dùng ASR_COMBINE khác')
    from asr.word_fusion import transcribe_fused,VERSION
    selector=os.environ['FUSION_SELECTOR_PATH']
    paths={'gipformer':os.environ['GIPFORMER_MODEL_PATH'],'zipformer':os.environ['ZIPFORMER_MODEL_PATH']}
    # The cached value holds the fused text plus both raw hypotheses; the marker keeps text-only entries out.
    key=digest(['text+transcripts',VERSION,model_fingerprint(selector),model_fingerprint('asr/word_fusion.py'),
                model_fingerprint('asr/timed.py'),{name:model_fingerprint(path) for name,path in paths.items()},
                hashlib.sha256(Path(audio).read_bytes()).hexdigest()])
    value=stage_cache(folder,stage,key,resume)
    if value is None:
        start=time.monotonic()
        result=transcribe_fused(audio,selector,os.environ['ASR_TIMED_CACHE_DIR'],paths,os.environ.get('ASR_CACHE_ONLY')=='1')
        value={'text':result['text'],'transcripts':result['transcripts']}
        trace(stage='asr_timestamp_fusion',input=str(audio),output=value['text'],transcripts=result['transcripts'],
              decisions=result['decisions'],selector=selector,latency=time.monotonic()-start)
        if save:save_cache(folder,stage,key,value)
    else:
        trace(stage='asr_fusion_cache_hit',input=str(audio),output=value['text'],cache_key=key)
    return value


def process(case, args, llm_holder):
    sid = str(case.get('id', 'sample'))
    if sid in ('', '.', '..') or Path(sid).name != sid:
        raise ValueError('id phải là một tên thư mục an toàn')
    folder = Path(os.environ['OUTPUT_DIR']) / sid
    folder.mkdir(parents=True, exist_ok=True)
    def trace(**entry):
        with (folder / 'trace.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
    debug = args.debug or os.environ['DEBUG'] == '1'
    resume = args.resume or os.environ['RESUME'] == '1'
    save = os.environ['SAVE_INTERMEDIATE'] == '1'
    if os.environ.get('AGENT_ASR_EVIDENCE', '0') not in ('0', '1'):
        raise ValueError('AGENT_ASR_EVIDENCE chỉ nhận 0 hoặc 1')
    evidence = os.environ.get('AGENT_ASR_EVIDENCE', '0') == '1'
    result = {'id': sid, 'asr_text': None, 'normalized_text': None, 'tool_calls': [], 'state_hash': None}
    text = args.text if args.text is not None else case.get('text')
    audio = args.audio or case.get('audio')
    transcripts = {}
    hypotheses = {}  # every raw ASR output for the final turn; empty when text is given directly
    if text is not None:
        transcripts['provided'] = text
    else:
        if not audio:
            raise ValueError('Cần --audio, --text hoặc audio trong case')
        if os.environ['ASR_MODE'] not in ('single', 'dual') or os.environ['ASR_COMBINE'] not in ('primary', 'llm_review', 'timestamp_word'):
            raise ValueError('ASR_MODE hoặc ASR_COMBINE không hợp lệ')
        if os.environ['ASR_COMBINE']=='timestamp_word':
            fused=recognize_fused(audio,folder,'asr_timestamp_word',trace,resume,save)
            transcripts['timestamp_word']=fused['text']
            hypotheses.update(fused['transcripts'])
        else:
            names = [os.environ['ASR_PRIMARY']]
            if os.environ['ASR_MODE'] == 'dual': names.append(os.environ['ASR_SECONDARY'])
            for name in names:
                transcripts[name] = recognize(audio, name, folder, 'asr_' + name, trace, resume, save)
        hypotheses = {**transcripts, **hypotheses}
    result['asr_text'] = next(iter(transcripts.values()))
    if debug: print('[ASR]', json.dumps(transcripts, ensure_ascii=False))
    if args.mode != 'asr':
        relevant = transcripts if os.environ['ASR_COMBINE'] == 'llm_review' else {next(iter(transcripts)): result['asr_text']}
        key = digest([model_fingerprint(os.environ['LLM_MODEL_PATH']), os.environ['NORMALIZE_SYSTEM_PROMPT'], relevant, os.environ['LLM_DTYPE'], os.environ['LLM_TEMPERATURE'], os.environ['NORMALIZE_MAX_NEW_TOKENS']])
        value = stage_cache(folder, 'normalize', key, resume)
        if value is None:
            if llm_holder[0] is None:
                from llm_qwen import Qwen, check_competition
                llm_holder[0] = Qwen(trace)
                check_competition(llm_holder[0], trace)
            llm_holder[0].trace = trace
            value = normalize(llm_holder[0], relevant, trace)
            if save: save_cache(folder, 'normalize', key, value)
        result['normalized_text'] = value
        if debug: print('[NORMALIZED]', value)
    if args.mode in ('agent', 'all'):
        history = []
        dual = os.environ.get('ASR_MODE', 'single') == 'dual'
        for index, original in enumerate(case.get('history', [])):
            turn = {k: v for k, v in original.items() if k != 'audio_text'}
            if turn.get('role') == 'user' and turn.get('audio') and not turn.get('text'):
                name = os.environ['ASR_PRIMARY']
                audio_path = Path(turn['audio'])
                if not audio_path.is_absolute():
                    audio_path = Path(os.environ['DATA_ROOT']) / os.environ['DATA_SPLIT'] / audio_path
                stage = f'history_{index}_{name}'
                if os.environ['ASR_COMBINE']=='timestamp_word':
                    fused=recognize_fused(audio_path,folder,f'history_{index}_timestamp_word',trace,resume,save)
                    value=fused['text']
                    alternatives={'timestamp_word':value,**fused['transcripts']}
                else:
                    value = recognize(audio_path, name, folder, stage, trace, resume, save, history=True)
                    alternatives = {name: value}
                review = os.environ.get('ASR_REVIEW_HISTORY') == '1' and dual and os.environ['ASR_COMBINE'] == 'llm_review'
                if dual and os.environ['ASR_COMBINE'] != 'timestamp_word' and (review or evidence):
                    secondary = os.environ['ASR_SECONDARY']
                    alternatives[secondary] = recognize(audio_path, secondary, folder, f'history_{index}_{secondary}', trace, resume, save, history=True)
                if review:
                    stage = f'history_{index}_normalize'
                    key = digest([model_fingerprint(os.environ['LLM_MODEL_PATH']), os.environ['NORMALIZE_SYSTEM_PROMPT'], alternatives, os.environ['LLM_DTYPE'], os.environ['LLM_TEMPERATURE'], os.environ['NORMALIZE_MAX_NEW_TOKENS']])
                    value = stage_cache(folder, stage, key, resume)
                    if value is None:
                        llm_holder[0].trace = trace
                        start = time.monotonic()
                        value = normalize(llm_holder[0], alternatives, trace)
                        trace(stage='history_normalize', history_index=index, input=alternatives, output=value, latency=time.monotonic()-start)
                        if save:
                            save_cache(folder, stage, key, value)
                turn['text'] = value
                turn.pop('audio', None)
                if evidence and (raw := asr_evidence(alternatives, value)):
                    turn['asr_alternatives'] = raw
            history.append(turn)
        env = make_environment(os.environ['ENV_MODE'], case)
        if save and hasattr(env, 'get_state'):
            (folder / 'environment_initial.json').write_text(json.dumps(env.get_state(), ensure_ascii=False, indent=2))
        if llm_holder[0] is None:
            from llm_qwen import Qwen, check_competition
            llm_holder[0] = Qwen(trace)
            check_competition(llm_holder[0], trace)
        llm_holder[0].trace = trace
        try:
            raw = asr_evidence(hypotheses, result['normalized_text']) if evidence else None
            examples = None
            if int(os.environ.get('AGENT_FEWSHOT_K') or 0) > 0:
                from fewshot import select_examples
                # Retrieval sees only this sample's own ASR/normalized text and tools, never its label.
                ids, examples = select_examples(os.environ['AGENT_FEWSHOT_POOL'], [result['normalized_text'], *hypotheses.values()],
                                                list(env.schemas), int(os.environ['AGENT_FEWSHOT_K']), sid,
                                                os.environ.get('AGENT_FEWSHOT_EXCLUDE') or None)
                trace(stage='fewshot', example_ids=ids)
            result['tool_calls'] = run_agent(llm_holder[0], env, result['normalized_text'], case.get('context', {}), history, trace, int(os.environ['MAX_AGENT_STEPS']), debug, asr_transcripts=raw, examples=examples)
        finally:
            if save and hasattr(env, 'get_state'):
                (folder / 'environment_final.json').write_text(json.dumps(env.get_state(), ensure_ascii=False, indent=2))
        if hasattr(env, 'get_tool_calls'):
            result['tool_calls'] = env.get_tool_calls()
        result['state_hash'] = env.get_state_hash()
    if save: (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if debug: print('[FINAL]', json.dumps(result, ensure_ascii=False))
    elif not args.input: print(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['asr', 'normalize', 'agent', 'all'], default=os.environ.get('RUN_MODE', 'all'))
    parser.add_argument('--audio'); parser.add_argument('--text'); parser.add_argument('--case'); parser.add_argument('--input'); parser.add_argument('--output')
    parser.add_argument('--id'); parser.add_argument('--split', choices=['train', 'public_test']);
    parser.add_argument('--benchmark', type=int); parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precache-asr', action='store_true')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.precache_asr:
        from cache_asr import run_precache
        run_precache(args)
        return
    if args.benchmark is not None:
        from benchmark import run_benchmark
        run_benchmark(args)
        return
    if args.input and args.output and os.environ['ENV_MODE'] != 'official':
        raise RuntimeError('Mock mode không được xuất submission; cần official devkit.')
    if int(os.environ['LLM_CONTEXT_TOKENS']) > 16384:
        raise ValueError('LLM_CONTEXT_TOKENS vượt 16384')
    if args.input:
        if args.mode != 'all': raise ValueError('Batch submission chỉ hỗ trợ mode all')
        if not args.output: raise ValueError('Batch cần --output')
        raise RuntimeError('Batch submission chưa triển khai; hiện chỉ hỗ trợ kiểm thử từng mẫu bằng --id.')
    if args.split:
        os.environ['DATA_SPLIT'] = args.split
    case = json.loads(Path(args.case).read_text()) if args.case else {}
    if args.id:
        root = Path(os.environ['DATA_ROOT'])
        split = os.environ['DATA_SPLIT']
        with (root / split / f'{split}.jsonl').open(encoding='utf-8') as f:
            case = next((item for line in f if (item := json.loads(line))['id'] == args.id), None)
        if case is None:
            raise ValueError(f'Không tìm thấy id: {args.id}')
        case['audio'] = str(root / split / case['audio'])
        case.pop('audio_text', None)
        case.pop('label', None)
    process(case, args, [None])

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
