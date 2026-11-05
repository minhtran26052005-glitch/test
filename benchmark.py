"""Reproducible labeled-train benchmark; never feed labels/transcripts to the agent."""
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from main import process


def run_benchmark(args):
    if args.benchmark < 1:
        raise ValueError('--benchmark phải lớn hơn 0')
    if os.environ['ENV_MODE'] != 'official':
        raise ValueError('Benchmark cần ENV_MODE=official')
    os.environ['DATA_SPLIT'] = 'train'
    root = Path(os.environ['DATA_ROOT']).resolve()
    source = root / 'train'
    sys.path.insert(0, str(source))
    from zaic_tools import state_hash
    with (source / 'train.jsonl').open(encoding='utf-8') as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    if args.benchmark > len(all_items):
        raise ValueError('Số mẫu vượt kích thước train')
    chosen = random.Random(args.seed).sample(all_items, args.benchmark)
    outdir = Path(os.environ['OUTPUT_DIR']) / f'benchmark_seed{args.seed}_n{args.benchmark}'
    outdir.mkdir(parents=True, exist_ok=True)
    from cache_asr import atomic_json
    from main import model_fingerprint
    from asr import model_spec
    metadata = outdir / 'run_metadata.json'
    configuration = {key: value for key, value in os.environ.items() if key.startswith((
        'ASR_', 'GIPFORMER_', 'ZIPFORMER_', 'PHOWHISPER_', 'LLM_', 'NORMALIZE_', 'AGENT_'))
        or key in ('DATA_ROOT', 'DATA_SPLIT', 'MAX_AGENT_STEPS', 'STRICT_COMPETITION', 'ENV_MODE')}
    if os.environ.get('ASR_COMBINE')!='timestamp_word':
        configuration.pop('ASR_TIMED_CACHE_DIR',None)
    for optional in ('ASR_CACHE_ONLY', 'ASR_REVIEW_HISTORY', 'AGENT_ASR_EVIDENCE'):
        if configuration.get(optional) == '0':
            configuration.pop(optional)
    identity = {'configuration': configuration, 'seed': args.seed, 'sample_count': args.benchmark,
                'asr_model_fingerprint': model_fingerprint(model_spec(os.environ['ASR_PRIMARY'])[1])}
    if os.environ.get('ASR_MODE', 'single') == 'dual':
        identity['secondary_asr_model_fingerprint'] = model_fingerprint(model_spec(os.environ['ASR_SECONDARY'])[1])
    if os.environ.get('ASR_COMBINE')=='timestamp_word':
        identity['fusion_selector_fingerprint']=model_fingerprint(os.environ['FUSION_SELECTOR_PATH'])
    identity = json.loads(json.dumps(identity))
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved['identity'] != identity:
            raise ValueError('Cấu hình/model khác lần benchmark trước; cần thư mục OUTPUT_DIR riêng')
    else:
        import hashlib
        files = ['run.sh', 'main.py', 'benchmark.py', 'agent.py', 'llm_qwen.py', 'env_adapter.py',
                 'cache_asr.py', 'asr/__init__.py', 'asr/gipformer.py', 'asr/zipformer.py',
                 'run_zipformer_benchmark.sh', 'export_debug_csv.py', 'compare_asr_runs.py']
        if os.environ.get('ASR_MODE', 'single') == 'dual':
            files += ['prepare_dual_asr.py', 'run_dual_asr_benchmark.sh', 'compare_dual_asr.py']
        if os.environ.get('AGENT_ASR_EVIDENCE') == '1':
            files += ['run_agent_asr_evidence_benchmark.sh', 'compare_benchmark_runs.py']
        atomic_json(metadata, {'identity': identity, 'created_at_unix': time.time(),
                    'python': sys.version, 'code_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},
                    'dataset_sha256': hashlib.sha256((source / 'train.jsonl').read_bytes()).hexdigest()})
    manifest = outdir / 'results.jsonl'
    previous = []
    if manifest.exists() and manifest.stat().st_size:
        if not args.resume:
            raise RuntimeError(f'Benchmark đã tồn tại tại {manifest}; dùng --resume để tiếp tục.')
        previous = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(chosen) == len(all_items):
            by_id = {item['id']: item for item in all_items}
            prior_ids = [entry['id'] for entry in previous]
            if len(set(prior_ids)) != len(prior_ids) or any(sid not in by_id for sid in prior_ids):
                raise ValueError('Kết quả đã lưu có id trùng hoặc không thuộc tập train')
            prior_set = set(prior_ids)
            chosen = [by_id[sid] for sid in prior_ids] + [item for item in chosen if item['id'] not in prior_set]
        elif [e['id'] for e in previous] != [x['id'] for x in chosen[:len(previous)]]:
            raise ValueError('Kết quả đã lưu không khớp thứ tự mẫu benchmark')
    os.environ['OUTPUT_DIR'] = str(outdir / 'samples')
    args.mode = 'all'
    args.input = '__benchmark__'
    args.text = None
    args.audio = None
    args.resume = True
    llm_holder = [None]
    counts = {'state_match': 0, 'exact_calls': 0, 'names_match': 0, 'both': 0, 'completed': 0, 'errors': 0}
    started = time.time()
    for entry in previous:
        if 'error' in entry:
            counts['errors'] += 1
        else:
            counts['completed'] += 1
            for key in ('state_match', 'exact_calls', 'names_match', 'both'):
                counts[key] += int(entry[key])
    with manifest.open('a' if previous else 'x', encoding='utf-8') as output:
        for i, item in enumerate(chosen, 1):
            sid = item['id']
            if i <= len(previous):
                continue
            atomic_json(outdir / 'progress.json', {'processed': i-1, 'total': len(chosen),
                        'active_id': sid, **counts, 'finished': False, 'updated_at_unix': time.time()})
            # Do not create a new session for a sample interrupted in an earlier run.
            if (outdir / 'samples' / sid).exists():
                entry = {'id': sid, 'error': 'Phiên trước bị gián đoạn; không khởi tạo lại', 'state_match': False, 'exact_calls': False, 'names_match': False, 'both': False}
                output.write(json.dumps(entry, ensure_ascii=False) + '\n')
                output.flush()
                counts['errors'] += 1
                continue
            case = {k: copy.deepcopy(item[k]) for k in ('id', 'context', 'history') if k in item}
            case['audio'] = str(source / item['audio'])
            for turn in case.get('history', []):
                turn.pop('audio_text', None)
            entry = {'id': sid, 'history_turns': len(case.get('history', [])),
                     'expected_tool_calls': item['label']['tool_calls'], 'reference_text': item['audio_text']}
            sample_started = time.monotonic()
            try:
                result = process(case, args, llm_holder)
                initial = json.loads((source / 'train_environment' / sid / 'state.json').read_text())
                reference_state = {k: v for k, v in initial.items() if k != '_meta'}
                for name in initial['_meta']['tools']:
                    app = name.split('.', 1)[0]
                    if app != 'sys': reference_state.setdefault(app, {})
                reference_state.update(item['label']['final_state'])
                gold = item['label']['tool_calls']
                pred = result['tool_calls']
                entry.update(state_match=result['state_hash'] == state_hash(reference_state), exact_calls=pred == gold,
                             names_match=[x['name'] for x in pred] == [x['name'] for x in gold],
                             predicted_tool_calls=pred, expected_tool_calls=gold,
                             asr_text=result['asr_text'], normalized_text=result['normalized_text'], reference_text=item['audio_text'])
                entry['both'] = entry['state_match'] and entry['exact_calls']
                counts['completed'] += 1
                for key in ('state_match', 'exact_calls', 'names_match', 'both'):
                    counts[key] += int(entry[key])
            except Exception as exc:
                entry.update(error=f'{type(exc).__name__}: {exc}', state_match=False, exact_calls=False, names_match=False, both=False)
                import traceback
                folder = outdir / 'samples' / sid
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / 'trace.jsonl').open('a', encoding='utf-8') as trace:
                    trace.write(json.dumps({'stage': 'pipeline_error', 'error': entry['error'], 'traceback': traceback.format_exc()}, ensure_ascii=False) + '\n')
                counts['errors'] += 1
            entry['latency_seconds'] = round(time.monotonic() - sample_started, 3)
            output.write(json.dumps(entry, ensure_ascii=False) + '\n')
            output.flush()
            atomic_json(outdir / 'progress.json', {'processed': i, 'total': len(chosen),
                        'active_id': None, **counts, 'finished': False, 'updated_at_unix': time.time(),
                        'elapsed_seconds_this_process': round(time.time()-started, 2)})
            if i == 1 or i % 5 == 0 or i == len(chosen):
                print(f"[BENCHMARK] {i}/{len(chosen)} state={counts['state_match']}/{i}, exact={counts['exact_calls']}/{i}, errors={counts['errors']}", flush=True)
    def wilson(n_success, n):
        z = 1.96
        p = n_success / n
        den = 1 + z*z/n
        center = (p + z*z/(2*n))/den
        margin = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
        return [round(100*(center-margin), 2), round(100*(center+margin), 2)]
    summary = {'split':'train','seed':args.seed,'sample_count':len(chosen),'population':len(all_items), **counts,
               'percentages': {key: round(100*counts[key]/len(chosen), 2) for key in ('state_match','exact_calls','names_match','both')},
               'confidence_95_wilson_percent':{key: wilson(counts[key],len(chosen)) for key in ('state_match','exact_calls','both')} if len(chosen) < len(all_items) else None,
               'note': 'Errored samples count as failures. Exact call comparison is stricter than semantic matching; state match alone does not establish intent correctness. Not an official leaderboard score.'}
    (outdir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    atomic_json(outdir / 'progress.json', {'processed': len(chosen), 'total': len(chosen),
                'active_id': None, **counts, 'finished': True, 'updated_at_unix': time.time()})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
