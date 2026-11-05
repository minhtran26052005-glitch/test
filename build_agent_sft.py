"""Build chat SFT data for the shared Qwen (tool agent + ASR normalizer) from labeled train data.

Agent records: every gold tool call is replayed in a fresh official devkit session, so observations are real;
each step becomes one record whose user message is the exact payload run_agent would send (agent.build_payload,
same JSON serialization) and whose assistant message is the gold decision. A sample is kept only if every gold
call validates and the replayed final state hash equals the label's state hash.

Normalizer records: saved ASR hypotheses of each audio (final turn and user history) -> the reference transcript,
in the payload format of agent.normalize.

Input sources for the agent's final-turn text (repeat --source):
  reference            clean reference transcript and history (upper bound, no ASR noise)
  run:DIR              exact inputs a saved benchmark run gave the agent (ASR-noisy normalized text, raw
                       hypotheses, history), read from DIR/samples/<id>/trace.jsonl
  generated[:CSV]      label-safe paraphrases from audit_finetune_data.py (default outputs/finetune_data_audit/variants.csv)
  synthetic[:JSONL]    hand-written paraphrase / missing-info / cancel records from build_synthetic_agent.py, each with its own label

Run through run.sh so the system prompts and agent flags match inference, e.g.:
  ENV_MODE=official AGENT_ASR_EVIDENCE=1 ./run.sh build_agent_sft.py --source reference \
      --source run:outputs/dual_asr_qwen/benchmark_seed42_n1545 --source generated --normalize-run outputs/dual_asr_qwen/benchmark_seed42_n1545
Splits follow the split manifest (fit -> train.jsonl, validation, test); keep test out of training.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from agent import build_payload, repeat_check

FINISH = {'action': 'finish'}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def reference_history(item):
    """History as process() builds it, but with the reference transcript in place of ASR."""
    history = []
    for original in item.get('history', []):
        turn = {k: v for k, v in original.items() if k != 'audio_text'}
        if turn.get('role') == 'user' and turn.get('audio') and not turn.get('text'):
            turn['text'] = original.get('audio_text', '')
            turn.pop('audio', None)
        history.append(turn)
    return history


def base(text, item, history=None, asr_transcripts=None, examples=None):
    return {'text': text, 'context': item.get('context', {}), 'history': reference_history(item) if history is None else history,
            'asr_transcripts': asr_transcripts, 'examples': examples}


def run_base(run_dir, item, drop_examples):
    """First agent payload of a saved run: exactly what the agent saw before any observation."""
    trace = Path(run_dir) / 'samples' / item['id'] / 'trace.jsonl'
    if not trace.is_file():
        return None
    for entry in jsonl(trace):
        if entry.get('stage') == 'agent' and isinstance(entry.get('input'), dict):
            payload = entry['input']
            history = [dict(turn) for turn in payload.get('history', [])]
            evidence = payload.get('asr_transcripts')
            if os.environ.get('AGENT_ASR_EVIDENCE') == '1':
                from main import asr_evidence
                folder, order = trace.parent, asr_order(run_dir)
                evidence = evidence or asr_evidence(saved_hypotheses(folder, 'asr_', order), payload['normalized_text'])
                for i, turn in enumerate(history):
                    if turn.get('role') == 'user' and turn.get('text') and not turn.get('asr_alternatives'):
                        raw = asr_evidence(saved_hypotheses(folder, f'history_{i}_', order), turn['text'])
                        if raw:
                            turn['asr_alternatives'] = raw
            return base(payload['normalized_text'], item, history, evidence,
                        None if drop_examples else payload.get('examples'))
    return None


def asr_order(run_dir):
    """ASR names in the order main.process puts them in the payload: primary, then secondary in dual mode."""
    metadata = Path(run_dir) / 'run_metadata.json'
    if not metadata.is_file():
        return []
    config = json.loads(metadata.read_text())['identity']['configuration']
    order = [config.get('ASR_PRIMARY')]
    if config.get('ASR_MODE') == 'dual':
        order.append(config.get('ASR_SECONDARY'))
    return [name for name in order if name]


def ordered(hypotheses, order):
    return {**{name: hypotheses[name] for name in order if name in hypotheses},
            **{name: hypotheses[name] for name in sorted(hypotheses) if name not in order}}


def saved_hypotheses(folder, prefix, order=()):
    result = {}
    for path in sorted(folder.glob(prefix + '*.cache.json')):
        name = path.name[len(prefix):-len('.cache.json')]
        value = json.loads(path.read_text())['value']
        if name not in {'normalize', 'timestamp_word'} and isinstance(value, str):
            result[name] = value
    return ordered(result, order)


def generated_variants(path, allow_free_text, allow_unreviewed=False):
    allowed = {'label_safe', 'label_safe_free_text'} if allow_free_text else {'label_safe'}
    by_sample = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if not allow_unreviewed and row.get('manual_decision') != 'pass_text':
                continue
            if row['turn'] == 'last' and row['agent_use'] in allowed:
                by_sample.setdefault(row['sample_id'], []).append(row)
    return by_sample


def replay(env, start, gold):
    """[(payload, decision)] for one trajectory; raises ValueError when a gold call does not validate."""
    observations, steps = [], []
    for call in gold:
        name, args = call['name'], call['arguments']
        if name not in env.schemas:
            raise ValueError(f'gold tool {name} not available')
        if hasattr(env, 'validate') and not (check := env.validate(name, args))['ok']:
            raise ValueError(f'gold call rejected: {check.get("error")}')
        decision = {'action': 'call', 'name': name, 'arguments': args}
        if os.environ.get('AGENT_REPEAT_GUARD') == '1' and (warning := repeat_check(call, observations)):
            payload = build_payload(start['text'], start['context'], start['history'], env, observations,
                                    start['asr_transcripts'], start['examples'])
            steps.append((json.dumps(payload, ensure_ascii=False), decision))
            observations.append(warning)
            if repeat_check(call, observations):
                raise ValueError('Gold trajectory exceeds the runtime repeat limit')
        payload = build_payload(start['text'], start['context'], start['history'], env, observations,
                                start['asr_transcripts'], start['examples'])
        steps.append((json.dumps(payload, ensure_ascii=False), decision))
        observations.append({'tool_call': {'name': name, 'arguments': args}, 'observation': env.execute(name, args)})
        if name == 'sys.ask_missing_info':
            return steps  # run_agent stops right after asking
    payload = build_payload(start['text'], start['context'], start['history'], env, observations,
                            start['asr_transcripts'], start['examples'])
    steps.append((json.dumps(payload, ensure_ascii=False), FINISH))
    return steps


def chat(system, user, assistant):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': json.dumps(assistant, ensure_ascii=False)}]


def failed(observations):
    """True if a replayed call reported an error: such a trajectory would teach a failing call."""
    for entry in observations:
        observation = entry.get('observation')
        if isinstance(observation, dict) and (observation.get('error') or observation.get('ok') is False):
            return True
    return False


def synthetic_records(path):
    """build_synthetic_agent.py records that passed their checks, grouped by seed sample."""
    by_sample = {}
    for record in jsonl(path):
        if record.get('check_ok'):
            by_sample.setdefault(record['sample_id'], []).append(record)
    return by_sample


def agent_records(item, starts, make_env, expected_hash, system):
    """(records, Counter of outcomes) for one sample over all its input variants."""
    records, outcome = [], Counter()
    for source, start in starts:
        env = make_env(item)
        # A start may carry its own label (synthetic records); the label state hash only applies to the seed's gold.
        gold = start.get('tool_calls', item['label']['tool_calls'])
        try:
            steps = replay(env, start, gold)
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the build
            outcome[f'skipped_replay:{type(exc).__name__}'] += 1
            continue
        if any(failed(json.loads(user)['observations']) for user, _ in steps[-1:]):
            outcome['skipped_tool_error'] += 1
            continue
        if start.get('verify_state', True) and expected_hash is not None and env.get_state_hash() != expected_hash:
            outcome['skipped_state_mismatch'] += 1
            continue
        outcome['kept_trajectories'] += 1
        for step, (user, decision) in enumerate(steps):
            records.append({'id': f"{item['id']}:{source}:{step}", 'task': 'agent', 'sample_id': item['id'], 'source': source,
                            'step': step, 'messages': chat(system, user, decision)})
    return records, outcome


def normalize_records(item, run_dir, system, single):
    """ASR hypotheses saved by a run -> reference transcript, for the final turn and user history turns."""
    folder = Path(run_dir) / 'samples' / item['id']
    audios = [('last', 'asr_', item['audio_text'])]
    audios += [(f'history_{i}', f'history_{i}_', turn['audio_text']) for i, turn in enumerate(item.get('history', []))
               if turn.get('role') == 'user' and turn.get('audio_text')]
    records = []
    for turn, prefix, reference in audios:
        hypotheses = {}
        for cache in sorted(folder.glob(prefix + '*.cache.json')):
            name = cache.name[len(prefix):-len('.cache.json')]
            if name in ('normalize', 'timestamp_word'):
                continue
            value = json.loads(cache.read_text(encoding='utf-8'))['value']
            if isinstance(value, str):
                hypotheses[name] = value
        inputs = [hypotheses] if len(hypotheses) > 1 else []
        if single or len(hypotheses) == 1:
            inputs += [{name: text} for name, text in hypotheses.items()]
        for index, transcripts in enumerate(inputs):
            records.append({'id': f"{item['id']}:{turn}:normalize:{index}", 'task': 'normalize', 'sample_id': item['id'],
                            'source': f'run:{run_dir}', 'step': 0,
                            'messages': chat(system, json.dumps({'transcripts': transcripts}, ensure_ascii=False),
                                             {'normalized_text': reference})})
    return records


def normalize_csv_records(rows, system, single, order=('zipformer', 'gipformer')):
    """Saved hypotheses of both ASRs for every audio (e.g. outputs/zipformer_150m/asr_comparison.csv) -> reference.

    The default order matches the dual run (primary zipformer, secondary gipformer) so the payload is identical
    to what agent.normalize receives with ASR_COMBINE=llm_review; --normalize-single adds one record per ASR.
    """
    for row in rows:
        hypotheses = {name: row[f'{name}_asr'] for name in order if row.get(f'{name}_asr', '').strip()}
        turn = row['turn'] + (f"_{row['history_index']}" if row.get('history_index') else '')
        inputs = [hypotheses] if len(hypotheses) > 1 else []
        if single or len(hypotheses) == 1:
            inputs += [{name: text} for name, text in hypotheses.items()]
        for index, transcripts in enumerate(inputs):
            yield {'id': f"{row['id']}:{turn}:normalize:{index}", 'task': 'normalize', 'sample_id': row['id'],
                   'source': 'asr_csv', 'step': 0,
                   'messages': chat(system, json.dumps({'transcripts': transcripts}, ensure_ascii=False),
                                    {'normalized_text': row['reference']})}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', action='append', default=[], help='reference | run:DIR | generated[:CSV]')
    parser.add_argument('--normalize-run', action='append', default=[], help='benchmark run dir with ASR caches')
    parser.add_argument('--normalize-single', action='store_true', help='also one record per single ASR hypothesis')
    parser.add_argument('--normalize-csv', help='CSV with id, turn, history_index, reference, <asr>_asr columns (no devkit needed)')
    parser.add_argument('--free-text', action='store_true', help='keep generated variants whose gold has free-text arguments')
    parser.add_argument('--allow-unreviewed-generated', action='store_true',
                        help='experimental: reuse labels based on anchor matching alone; default requires a manual pass')
    parser.add_argument('--drop-examples', action='store_true', help='remove few-shot examples copied from run traces')
    parser.add_argument('--manifest', default='outputs/timestamp_word_fusion/split_manifest.json')
    parser.add_argument('--out', default='outputs/agent_sft')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not args.source and not args.normalize_run and not args.normalize_csv:
        parser.error('cần ít nhất một --source, --normalize-run hoặc --normalize-csv')
    root = Path(os.environ.get('DATA_ROOT', 'data/train+public_test')).resolve()
    source_dir = root / 'train'
    items = jsonl(source_dir / 'train.jsonl')[:args.limit] if args.source or args.normalize_run else []
    splits = {sid: entry['split'] for sid, entry in json.loads(Path(args.manifest).read_text()).items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    handles = {}
    def write(record):
        split = {'fit': 'train'}.get(splits.get(record['sample_id'], 'unsplit'), splits.get(record['sample_id'], 'unsplit'))
        record['split'] = split
        if split not in handles:
            handles[split] = (out / f'{split}.jsonl').open('w', encoding='utf-8')
        handles[split].write(json.dumps(record, ensure_ascii=False) + '\n')
        stats[f'{record["task"]}_records_{split}'] += 1
    try:
        if args.source:
            os.environ['DATA_SPLIT'] = 'train'
            sys.path.insert(0, str(source_dir))
            from zaic_tools import state_hash
            from benchmark import reference_state
            from env_adapter import OfficialEnvironment
            system = os.environ['AGENT_SYSTEM_PROMPT']
            generated, synthetic = {}, {}
            for spec in args.source:
                if spec.startswith('synthetic'):
                    synthetic = synthetic_records(spec.split(':', 1)[1] if ':' in spec else 'outputs/synthetic_agent/synthetic.jsonl')
                if spec.startswith('generated'):
                    generated = generated_variants(spec.split(':', 1)[1] if ':' in spec else 'outputs/finetune_data_audit/variants.csv',
                                                    args.free_text, args.allow_unreviewed_generated)
            for item in items:
                starts = []
                for spec in args.source:
                    if spec == 'reference':
                        starts.append(('reference', base(item['audio_text'], item)))
                    elif spec.startswith('run:'):
                        start = run_base(spec[4:], item, args.drop_examples)
                        if start: starts.append((spec, start))
                        else: stats['missing_run_trace'] += 1
                    elif spec.startswith('synthetic'):
                        for record in synthetic.get(item['id'], []):
                            start = base(record['text'], item)
                            start.update(tool_calls=record['tool_calls'], verify_state=record['verify_state'])
                            starts.append((f"synthetic:{record['id']}", start))
                    elif spec.startswith('generated'):
                        starts += [(f"generated:{row['variant_id']}", base(row['text'], item)) for row in generated.get(item['id'], [])]
                    else:
                        parser.error(f'--source không hợp lệ: {spec}')
                records, outcome = agent_records(item, starts, lambda item: OfficialEnvironment({'id': item['id']}),
                                                 state_hash(reference_state(source_dir, item)), system)
                stats.update(outcome)
                for record in records:
                    write(record)
        if args.normalize_csv:
            with Path(args.normalize_csv).open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))[:args.limit]
            for record in normalize_csv_records(rows, os.environ['NORMALIZE_SYSTEM_PROMPT'], args.normalize_single):
                write(record)
        if args.normalize_run:
            system = os.environ['NORMALIZE_SYSTEM_PROMPT']
            for item in items:
                for run_dir in args.normalize_run:
                    for record in normalize_records(item, run_dir, system, args.normalize_single):
                        write(record)
    finally:
        for handle in handles.values():
            handle.close()
    summary = {'samples': len(items), **dict(sorted(stats.items())),
               'agent_system_prompt_sha256': hashlib.sha256(os.environ.get('AGENT_SYSTEM_PROMPT', '').encode()).hexdigest(),
               'normalize_system_prompt_sha256': hashlib.sha256(os.environ.get('NORMALIZE_SYSTEM_PROMPT', '').encode()).hexdigest(),
               'sources': args.source, 'normalize_runs': args.normalize_run, 'normalize_csv': args.normalize_csv,
               'note': 'Train on train.jsonl only; validation for early stopping; test is held out for the agent benchmark.'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
