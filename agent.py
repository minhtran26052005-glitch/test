import json
import os
import time
from jsonschema import Draft202012Validator


def parse_json(raw):
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError('Output phải là JSON object')
    return obj


def normalize(llm, transcripts, trace):
    payload = {'transcripts': transcripts}
    for attempt in range(2):
        raw = llm.generate(os.environ['NORMALIZE_SYSTEM_PROMPT'], payload, int(os.environ['NORMALIZE_MAX_NEW_TOKENS']), 'normalize')
        try:
            result = parse_json(raw)
            if set(result) != {'normalized_text'} or not isinstance(result['normalized_text'], str) or not result['normalized_text'].strip():
                raise ValueError('normalized_text phải là string không rỗng và là khóa duy nhất')
            return result['normalized_text']
        except (ValueError, json.JSONDecodeError) as exc:
            trace(stage='normalize_error', error=str(exc), output=raw)
            payload = {'transcripts': transcripts, 'previous_output': raw, 'error': str(exc)}
    raise ValueError('Qwen không trả JSON normalization hợp lệ sau 2 lần')


def run_agent(llm, env, text, context, history, trace, max_steps, debug=False):
    calls, observations = [], []
    for _ in range(max_steps):
        payload = {'normalized_text': text, 'context': context, 'history': history, 'available_tools': list(env.schemas), 'tool_schemas': getattr(env, 'specs', env.schemas), 'observations': observations}
        raw = llm.generate(os.environ['AGENT_SYSTEM_PROMPT'], payload, int(os.environ['AGENT_MAX_NEW_TOKENS']), 'agent')
        try:
            decision = parse_json(raw)
            if decision == {'action': 'finish'}:
                return calls
            if set(decision) != {'action', 'name', 'arguments'} or decision['action'] != 'call' or not isinstance(decision['name'], str) or not isinstance(decision['arguments'], dict):
                raise ValueError('Quyết định không đúng giao thức')
            name, args = decision['name'], decision['arguments']
            if name not in env.schemas:
                raise ValueError(f'Tool không có sẵn: {name}')
            if any(item.get('tool_call') == {'name': name, 'arguments': args} and isinstance(item.get('observation'), dict) and item['observation'].get('error') for item in observations):
                raise ValueError('Không tự động gọi lại action đã lỗi khi chưa xác định kết quả')
            if hasattr(env, 'validate'):
                validation = env.validate(name, args)
                if not validation['ok']:
                    raise ValueError(validation['error'])
            Draft202012Validator.check_schema(env.schemas[name])
            Draft202012Validator(env.schemas[name]).validate(args)
        except Exception as exc:
            trace(stage='agent_error', error=str(exc), output=raw)
            observations.append({'error': str(exc), 'invalid_decision': raw})
            continue
        start = time.monotonic()
        call = {'name': name, 'arguments': args}
        calls.append(call)
        try:
            obs = env.execute(name, args)
        except Exception as exc:
            # Execution may have had a side effect; do not retry an uncertain action.
            trace(stage='tool', name=name, arguments=args, error=str(exc), latency=time.monotonic()-start)
            raise RuntimeError(f'Kết quả thực thi {name} không xác định; dừng để tránh lặp side effect: {exc}') from exc
        observations.append({'tool_call': call, 'observation': obs})
        trace(stage='tool', name=name, arguments=args, observation=obs, latency=time.monotonic()-start)
        if debug:
            print('[TOOL]', json.dumps(observations[-1], ensure_ascii=False))
        if name == 'sys.ask_missing_info':
            return calls
    raise RuntimeError(f'Agent chưa hoàn tất sau {max_steps} bước')
