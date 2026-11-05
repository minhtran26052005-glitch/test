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


def build_payload(text, context, history, env, observations, asr_transcripts=None, examples=None):
    """Agent input for one step; shared with build_agent_sft.py so training data matches inference exactly."""
    payload = {'normalized_text': text}
    if asr_transcripts:
        # Raw ASR hypotheses of the final turn (AGENT_ASR_EVIDENCE=1) let the agent recover words the normalizer dropped.
        payload['asr_transcripts'] = asr_transcripts
    if examples:
        payload['examples'] = examples  # solved train requests similar to this one (AGENT_FEWSHOT_K>0)
    payload.update(context=context, history=history, available_tools=list(env.schemas),
                   tool_schemas=getattr(env, 'specs', env.schemas), observations=observations)
    return payload


def repeat_check(call, observations):
    """None when the call may run; otherwise the observation that replaces it.

    Gold sometimes repeats a write on purpose (redeem twice), so the first identical repeat only gets a warning
    with the earlier result; re-sending it right after that warning runs it, at most twice in total.
    """
    executed = [item for item in observations if item.get('tool_call') == call]
    if not executed:
        return None
    last = observations[-1] if observations else {}
    if len(executed) == 1 and last.get('repeated_call') == call:
        return None
    if len(executed) >= 2:
        warning = 'Lời gọi này đã được thực hiện 2 lần; không thực hiện thêm. Dùng kết quả đã có hoặc finish.'
    else:
        warning = ('Lời gọi này giống hệt một lời gọi đã thực hiện; kết quả nằm ở previous_observation. '
                   'Không gọi lại tool đọc với cùng tham số. Chỉ khi người dùng thật sự yêu cầu làm hành động này thêm một lần, '
                   'gửi lại đúng lời gọi này ở bước kế tiếp.')
    return {'repeated_call': call, 'warning': warning, 'previous_observation': executed[-1].get('observation')}


def run_agent(llm, env, text, context, history, trace, max_steps, debug=False, asr_transcripts=None, examples=None):
    calls, observations = [], []
    repeat_guard = os.environ.get('AGENT_REPEAT_GUARD', '0') == '1'
    for _ in range(max_steps):
        payload = build_payload(text, context, history, env, observations, asr_transcripts, examples)
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
        call = {'name': name, 'arguments': args}
        if repeat_guard and (blocked := repeat_check(call, observations)):
            trace(stage='agent_repeat', name=name, arguments=args, warning=blocked['warning'])
            observations.append(blocked)
            continue
        start = time.monotonic()
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
    # Executed calls already changed the environment: keep them and the real state hash instead of losing the sample.
    trace(stage='agent_step_limit', max_steps=max_steps, executed_calls=len(calls))
    return calls
