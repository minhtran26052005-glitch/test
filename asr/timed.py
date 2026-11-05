"""Detailed transducer cache and BPE-to-space-unit reconstruction."""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'rnnt-timed-v1'
DECODER = {'decoding_method': 'greedy_search', 'sample_rate': 16000,
           'feature_dim': 80, 'num_threads': 4, 'provider': 'cpu'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def to_words(detail):
    tokens, times, scores = detail['tokens'], detail['timestamps'], detail['ys_log_probs']
    duration = detail['duration_seconds']
    if not len(tokens) == len(times) == len(scores):
        raise ValueError('Token, timestamp and log probability lengths differ')
    if any(not math.isfinite(t) or t < 0 or t > duration + .1 for t in times):
        raise ValueError('Invalid token timestamp')
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError('Non-monotone token timestamps')
    if any(not math.isfinite(score) or score > .001 for score in scores):
        raise ValueError('Invalid token log probability')
    words = []
    for index, (token, timestamp, logp) in enumerate(zip(tokens, times, scores)):
        token = token.replace('▁', ' ')
        if not token:
            continue
        if token.startswith(' ') or not words:
            words.append({'text': token.strip(), 'start': timestamp, 'last_token_time': timestamp,
                          'token_indices': [index], 'log_probs': [logp]})
        else:
            words[-1]['text'] += token
            words[-1]['last_token_time'] = timestamp
            words[-1]['token_indices'].append(index)
            words[-1]['log_probs'].append(logp)
    for index, word in enumerate(words):
        word['end_estimated'] = words[index+1]['start'] if index+1 < len(words) else duration
        word['end_is_estimated'] = True
        word['mean_logp'] = sum(word['log_probs']) / len(word['log_probs'])
        word['min_logp'] = min(word['log_probs'])
    if ''.join(w['text'] for w in words).casefold() != ''.join(detail['text'].split()).casefold():
        raise ValueError('BPE reconstruction does not match decoder text')
    return words


def cache_identity(audio_path, fingerprint, runtime_version):
    audio_sha = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
    identity = {'schema': SCHEMA, 'model_fingerprint': fingerprint, 'decoder': DECODER,
                'runtime_version': runtime_version, 'audio_sha256': audio_sha}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return key, identity
