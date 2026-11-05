"""Retrieve solved train requests similar to the current one, to show the agent as examples.

The pool is built offline from labeled train data (build_fewshot_pool.py). At inference only the current
sample's ASR/normalized text and available tools are used for retrieval; its own record and, when a split
manifest is given, every record of the same conversation/template group are excluded.
"""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

TOOL_WEIGHT = 0.3     # added to text cosine: prefer examples from the same tool environment
NEAR_DUPLICATE = 0.85  # skip a candidate this similar to an already chosen example (keeps examples diverse)
MAX_TEXT = 300


def features(text):
    words = re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text or '').casefold())
    return Counter(words + [a + ' ' + b for a, b in zip(words, words[1:])])


def cosine(left, right):
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    return dot / (norm(left) * norm(right)) if dot else 0.0


def norm(vector):
    return math.sqrt(sum(value * value for value in vector.values())) or 1.0


def write_tools(calls):
    return {call['name'] for call in calls if not call['name'].startswith('sys.')}


def clip(value):
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + '…'
    if isinstance(value, list):
        return [clip(item) for item in value]
    if isinstance(value, dict):
        return {key: clip(item) for key, item in value.items()}
    return value


class FewShotIndex:
    def __init__(self, records):
        self.records = records
        counts = [features(record['text']) for record in records]
        df = Counter(key for count in counts for key in count)
        total = len(records)
        self.idf = {key: math.log((total + 1) / (value + 1)) + 1 for key, value in df.items()}
        self.vectors = [self.weigh(count) for count in counts]

    @classmethod
    def load(cls, path):
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        return cls([json.loads(line) for line in lines if line.strip()])

    def weigh(self, count):
        return {key: value * self.idf.get(key, 1.0) for key, value in count.items()}

    def select(self, query_texts, available_tools, k, exclude_ids=(), exclude_groups=(), groups=None, max_similarity=None):
        """max_similarity drops near-copies of the query (pessimistic evaluation on train templates)."""
        groups = groups or {}
        query = self.weigh(features(' '.join(text for text in query_texts if text)))
        available = set(available_tools or ())
        scored = []
        for index, record in enumerate(self.records):
            if record['id'] in exclude_ids or groups.get(record['id']) in exclude_groups:
                continue
            if available and not write_tools(record['tool_calls']) <= available:
                continue  # the example would demonstrate tools this sample does not have
            tools = set(record.get('tools', ()))
            overlap = len(tools & available) / len(tools | available) if tools and available else 0.0
            similarity = cosine(query, self.vectors[index])
            if max_similarity is not None and similarity >= max_similarity:
                continue
            scored.append((similarity + TOOL_WEIGHT * overlap, record['id'], index))
        chosen = []
        for _, _, index in sorted(scored, key=lambda item: (-item[0], item[1])):
            if all(cosine(self.vectors[index], self.vectors[other]) < NEAR_DUPLICATE for other in chosen):
                chosen.append(index)
            if len(chosen) == k:
                break
        return [self.records[index]['id'] for index in chosen], [example(self.records[index]) for index in chosen]


def example(record):
    shown = {'user': clip(record['text'])}
    if record.get('history_tail'):
        shown['history_tail'] = clip(record['history_tail'])
    shown['tool_calls'] = clip(record['tool_calls'])
    return shown


_INDEX = {}
_GROUPS = {}


def select_examples(pool_path, query_texts, available_tools, k, sample_id, manifest_path=None):
    """(ids, examples) for one sample; indexes are cached per path for a whole benchmark process."""
    if pool_path not in _INDEX:
        if not Path(pool_path).is_file():
            raise RuntimeError(f'Thiếu few-shot pool {pool_path}; chạy build_fewshot_pool.py trước hoặc đặt AGENT_FEWSHOT_K=0')
        _INDEX[pool_path] = FewShotIndex.load(pool_path)
    groups = {}
    if manifest_path:
        if manifest_path not in _GROUPS:
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            _GROUPS[manifest_path] = {sid: item['group'] for sid, item in manifest.items()}
        groups = _GROUPS[manifest_path]
    own_group = groups.get(sample_id)
    return _INDEX[pool_path].select(query_texts, available_tools, k, exclude_ids={sample_id},
                                    exclude_groups={own_group} if own_group else set(), groups=groups)
