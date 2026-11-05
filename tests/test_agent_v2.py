import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from agent import build_payload, repeat_check, run_agent
from audit_finetune_data import assess
from build_agent_sft import normalize_records, replay
from env_adapter import MockEnvironment
from fewshot import FewShotIndex
from llm_qwen import split_thinking
from main import process

os.environ.update(AGENT_SYSTEM_PROMPT='test', NORMALIZE_SYSTEM_PROMPT='test', AGENT_MAX_NEW_TOKENS='100', NORMALIZE_MAX_NEW_TOKENS='100')
ROOT = Path(__file__).resolve().parents[1]
TRACE = lambda **kwargs: None


class ScriptedLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.inputs = []
    def generate(self, system, payload, max_tokens, stage):
        self.inputs.append(payload)
        return json.dumps(next(self.decisions), ensure_ascii=False)


def call(name, **arguments):
    return {'action': 'call', 'name': name, 'arguments': arguments}


class RepeatGuardTests(unittest.TestCase):
    def run_guarded(self, decisions, steps=12):
        env = MockEnvironment({})
        llm = ScriptedLLM(decisions)
        with patch.dict(os.environ, {'AGENT_REPEAT_GUARD': '1'}):
            calls = run_agent(llm, env, 'x', {}, [], TRACE, steps)
        return calls, llm, env

    def test_identical_read_is_not_rerun_and_earlier_result_is_returned(self):
        calls, llm, _ = self.run_guarded([call('smart_home.get_status'), call('smart_home.get_status'), {'action': 'finish'}])
        self.assertEqual([c['name'] for c in calls], ['smart_home.get_status'])
        warning = llm.inputs[2]['observations'][-1]
        self.assertEqual(warning['repeated_call'], {'name': 'smart_home.get_status', 'arguments': {}})
        self.assertEqual(warning['previous_observation'][0]['device_id'], 'ac_living')

    def test_insisting_right_after_warning_runs_once_more_then_blocks(self):
        ride = call('ride.book', pickup='a', destination='b')
        calls, _, env = self.run_guarded([ride, ride, ride, ride, {'action': 'finish'}])
        self.assertEqual(len(calls), 2)  # gold does repeat writes such as redeem twice
        self.assertEqual(len(env.state['rides']), 2)

    def test_guard_off_keeps_old_behaviour(self):
        env = MockEnvironment({})
        calls = run_agent(ScriptedLLM([call('smart_home.get_status')] * 2 + [{'action': 'finish'}]), env, 'x', {}, [], TRACE, 12)
        self.assertEqual(len(calls), 2)

    def test_repeat_check_allows_new_arguments(self):
        observations = [{'tool_call': {'name': 'a', 'arguments': {'q': 1}}, 'observation': {}}]
        self.assertIsNone(repeat_check({'name': 'a', 'arguments': {'q': 2}}, observations))


class AgentLoopTests(unittest.TestCase):
    def test_step_limit_keeps_executed_calls(self):
        env = MockEnvironment({})
        events = []
        calls = run_agent(ScriptedLLM([call('ride.book', pickup='a', destination='b'), 'bad', 'bad']), env, 'x', {}, [],
                          lambda **kwargs: events.append(kwargs), 3)
        self.assertEqual(calls, [{'name': 'ride.book', 'arguments': {'pickup': 'a', 'destination': 'b'}}])
        self.assertEqual(events[-1]['stage'], 'agent_step_limit')

    def test_payload_order_without_extras_is_unchanged(self):
        payload = build_payload('x', {}, [], MockEnvironment({}), [])
        self.assertEqual(list(payload), ['normalized_text', 'context', 'history', 'available_tools', 'tool_schemas', 'observations'])
        payload = build_payload('x', {}, [], MockEnvironment({}), [], {'gip': 'X'}, [{'user': 'y'}])
        self.assertEqual(list(payload)[:3], ['normalized_text', 'asr_transcripts', 'examples'])

    def test_split_thinking(self):
        self.assertEqual(split_thinking('cần làm A\n</think>\n\n{"action":"finish"}'), ('cần làm A', '{"action":"finish"}'))
        self.assertEqual(split_thinking('bị cắt giữa chừng'), (None, 'bị cắt giữa chừng'))


POOL = [
    {'id': 'p1', 'text': 'Bật điều hòa phòng khách 26 độ', 'tools': ['smart_home.get_status', 'smart_home.climate_set'],
     'tool_calls': [{'name': 'smart_home.get_status', 'arguments': {}}, {'name': 'smart_home.climate_set', 'arguments': {'device_id': 'ac_1', 'temperature': 26}}]},
    {'id': 'p2', 'text': 'Bật điều hòa phòng khách 26 độ nhé', 'tools': ['smart_home.climate_set'],
     'tool_calls': [{'name': 'smart_home.climate_set', 'arguments': {'device_id': 'ac_1', 'temperature': 26}}]},
    {'id': 'p3', 'text': 'Đặt xe từ nhà tới công ty', 'tools': ['ride.book'], 'tool_calls': [{'name': 'ride.book', 'arguments': {'pickup': 'nhà', 'destination': 'công ty'}}]},
    {'id': 'p4', 'text': 'Chuyển tiền cho mẹ', 'tools': ['banking.transfer'], 'tool_calls': [{'name': 'banking.transfer', 'arguments': {'amount': 1}}]},
]


class FewShotTests(unittest.TestCase):
    def test_ranking_exclusion_tool_filter_and_diversity(self):
        index = FewShotIndex(POOL)
        tools = ['smart_home.get_status', 'smart_home.climate_set', 'ride.book']
        ids, examples = index.select(['BẬT ĐIỀU HÒA PHÒNG KHÁCH HAI SÁU ĐỘ', 'Bật điều hòa phòng khách 26 độ.'], tools, 3)
        self.assertEqual(ids[0], 'p1')
        self.assertNotIn('p2', ids)  # near copy of p1
        self.assertNotIn('p4', ids)  # banking.transfer is not available here
        self.assertEqual(set(examples[0]), {'user', 'tool_calls'})
        ids, _ = index.select(['bật điều hòa'], tools, 2, exclude_ids={'p1'}, exclude_groups={'g'}, groups={'p2': 'g'})
        self.assertNotIn('p1', ids)
        self.assertNotIn('p2', ids)

    def test_process_passes_examples_and_never_its_own_label(self):
        llm = ScriptedLLM([{'normalized_text': 'Bật điều hòa phòng khách 26 độ.'}, {'action': 'finish'}])
        args = Namespace(text='bật điều hòa phòng khách hai sáu độ', audio=None, debug=False, resume=False, mode='agent', input='benchmark')
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / 'pool.jsonl'
            pool.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in POOL + [dict(POOL[0], id='case')]))
            env = {'OUTPUT_DIR': tmp, 'DEBUG': '0', 'RESUME': '0', 'SAVE_INTERMEDIATE': '0', 'ASR_COMBINE': 'primary',
                   'LLM_MODEL_PATH': tmp, 'LLM_DTYPE': 'bfloat16', 'LLM_TEMPERATURE': '0', 'ENV_MODE': 'mock', 'MAX_AGENT_STEPS': '12',
                   'AGENT_FEWSHOT_K': '2', 'AGENT_FEWSHOT_POOL': str(pool), 'AGENT_FEWSHOT_EXCLUDE': ''}
            with patch.dict(os.environ, env):
                process({'id': 'case'}, args, [llm])
        examples = llm.inputs[-1]['examples']
        self.assertEqual(examples[0]['tool_calls'][1]['name'], 'smart_home.climate_set')
        self.assertEqual(len(examples), 2)
        self.assertTrue(all(e['tool_calls'] != [] for e in examples))


class RunShFlagTests(unittest.TestCase):
    def run_sh(self, env, *argv):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / 'python'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s|ARGS:%s" "$AGENT_SYSTEM_PROMPT" "$*"\n')
            stub.chmod(0o755)
            return subprocess.run(['bash', str(ROOT / 'run.sh'), *argv], env={**os.environ, 'PYTHON_BIN': str(stub), **env},
                                  check=True, capture_output=True, text=True).stdout

    def test_blocks_follow_flags_and_off_is_original_prompt(self):
        off = self.run_sh({}).split('|ARGS:')[0]
        self.assertEqual(off, (ROOT / 'prompts' / 'tool_agent_system.txt').read_text().rstrip('\n'))
        on = self.run_sh({'AGENT_TASK_RULES': '1', 'AGENT_FEWSHOT_K': '3', 'AGENT_REPEAT_GUARD': '1', 'AGENT_THINKING': '1'}).split('|ARGS:')[0]
        for marker in ('Làm đủ yêu cầu:', 'Ví dụ tham khảo:', 'repeated_call', 'Khi suy nghĩ'):
            self.assertIn(marker, on)
            self.assertNotIn(marker, off)
        self.assertTrue(on.endswith('{"action":"finish"}.'))
        for name, marker in (('tool_agent_task_rules.txt', 'Làm đủ yêu cầu:'), ('tool_agent_fewshot.txt', 'Ví dụ tham khảo:')):
            self.assertIn((ROOT / 'prompts' / name).read_text().rstrip('\n'), on, name)

    def test_python_script_passthrough(self):
        self.assertTrue(self.run_sh({}, 'build_agent_sft.py', '--help').endswith('ARGS:build_agent_sft.py --help'))
        self.assertTrue(self.run_sh({}, 'agent', '--id', 'x').endswith('ARGS:main.py agent --id x'))


FAKE_DEVKIT = '''
import copy, hashlib, json
from pathlib import Path

def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

class ToolEnv:
    def __init__(self, split, sample_id, root):
        initial = json.loads((Path(root) / split / f"{split}_environment" / sample_id / "state.json").read_text())
        self.state = {k: copy.deepcopy(v) for k, v in initial.items() if k != "_meta"}
        for name in initial["_meta"]["tools"]:
            if not name.startswith("sys."):
                self.state.setdefault(name.split(".")[0], {})
        obj = {"type": "object", "additionalProperties": False}
        self.tools = {"notes.list": {"parameters": dict(obj, properties={})},
                      "notes.add": {"parameters": dict(obj, properties={"text": {"type": "string"}}, required=["text"])}}
        self.allowed = set(initial["_meta"]["tools"])
    def validate(self, name, arguments):
        return {"ok": name in self.allowed, "error": None if name in self.allowed else "not allowed"}
    def call(self, name, arguments):
        if name == "notes.list":
            return {"items": list(self.state["notes"].get("items", []))}
        self.state["notes"].setdefault("items", []).append(arguments["text"])
        return {"ok": True}
    def get_state_hash(self):
        return state_hash(self.state)
    def get_tool_calls(self):
        return []
    def get_state(self):
        return copy.deepcopy(self.state)
'''


class SftBuilderTests(unittest.TestCase):
    def test_replay_matches_inference_payload_and_ends_with_finish(self):
        env = MockEnvironment({})
        start = {'text': 'đặt xe', 'context': {}, 'history': [], 'asr_transcripts': {'gip': 'ĐẶT XE'}, 'examples': None}
        steps = replay(env, start, [{'name': 'smart_home.get_status', 'arguments': {}}, {'name': 'ride.book', 'arguments': {'pickup': 'a', 'destination': 'b'}}])
        self.assertEqual([d.get('name', d['action']) for _, d in steps], ['smart_home.get_status', 'ride.book', 'finish'])
        last = json.loads(steps[-1][0])
        self.assertEqual(last['asr_transcripts'], {'gip': 'ĐẶT XE'})
        self.assertEqual(last['observations'][0]['observation'][0]['device_id'], 'ac_living')
        ask = replay(MockEnvironment({}), start, [{'name': 'sys.ask_missing_info', 'arguments': {'question': 'Đi đâu?'}}])
        self.assertEqual(ask[-1][1]['name'], 'sys.ask_missing_info')  # run_agent stops after asking: no finish step

    def test_replay_rejects_unavailable_gold_tool(self):
        with self.assertRaises(ValueError):
            replay(MockEnvironment({}), {'text': 'x', 'context': {}, 'history': [], 'asr_transcripts': None, 'examples': None},
                   [{'name': 'banking.transfer', 'arguments': {}}])

    def test_end_to_end_with_fake_devkit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            train = root / 'train'
            (train / 'zaic_tools').mkdir(parents=True)
            (train / 'zaic_tools' / '__init__.py').write_text(FAKE_DEVKIT)
            items = [
                {'id': 's1', 'audio_text': 'Ghi chú mua sữa.', 'context': {'time': 't'},
                 'history': [{'role': 'user', 'audio': 'u1.wav', 'audio_text': 'Xin chào.'}, {'role': 'assistant', 'text': 'Chào bạn.'}],
                 'label': {'tool_calls': [{'name': 'notes.list', 'arguments': {}}, {'name': 'notes.add', 'arguments': {'text': 'mua sữa'}}],
                           'final_state': {'notes': {'items': ['mua sữa']}}}},
                # Wrong label: replay leaves notes empty, so the sample must be dropped.
                {'id': 's2', 'audio_text': 'Xem ghi chú.', 'context': {}, 'history': [],
                 'label': {'tool_calls': [{'name': 'notes.list', 'arguments': {}}], 'final_state': {'notes': {'items': ['x']}}}},
            ]
            (train / 'train.jsonl').write_text('\n'.join(json.dumps(i, ensure_ascii=False) for i in items))
            for item in items:
                folder = train / 'train_environment' / item['id']
                folder.mkdir(parents=True)
                (folder / 'state.json').write_text(json.dumps({'_meta': {'tools': ['notes.list', 'notes.add', 'sys.ask_missing_info']}}))
            run = Path(tmp) / 'run' / 'samples' / 's1'
            run.mkdir(parents=True)
            (run / 'asr_gipformer.cache.json').write_text(json.dumps({'key': 'k', 'value': ' GHI CHÚ MUA XỮA'}))
            (run / 'asr_zipformer.cache.json').write_text(json.dumps({'key': 'k', 'value': ' GHI CHÚ MUA SỮA'}))
            (run / 'history_0_gipformer.cache.json').write_text(json.dumps({'key': 'k', 'value': ' XIN CHÀO'}))
            (run / 'normalize.cache.json').write_text(json.dumps({'key': 'k', 'value': 'Ghi chú mua xữa.'}))
            agent_input = {'normalized_text': 'Ghi chú mua xữa.', 'asr_transcripts': {'gipformer': 'GHI CHÚ MUA XỮA'},
                           'examples': [{'user': 'u', 'tool_calls': []}], 'context': {'time': 't'},
                           'history': [{'role': 'user', 'text': ' XIN CHÀO'}, {'role': 'assistant', 'text': 'Chào bạn.'}]}
            (run / 'trace.jsonl').write_text(json.dumps({'stage': 'agent', 'input': agent_input}, ensure_ascii=False) + '\n')
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps({'s1': {'group': 'a', 'split': 'fit'}, 's2': {'group': 'b', 'split': 'test'}}))
            out = Path(tmp) / 'sft'
            env = {**os.environ, 'DATA_ROOT': str(root), 'AGENT_SYSTEM_PROMPT': 'AGENT', 'NORMALIZE_SYSTEM_PROMPT': 'NORM'}
            subprocess.run([sys.executable, str(ROOT / 'build_agent_sft.py'), '--source', 'reference', '--source', f'run:{Path(tmp) / "run"}',
                            '--drop-examples', '--normalize-run', str(Path(tmp) / 'run'), '--manifest', str(manifest), '--out', str(out)],
                           env=env, check=True, capture_output=True, text=True, cwd=ROOT)
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['kept_trajectories'], 2)
            self.assertEqual(summary['skipped_state_mismatch'], 1)
            self.assertEqual(summary['missing_run_trace'], 1)
            records = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
            agent = [r for r in records if r['task'] == 'agent']
            self.assertEqual(len(agent), 6)  # 2 sources x (list, add, finish)
            reference = [r for r in agent if r['source'] == 'reference']
            payload = json.loads(reference[0]['messages'][1]['content'])
            self.assertEqual(payload['normalized_text'], 'Ghi chú mua sữa.')
            self.assertEqual(payload['history'], [{'role': 'user', 'text': 'Xin chào.'}, {'role': 'assistant', 'text': 'Chào bạn.'}])
            self.assertEqual(json.loads(reference[1]['messages'][1]['content'])['observations'][0]['observation'], {'items': []})
            self.assertEqual(reference[0]['messages'][0], {'role': 'system', 'content': 'AGENT'})
            noisy = json.loads([r for r in agent if r['source'].startswith('run:')][0]['messages'][1]['content'])
            self.assertEqual(noisy['asr_transcripts'], {'gipformer': 'GHI CHÚ MUA XỮA'})
            self.assertNotIn('examples', noisy)
            normalize = [r for r in records if r['task'] == 'normalize']
            targets = {json.loads(r['messages'][2]['content'])['normalized_text'] for r in normalize}
            self.assertEqual(targets, {'Ghi chú mua sữa.', 'Xin chào.'})
            both = json.loads(normalize[0]['messages'][1]['content'])['transcripts']
            self.assertEqual(set(both), {'gipformer', 'zipformer'})


class AuditTests(unittest.TestCase):
    def row(self, **changes):
        row = {'variant_id': 'v1', 'seed_id': 's_last', 'sample_id': 's', 'turn': 'last', 'source_original_split': 'fit',
               'target_model': 'gipformer', 'domain': 'lien_he', 'reference': 'Gọi cho Minh lúc 9 giờ.', 'text': 'Hãy gọi Minh lúc 9 giờ nhé.',
               'spoken_text': 'Hãy gọi Minh lúc chín giờ nhé.', 'slot_changes': '[]'}
        row.update(changes)
        return row

    def test_agent_label_reuse_requires_anchor_tokens(self):
        gold = {'s': [{'name': 'contacts.search', 'arguments': {'query': 'Minh'}}]}
        self.assertEqual(assess(self.row(), gold, {}, set())['agent_use'], 'label_safe')
        lost = assess(self.row(text='Hãy gọi Lan lúc 9 giờ.', spoken_text='Hãy gọi Lan lúc chín giờ.'), gold, {}, set())
        self.assertEqual((lost['agent_use'], lost['missing_anchor_tokens']), ('needs_new_label', 'minh'))
        swapped = assess(self.row(slot_changes='[{"from": "Minh", "to": "Lan"}]'), gold, {}, set())
        self.assertEqual(swapped['agent_use'], 'needs_new_label')
        self.assertEqual(swapped['asr_use'], 'ok')  # an entity swap is still good ASR text
        sms = {'s': [{'name': 'sms.send', 'arguments': {'contact_id': 'c1', 'content': 'gọi lúc 9 giờ'}}]}
        self.assertEqual(assess(self.row(), sms, {}, set())['agent_use'], 'label_safe_free_text')

    def test_asr_rejects(self):
        self.assertEqual(assess(self.row(spoken_text='Gọi Минh'), {}, {}, set())['asr_use'], 'reject')
        self.assertEqual(assess(self.row(), {}, {'v1': 'reject_text'}, set())['asr_use'], 'reject')
        self.assertEqual(assess(self.row(turn='history'), {}, {}, set())['agent_use'], 'history_paraphrase')


if __name__ == '__main__':
    unittest.main()
