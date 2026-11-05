import json
import os
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from agent import run_agent
from compare_benchmark_runs import compare
from env_adapter import MockEnvironment
from main import asr_evidence, process

os.environ.update(AGENT_SYSTEM_PROMPT='test', NORMALIZE_SYSTEM_PROMPT='test', AGENT_MAX_NEW_TOKENS='100', NORMALIZE_MAX_NEW_TOKENS='100')
ROOT = Path(__file__).resolve().parents[1]


class ScriptedLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.inputs = []
    def generate(self, system, payload, max_tokens, stage):
        self.inputs.append(payload)
        return json.dumps(next(self.decisions), ensure_ascii=False)


def dual_env(tmp, evidence, combine='primary'):
    return {'OUTPUT_DIR': tmp, 'DEBUG': '0', 'RESUME': '0', 'SAVE_INTERMEDIATE': '0',
            'ASR_MODE': 'dual', 'ASR_PRIMARY': 'zipformer', 'ASR_SECONDARY': 'gipformer', 'ASR_COMBINE': combine,
            'ASR_REVIEW_HISTORY': '0', 'AGENT_ASR_EVIDENCE': evidence, 'DATA_ROOT': tmp, 'DATA_SPLIT': 'train',
            'LLM_MODEL_PATH': tmp, 'LLM_DTYPE': 'bfloat16', 'LLM_TEMPERATURE': '0', 'ENV_MODE': 'mock', 'MAX_AGENT_STEPS': '12'}


HYPOTHESES = {'last.wav': {'zipformer': ' CHUYỂN ÍT TIỀN QUA VÍ CHO HUY', 'gipformer': ' CHUYỂN ÍT TIỀN QUA VÍ CHO UY'},
              'history.wav': {'zipformer': ' NHẮN XÍCH DŨNG', 'gipformer': ' NHẮN SẾP DŨNG'}}


def fake_recognize(audio, name, *args, **kwargs):
    return HYPOTHESES[Path(audio).name][name]


class AgentAsrEvidenceTests(unittest.TestCase):
    def run_process(self, evidence, normalized='Chuyển ít tiền qua ví.'):
        llm = ScriptedLLM([{'normalized_text': normalized}, {'action': 'finish'}])
        args = Namespace(text=None, audio=None, debug=False, resume=False, mode='agent', input='benchmark')
        case = {'id': 'case', 'audio': 'last.wav', 'history': [{'role': 'user', 'audio': 'history.wav', 'audio_text': 'nhãn gold không được lộ'}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, dual_env(tmp, evidence)), \
                patch('main.recognize', side_effect=fake_recognize):
            process(case, args, [llm])
        return llm

    def test_agent_sees_raw_hypotheses_for_final_turn_and_history(self):
        llm = self.run_process('1')
        payload = llm.inputs[-1]
        # The word the normalizer dropped ("Huy") is still visible to the agent.
        self.assertEqual(payload['asr_transcripts'], {'zipformer': 'CHUYỂN ÍT TIỀN QUA VÍ CHO HUY', 'gipformer': 'CHUYỂN ÍT TIỀN QUA VÍ CHO UY'})
        self.assertEqual(payload['history'], [{'role': 'user', 'text': ' NHẮN XÍCH DŨNG',
                                               'asr_alternatives': {'zipformer': 'NHẮN XÍCH DŨNG', 'gipformer': 'NHẮN SẾP DŨNG'}}])
        self.assertEqual(list(payload)[:2], ['normalized_text', 'asr_transcripts'])
        self.assertNotIn('nhãn gold', json.dumps(llm.inputs, ensure_ascii=False))

    def test_flag_off_keeps_previous_agent_payload(self):
        payload = self.run_process('0').inputs[-1]
        self.assertNotIn('asr_transcripts', payload)
        self.assertEqual(payload['history'], [{'role': 'user', 'text': ' NHẮN XÍCH DŨNG'}])

    def test_invalid_flag_rejected(self):
        with self.assertRaisesRegex(ValueError, 'AGENT_ASR_EVIDENCE'):
            self.run_process('yes')

    def test_evidence_skipped_when_only_case_or_punctuation_differs(self):
        self.assertIsNone(asr_evidence({'gipformer': ' BẬT ĐÈN PHÒNG KHÁCH'}, 'Bật đèn phòng khách.'))
        self.assertEqual(asr_evidence({'gipformer': ' BẬT ĐÈN PHÒNG KHÁCH'}, 'Bật đèn.'), {'gipformer': 'BẬT ĐÈN PHÒNG KHÁCH'})
        self.assertIsNone(asr_evidence({}, 'Bật đèn.'))

    def test_timestamp_fusion_passes_both_raw_hypotheses(self):
        llm = ScriptedLLM([{'normalized_text': 'Nhắn Sếp Dũng.'}, {'action': 'finish'}])
        args = Namespace(text=None, audio=None, debug=False, resume=False, mode='agent', input='benchmark')
        fused = {'text': 'NHẮN SẾP DŨNG', 'transcripts': {'gipformer': ' NHẮN SẾP DŨNG', 'zipformer': ' NHẮN XÍCH DŨNG'}}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, dual_env(tmp, '1', 'timestamp_word')), \
                patch('main.recognize_fused', return_value=fused), patch('main.recognize') as single:
            process({'id': 'case', 'audio': 'last.wav'}, args, [llm])
            single.assert_not_called()
        self.assertEqual(llm.inputs[-1]['asr_transcripts'],
                         {'timestamp_word': 'NHẮN SẾP DŨNG', 'gipformer': 'NHẮN SẾP DŨNG', 'zipformer': 'NHẮN XÍCH DŨNG'})

    def test_run_agent_payload(self):
        env = MockEnvironment({})
        llm = ScriptedLLM([{'action': 'finish'}])
        run_agent(llm, env, 'x', {}, [], lambda **kwargs: None, 3, asr_transcripts={'gipformer': 'X'})
        self.assertEqual(llm.inputs[0]['asr_transcripts'], {'gipformer': 'X'})
        llm = ScriptedLLM([{'action': 'finish'}])
        run_agent(llm, env, 'x', {}, [], lambda **kwargs: None, 3)
        self.assertNotIn('asr_transcripts', llm.inputs[0])


class RunShPromptTests(unittest.TestCase):
    def prompt(self, flag):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / 'python'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s" "$AGENT_SYSTEM_PROMPT"\n')
            stub.chmod(0o755)
            env = {**os.environ, 'PYTHON_BIN': str(stub), 'AGENT_ASR_EVIDENCE': flag}
            return subprocess.run(['bash', str(ROOT / 'run.sh')], env=env, check=True, capture_output=True, text=True).stdout

    def test_evidence_rules_only_when_enabled(self):
        off, on = self.prompt('0'), self.prompt('1')
        output_rule = 'Mỗi lượt chỉ trả về {"action":"call","name":"tool_name","arguments":{...}} hoặc {"action":"finish"}.'
        self.assertEqual(off, (ROOT / 'prompts' / 'tool_agent_system.txt').read_text().rstrip('\n'))
        self.assertNotIn('asr_transcripts', off)
        self.assertIn('asr_transcripts', on)
        self.assertTrue(on.endswith(output_rule) and off.endswith(output_rule))
        evidence_rules = (ROOT / 'prompts' / 'tool_agent_asr_evidence.txt').read_text().rstrip('\n')
        self.assertEqual(on, off.removesuffix(output_rule) + evidence_rules + '\n' + output_rule)


class CompareRunsTests(unittest.TestCase):
    def test_paired_changes_by_group(self):
        def row(sid, state, calls=1):
            return {'id': sid, 'state_match': state, 'exact_calls': False, 'expected_tool_calls': [{}] * calls}
        base = [row('a', False), row('b', True), row('c', False, 5)]
        new = [row('a', True), row('b', False), row('c', False, 5)]
        rows, summary = compare(base, new, {'a': 'mot_asr_giu_duoc', 'b': 'khong_lech'})
        self.assertEqual(summary['state_outcome_changes'], {'improved': 1, 'regressed': 1, 'both_failed': 1})
        self.assertEqual(summary['by_keyword_group']['mot_asr_giu_duoc']['improved'], 1)
        self.assertEqual(summary['by_keyword_group']['khong_ro']['samples'], 1)
        self.assertEqual(summary['by_gold_calls']['4+']['samples'], 1)
        with self.assertRaises(ValueError):
            compare(base[:1], new, {})


if __name__ == '__main__':
    unittest.main()
