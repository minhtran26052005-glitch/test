import json
import os
import unittest
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from main import process
from agent import run_agent, normalize
from env_adapter import MockEnvironment

os.environ.update(AGENT_SYSTEM_PROMPT='test', NORMALIZE_SYSTEM_PROMPT='test', AGENT_MAX_NEW_TOKENS='100', NORMALIZE_MAX_NEW_TOKENS='100')

class ScriptedLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.inputs = []
    def generate(self, system, payload, max_tokens, stage):
        self.inputs.append(payload)
        value = next(self.decisions)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def call(name, **kwargs):
    return {'action': 'call', 'name': name, 'arguments': kwargs}

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.env = MockEnvironment({})
        self.trace = lambda **kwargs: None
    def run_script(self, llm):
        return run_agent(llm, self.env, 'test', {'location': '227 Nguyễn Văn Cừ, Quận 5, TP.HCM', 'home': '123 Trần Hưng Đạo, Quận 5, TP.HCM'}, [], self.trace, 12)
    def test_multi_step(self):
        decisions = [call('smart_home.get_status'), call('smart_home.climate_set', device_id='ac_living', temperature=26), call('ride.book', pickup='227 Nguyễn Văn Cừ, Quận 5, TP.HCM', destination='123 Trần Hưng Đạo, Quận 5, TP.HCM'), {'action': 'finish'}]
        llm = ScriptedLLM(decisions)
        calls = self.run_script(llm)
        self.assertEqual([c['name'] for c in calls], ['smart_home.get_status', 'smart_home.climate_set', 'ride.book'])
        self.assertEqual(llm.inputs[1]['observations'][0]['observation'][0]['device_id'], calls[1]['arguments']['device_id'])
        self.assertEqual(self.env.state['devices'][0]['temperature'], 26)
        self.assertEqual(self.env.state['rides'][0]['destination'], '123 Trần Hưng Đạo, Quận 5, TP.HCM')
    def test_missing(self):
        calls = self.run_script(ScriptedLLM([call('sys.ask_missing_info', question='Bạn muốn đến đâu?')]))
        self.assertEqual(calls, [{'name': 'sys.ask_missing_info', 'arguments': {'question': 'Bạn muốn đến đâu?'}}])
        self.assertEqual(self.env.state['rides'], [])
    def test_out_of_scope(self):
        self.assertEqual(self.run_script(ScriptedLLM([{'action': 'finish'}])), [])
    def test_invalid_not_executed(self):
        llm = ScriptedLLM(['bad json', call('ride.book', pickup='đây'), {'action': 'finish'}])
        self.assertEqual(self.run_script(llm), [])
        self.assertEqual(self.env.state['rides'], [])
        self.assertEqual(len(llm.inputs[2]['observations']), 2)
    def test_normalize_retry(self):
        llm = ScriptedLLM(['not json', {'normalized_text': 'Bật điều hòa 26 độ.'}])
        self.assertEqual(normalize(llm, {'provided': 'bật điều hòa hai mươi sáu độ'}, self.trace), 'Bật điều hòa 26 độ.')
    def test_process_saves_result_and_trace(self):
        llm = ScriptedLLM([{'normalized_text': 'Kiểm tra điều hòa.'}, call('smart_home.get_status'), {'action': 'finish'}])
        args = Namespace(text='kiểm tra điều hòa', audio=None, debug=False, resume=False, mode='agent', input=None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OUTPUT_DIR': tmp, 'DEBUG': '0', 'RESUME': '0', 'SAVE_INTERMEDIATE': '1', 'ASR_COMBINE': 'primary', 'LLM_MODEL_PATH': '/missing/model', 'LLM_DTYPE': 'bfloat16', 'LLM_TEMPERATURE': '0', 'ENV_MODE': 'mock', 'MAX_AGENT_STEPS': '12'}):
            result = process({'id': 'case'}, args, [llm])
            stored = json.loads((Path(tmp) / 'case' / 'result.json').read_text())
            self.assertEqual(stored, result)
            self.assertEqual(result['tool_calls'][0]['name'], 'smart_home.get_status')
            self.assertTrue(result['state_hash'])
            self.assertIn('smart_home.get_status', (Path(tmp) / 'case' / 'trace.jsonl').read_text())

    def test_history_audio_uses_asr_not_reference_transcript(self):
        llm = ScriptedLLM([{'normalized_text': 'Tiếp tục.'}, {'action': 'finish'}])
        args = Namespace(text='tiếp tục', audio=None, debug=False, resume=False, mode='agent', input='benchmark')
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'history.wav'
            audio.write_bytes(b'test-audio')
            case = {'id': 'case', 'history': [{'role': 'user', 'audio': str(audio), 'audio_text': 'nhãn không được đưa vào model'}]}
            env = {'OUTPUT_DIR': tmp, 'DEBUG': '0', 'RESUME': '0', 'SAVE_INTERMEDIATE': '0', 'ASR_COMBINE': 'primary', 'ASR_PRIMARY': 'gipformer', 'ASR_DEVICE': 'cpu', 'GIPFORMER_MODEL_PATH': tmp, 'LLM_MODEL_PATH': tmp, 'LLM_DTYPE': 'bfloat16', 'LLM_TEMPERATURE': '0', 'ENV_MODE': 'mock', 'MAX_AGENT_STEPS': '12'}
            with patch.dict(os.environ, env), patch('asr.gipformer.GipFormer') as model:
                model.return_value.transcribe.return_value = 'lời ASR'
                process(case, args, [llm])
            history = llm.inputs[-1]['history']
            self.assertEqual(history, [{'role': 'user', 'text': 'lời ASR'}])
            self.assertNotIn('audio_text', history[0])

    def test_uncertain_exception_stops(self):
        def fail(name, args): raise RuntimeError('timeout')
        self.env.execute = fail
        with self.assertRaisesRegex(RuntimeError, 'không xác định'):
            self.run_script(ScriptedLLM([call('ride.book', pickup='a', destination='b')]))

    def test_dual_asr_history_review_uses_both_predictions_without_reference(self):
        llm = ScriptedLLM([{'normalized_text': 'Tiếp tục.'},
                           {'normalized_text': 'Đặt báo thức sáu mươi phút sau.'}, {'action': 'finish'}])
        args = Namespace(text='tiếp tục', audio=None, debug=False, resume=False, mode='agent', input='benchmark')
        with tempfile.TemporaryDirectory() as tmp:
            case = {'id': 'case', 'history': [{'role': 'user', 'audio': 'history.wav', 'audio_text': 'gold tuyệt đối không được dùng'}]}
            env = {'OUTPUT_DIR': tmp, 'DEBUG': '0', 'RESUME': '0', 'SAVE_INTERMEDIATE': '0',
                   'ASR_MODE': 'dual', 'ASR_PRIMARY': 'zipformer', 'ASR_SECONDARY': 'gipformer',
                   'ASR_COMBINE': 'llm_review', 'ASR_REVIEW_HISTORY': '1',
                   'DATA_ROOT': tmp, 'DATA_SPLIT': 'train', 'LLM_MODEL_PATH': tmp,
                   'LLM_DTYPE': 'bfloat16', 'LLM_TEMPERATURE': '0', 'ENV_MODE': 'mock', 'MAX_AGENT_STEPS': '12'}
            def recognize(audio, name, *args, **kwargs):
                return {'zipformer': 'đặt báo thức sáu phút sau', 'gipformer': 'đặt báo thức sáu mươi phút sau'}[name]
            with patch.dict(os.environ, env), patch('main.recognize', side_effect=recognize):
                process(case, args, [llm])
            self.assertEqual(set(llm.inputs[1]['transcripts']), {'zipformer', 'gipformer'})
            self.assertEqual(llm.inputs[-1]['history'], [{'role': 'user', 'text': 'Đặt báo thức sáu mươi phút sau.'}])
            self.assertNotIn('gold tuyệt đối', json.dumps(llm.inputs, ensure_ascii=False))

    def test_timestamp_fusion_history_reaches_agent_without_primary_substitution(self):
        llm = ScriptedLLM([{'normalized_text': 'Tiếp tục.'}, {'action': 'finish'}])
        args = Namespace(text='tiếp tục',audio=None,debug=False,resume=False,mode='agent',input='benchmark')
        with tempfile.TemporaryDirectory() as tmp:
            case={'id':'case','history':[{'role':'user','audio':'history.wav','audio_text':'reference must stay out'}]}
            env={'OUTPUT_DIR':tmp,'DEBUG':'0','RESUME':'0','SAVE_INTERMEDIATE':'0',
                 'ASR_COMBINE':'timestamp_word','ASR_PRIMARY':'zipformer','DATA_ROOT':tmp,'DATA_SPLIT':'train',
                 'LLM_MODEL_PATH':tmp,'LLM_DTYPE':'bfloat16','LLM_TEMPERATURE':'0','ENV_MODE':'mock','MAX_AGENT_STEPS':'12'}
            with patch.dict(os.environ,env),patch('main.recognize_fused',return_value={'text':'giữ đủ số và tên','transcripts':{'gipformer':'GIỮ ĐỦ SỐ VÀ TÊN','zipformer':'GIỮ ĐỦ SỐ VÀ TÊM'}}) as fused,patch('main.recognize') as single:
                process(case,args,[llm])
                fused.assert_called_once();single.assert_not_called()
            self.assertEqual(llm.inputs[-1]['history'],[{'role':'user','text':'giữ đủ số và tên'}])
            self.assertNotIn('reference must stay out',json.dumps(llm.inputs,ensure_ascii=False))

if __name__ == '__main__': unittest.main()
