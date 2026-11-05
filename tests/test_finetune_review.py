import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from asr_lora.data import read_manifests
from build_agent_sft import generated_variants, replay, run_base
from prepare_asr_finetune_data import tts_requests
from train_qwen_lora import configure_generation
from synthesize_tts import synthesize_request


class DataReviewTests(unittest.TestCase):
    def test_tts_pronunciation_does_not_replace_acronym_label(self):
        variant = {'variant_id':'s_v1','sample_id':'s','asr_use':'ok','spoken_text':'Nhắn ét em ét cho Lan.',
                   'turn':'last','target_model':'zipformer','domain':'lien_he'}
        written = {'variant_id':'s_v1','source_audio':'data/train+public_test/train/s.wav','text':'Nhắn SMS cho Lan.'}
        record = next(tts_requests([variant],[written],{'data/train+public_test/train/s.wav':'Xin chào.'},{'s':'fit'}))
        self.assertEqual(record['text'],'NHẮN SMS CHO LAN')
        self.assertIn('ét em ét',record['tts_text'])

    def test_training_keeps_shared_real_audio_but_filters_tts_by_model(self):
        records = [{'id':'r','audio':'r.wav','source':'train_audio'},
                   {'id':'z','audio':'z.wav','source':'tts','target_model':'zipformer'},
                   {'id':'g','audio':'g.wav','source':'tts','target_model':'gipformer'}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            self.assertEqual([r['id'] for r in read_manifests([path],target_model='gipformer')],['r','g'])
            self.assertEqual([r['id'] for r in read_manifests([path],target_model='zipformer')],['r','z'])

    def test_voice_cloning_gets_the_reference_audio_and_its_transcript(self):
        class TTS:
            def infer(self, text, *, ref_audio, ref_text, apply_watermark):
                self.received = (text, ref_audio, ref_text, apply_watermark)
                return 'audio'
        tts = TTS()
        request = {'id': 's', 'tts_text': 'Câu mới.', 'speaker_text': 'Câu trong audio gốc.'}
        self.assertEqual(synthesize_request(tts, request, Path('s.wav')), 'audio')
        self.assertEqual(tts.received, ('Câu mới.', 's.wav', 'Câu trong audio gốc.', False))
        with self.assertRaisesRegex(ValueError, 'speaker_text'):
            synthesize_request(tts, {**request, 'speaker_text': ''}, Path('s.wav'))

    def test_unreviewed_anchor_match_is_not_a_gold_agent_label(self):
        rows = [{'sample_id':'s','turn':'last','agent_use':'label_safe','manual_decision':'','variant_id':'unchecked'},
                {'sample_id':'s','turn':'last','agent_use':'label_safe','manual_decision':'pass_text','variant_id':'reviewed'}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'variants.csv'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
            self.assertEqual([r['variant_id'] for r in generated_variants(path,False)['s']],['reviewed'])
            self.assertEqual(len(generated_variants(path,False,allow_unreviewed=True)['s']),2)

    def test_no_improvement_does_not_try_to_export_a_missing_adapter(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            tmp=Path(folder);shutil.copyfile(root/'run_finetune.sh',tmp/'run_finetune.sh')
            for name in ['model.pt','bpe.model','jit_script.pt']:(tmp/name).touch()
            log=tmp/'calls.txt';fake=tmp/'python'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> "$SMOKE_CALL_LOG"\n')
            fake.chmod(0o755)
            env={**os.environ,'PYTHON_BIN':str(fake),'SMOKE_CALL_LOG':str(log),
                 'GIPFORMER_MODEL_PATH':str(tmp),'ZIPFORMER_MODEL_PATH':str(tmp)}
            for kind in ['gipformer','zipformer']:
                subprocess.run(['bash',str(tmp/'run_finetune.sh'),kind],env=env,check=True,capture_output=True)
            self.assertEqual(log.read_text().splitlines(),['train_asr_lora.py','train_asr_lora.py'])

    def test_chat_terminator_is_in_generation_stop_ids(self):
        class Tokenizer:
            pad_token_id=248044
            def convert_tokens_to_ids(self,text):return 248046
            def __call__(self,text,**kwargs):return {'input_ids':[248046]}
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044,pad_token_id=None))
        self.assertEqual(configure_generation(model,Tokenizer()),[248046,248044])

    def test_previous_run_best_is_preserved_without_becoming_current_best(self):
        try:
            from train_asr_lora import archive_best_adapter
        except ModuleNotFoundError as exc:
            if exc.name == 'torch':
                self.skipTest('torch not installed')
            raise
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'lora-best.pt').write_bytes(b'previous checkpoint')
            archive_best_adapter(root)
            self.assertFalse((root/'lora-best.pt').exists())
            archived = list(root.glob('lora-best.previous-*.pt'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), b'previous checkpoint')

    def test_old_agent_trace_can_recover_raw_asr_evidence_from_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            sample=Path(folder)/'samples/s';sample.mkdir(parents=True)
            payload={'normalized_text':'Nhắn Minh.','history':[{'role':'user','text':'Gọi Lan.'}]}
            (sample/'trace.jsonl').write_text(json.dumps({'stage':'agent','input':payload})+'\n')
            for name,value in [('asr_gipformer',' NHẮN MÌNH'),('asr_zipformer',' NHẮN MINH'),
                               ('history_0_gipformer',' GỌI LÀN')]:
                (sample/f'{name}.cache.json').write_text(json.dumps({'value':value}))
            with patch.dict(os.environ,AGENT_ASR_EVIDENCE='1'):
                start=run_base(folder,{'id':'s'},True)
            self.assertEqual(set(start['asr_transcripts']),{'gipformer','zipformer'})
            self.assertIn('asr_alternatives',start['history'][0])

    def test_replay_includes_the_same_repeat_warning_as_runtime(self):
        class CounterEnv:
            schemas={'loyalty.redeem':{'type':'object'}}
            count=0
            def execute(self,name,args):
                self.count+=1
                return {'count':self.count}
        env=CounterEnv();call={'name':'loyalty.redeem','arguments':{}}
        start={'text':'Đổi hai lần.','context':{},'history':[],'asr_transcripts':None,'examples':None}
        with patch.dict(os.environ,AGENT_REPEAT_GUARD='1'):
            steps=replay(env,start,[call,call])
        self.assertEqual(env.count,2)
        self.assertEqual(len(steps),4)
        self.assertEqual(json.loads(steps[2][0])['observations'][-1]['repeated_call'],call)


class NumericalReviewTests(unittest.TestCase):
    def test_jit_parity_rejects_nan_instead_of_treating_it_as_close(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed')
        from asr_lora.models import compare_with_jit
        class Model:
            def forward_encoder(self,x,lengths):return torch.full((2,3,4),float('nan')),lengths
        jit=SimpleNamespace(encoder=lambda x,lengths:(torch.zeros(2,3,4),lengths))
        with self.assertRaisesRegex(RuntimeError,'Non-finite'):
            compare_with_jit(Model(),jit)


if __name__=='__main__':
    unittest.main()
