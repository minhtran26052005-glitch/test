import os
import io
import json
import tempfile
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from main import asr_cache_key, model_fingerprint, recognize, save_cache


class ASRCacheTests(unittest.TestCase):
    def test_cache_only_mode_never_starts_asr_for_a_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'audio.wav'
            audio.write_bytes(b'missing-cache')
            env = {'ZIPFORMER_MODEL_PATH': tmp, 'ASR_CACHE_DIR': '', 'ASR_DEVICE': 'cpu', 'ASR_CACHE_ONLY': '1'}
            with patch.dict(os.environ, env), patch('asr.zipformer.ZipFormer') as model:
                with self.assertRaisesRegex(RuntimeError, 'ASR_CACHE_ONLY=1'):
                    recognize(audio, 'zipformer', root, 'asr_zipformer', lambda **kw: None, True, False)
                model.assert_not_called()

    def test_cached_zipformer_is_used_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / 'model'
            model_path.mkdir()
            (model_path / 'tokens.txt').write_text('<blk> 0')
            shared = root / 'shared'
            shared.mkdir()
            sample = root / 'sample'
            sample.mkdir()
            audio = root / 'audio.wav'
            audio.write_bytes(b'one-audio')
            key = asr_cache_key('zipformer', model_fingerprint(model_path), audio, 'cpu')
            save_cache(shared, key, key, 'thuốc Omeprazole')
            env = {'ZIPFORMER_MODEL_PATH': str(model_path), 'ASR_CACHE_DIR': str(shared), 'ASR_DEVICE': 'cpu'}
            with patch.dict(os.environ, env), patch('asr.zipformer.ZipFormer') as model:
                actual = recognize(audio, 'zipformer', sample, 'history_0_zipformer', lambda **kw: None, True, True, history=True)
                model.assert_not_called()
            self.assertEqual(actual, 'thuốc Omeprazole')
            self.assertTrue((sample / 'history_0_zipformer.cache.json').exists())

    def test_modified_audio_does_not_reuse_old_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'audio.wav'
            audio.write_bytes(b'old-audio')
            key1 = asr_cache_key('zipformer', ['revision-1'], audio, 'cpu')
            audio.write_bytes(b'new-audio')
            key2 = asr_cache_key('zipformer', ['revision-1'], audio, 'cpu')
            key3 = asr_cache_key('zipformer', ['revision-2'], audio, 'cpu')
            self.assertEqual(len({key1, key2, key3}), 3)

    def test_completed_benchmark_resumes_without_inference(self):
        from benchmark import run_benchmark
        from main import digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / 'data' / 'train'
            state_dir = train / 'train_environment' / 'case'
            state_dir.mkdir(parents=True)
            (state_dir / 'state.json').write_text(json.dumps({'_meta': {'tools': []}}))
            item = {'id': 'case', 'audio': 'audio.wav', 'audio_text': 'chuẩn',
                    'label': {'tool_calls': [], 'final_state': {}}}
            (train / 'train.jsonl').write_text(json.dumps(item)+'\n')
            model_path = root / 'model'
            model_path.mkdir()
            (model_path / 'tokens.txt').write_text('<blk> 0')
            env = {'ENV_MODE': 'official', 'DATA_ROOT': str(root / 'data'),
                   'OUTPUT_DIR': str(root / 'outputs'), 'ASR_PRIMARY': 'zipformer',
                   'ZIPFORMER_MODEL_PATH': str(model_path)}
            result = {'tool_calls': [], 'state_hash': digest({}), 'asr_text': 'ASR', 'normalized_text': 'ASR'}
            fake = types.ModuleType('zaic_tools')
            fake.state_hash = digest
            args = Namespace(benchmark=1, seed=42, resume=False)
            with patch.dict('sys.modules', {'zaic_tools': fake}), redirect_stdout(io.StringIO()):
                with patch.dict(os.environ, env), patch('benchmark.process', return_value=result) as infer:
                    run_benchmark(args)
                    infer.assert_called_once()
                args.resume = True
                with patch.dict(os.environ, env), patch('benchmark.process') as infer:
                    run_benchmark(args)
                    infer.assert_not_called()
                manifest = root / 'outputs' / 'benchmark_seed42_n1' / 'results.jsonl'
                self.assertEqual(len(manifest.read_text().splitlines()), 1)
