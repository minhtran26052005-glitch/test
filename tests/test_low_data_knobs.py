"""Small-data options: decode-time hotwords, LoRA interpolation, rejection-sampled SFT, paired significance."""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from asr import decoding_spec
from asr.gipformer import bpe_vocab, decoding_identity, decoding_options
from build_hotwords import foreign_words, is_vietnamese_syllable, miss_counts, select
from compare_benchmark_runs import compare, mcnemar_exact
from eval_asr_onnx import keyword_scores, parse_spec
from main import asr_cache_key, digest

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_PEFT = HAS_TORCH and importlib.util.find_spec('peft') is not None
HAS_SPM = importlib.util.find_spec('sentencepiece') is not None
HAS_LHOTSE = all(importlib.util.find_spec(m) is not None for m in ('lhotse', 'soundfile', 'numpy'))


class DecodingOptionTests(unittest.TestCase):
    def test_default_is_greedy_and_keeps_old_cache_keys(self):
        self.assertEqual(decoding_options('GIPFORMER', {}), {})
        self.assertIsNone(decoding_identity({}))
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'a.wav'
            audio.write_bytes(b'RIFF')
            old = digest(['gipformer', ['fp'], hashlib.sha256(b'RIFF').hexdigest(), 'cpu'])
            self.assertEqual(asr_cache_key('gipformer', ['fp'], audio, 'cpu'), old)
            self.assertEqual(asr_cache_key('gipformer', ['fp'], audio, 'cpu', None), old)
            beam = {'decoding_method': 'modified_beam_search', 'max_active_paths': 4}
            self.assertNotEqual(asr_cache_key('gipformer', ['fp'], audio, 'cpu', beam), old)

    def test_beam_and_hotwords(self):
        with tempfile.TemporaryDirectory() as tmp:
            words = Path(tmp) / 'hot.txt'
            words.write_text('SJC\nZALO\n')
            env = {'ZIPFORMER_DECODING': 'modified_beam_search', 'ZIPFORMER_HOTWORDS': str(words), 'ZIPFORMER_HOTWORDS_SCORE': '2'}
            options = decoding_options('ZIPFORMER', env)
            self.assertEqual(options, {'decoding_method': 'modified_beam_search', 'max_active_paths': 4,
                                       'hotwords_file': str(words), 'hotwords_score': 2.0})
            identity = decoding_identity(options)
            self.assertNotIn('hotwords_file', identity)  # the list's content, not its path, defines the transcripts
            self.assertEqual(identity['hotwords_sha256'], hashlib.sha256(b'SJC\nZALO\n').hexdigest())
            self.assertEqual(decoding_spec('zipformer', env), options)
            for bad in ({'GIPFORMER_HOTWORDS': str(words)}, {'GIPFORMER_BEAM': '8'}, {'GIPFORMER_DECODING': 'beam'},
                        {'GIPFORMER_DECODING': 'modified_beam_search', 'GIPFORMER_HOTWORDS': str(Path(tmp) / 'missing.txt')}):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    decoding_options('GIPFORMER', bad)
            with self.assertRaises(ValueError):
                decoding_spec('phowhisper_large', {'PHOWHISPER_HOTWORDS': str(words)})
            self.assertEqual(decoding_spec('phowhisper_large', {}), {})

    def test_recognizer_receives_the_options(self):
        calls = []
        fake = types.ModuleType('sherpa_onnx')
        fake.OfflineRecognizer = types.SimpleNamespace(from_transducer=lambda **kw: calls.append(kw) or object())
        hub = types.ModuleType('huggingface_hub')
        hub.snapshot_download = None
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'sherpa_onnx': fake, 'huggingface_hub': hub}):
            for name in ('encoder.onnx', 'decoder.onnx', 'joiner.onnx', 'tokens.txt'):
                (Path(tmp) / name).write_text('x')
            (Path(tmp) / 'bpe.vocab').write_text('<blk>\t0\n▁SJC\t-1\n')
            words = Path(tmp) / 'hot.txt'
            words.write_text('SJC\n')
            from asr.gipformer import GipFormer
            GipFormer(tmp, 'cpu')
            GipFormer(tmp, 'cpu', {'decoding_method': 'modified_beam_search', 'max_active_paths': 4,
                                   'hotwords_file': str(words), 'hotwords_score': 1.5})
        self.assertEqual(calls[0]['decoding_method'], 'greedy_search')
        self.assertNotIn('hotwords_file', calls[0])
        self.assertEqual((calls[1]['decoding_method'], calls[1]['modeling_unit'], calls[1]['bpe_vocab']),
                         ('modified_beam_search', 'bpe', str(Path(tmp) / 'bpe.vocab')))

    @unittest.skipUnless(HAS_SPM, 'sentencepiece not installed')
    def test_bpe_vocab_is_exported_outside_the_model_dir(self):
        import sentencepiece as spm
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / 'corpus.txt'
            corpus.write_text('\n'.join(['VÀNG SJC BÁN RA', 'MỞ ZALO NHẮN TIN', 'NHẮN SMS CHO LAN'] * 20))
            model_dir = Path(tmp) / 'model'
            model_dir.mkdir()
            spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(model_dir / 'bpe'), vocab_size=32,
                                           model_type='bpe', character_coverage=1.0, hard_vocab_limit=False)
            pieces = spm.SentencePieceProcessor(model_file=str(model_dir / 'bpe.model')).get_piece_size()
            (model_dir / 'bpe.vocab').unlink()  # the trainer writes one; a model dir without it is the case to cover
            before = sorted(p.name for p in model_dir.iterdir())
            path = bpe_vocab(model_dir)
            self.assertEqual(sorted(p.name for p in model_dir.iterdir()), before)  # model fingerprint unchanged
            lines = Path(path).read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), pieces)
            self.assertTrue(all(len(line.split('\t')) == 2 for line in lines))
            self.assertEqual(bpe_vocab(model_dir), path)


class HotwordTests(unittest.TestCase):
    def test_vietnamese_syllables_are_not_candidates(self):
        for word in ('NGHIÊNG', 'KHUYA', 'LAN', 'XOONG', 'GÌ', 'QUẢ', 'ĐƯỜNG', 'NGƯỜI', 'HUYỆN', 'GIỮA', 'KHUỶU', 'NGOẶT', 'BA'):
            self.assertTrue(is_vietnamese_syllable(word), word)
        for word in ('SJC', 'ZALO', 'SMS', 'TIVI', 'OMEPRAZOLE', 'MEETING', 'CHECK', 'OK', 'EMAIL', 'HTV'):
            self.assertFalse(is_vietnamese_syllable(word), word)
        self.assertEqual(foreign_words('VÀNG SJC BÁN RA E NĂM HTV BẢY'.split()), ['SJC', 'HTV'])

    def test_selection_uses_misses(self):
        rows = [('VÀNG SJC BÁN RA', {'gipformer': 'VÀNG SC BÁN RA'}), ('GIÁ VÀNG SJC', {'gipformer': 'GIÁ VÀNG SJC'}),
                ('VÀNG SJC HÔM NAY', {'gipformer': 'VÀNG SÊ HÔM NAY'}), ('MỞ ZALO', {'gipformer': 'MỞ ZALO'}),
                ('MỞ ZALO NHA', {'gipformer': 'MỞ ZALO NHA'}), ('NHẮN ZALO', {'gipformer': 'NHẮN ZALO'})]
        seen, missed = miss_counts(rows, 'gipformer')
        self.assertEqual((seen['SJC'], missed['SJC'], seen['ZALO'], missed['ZALO']), (3, 2, 3, 0))
        self.assertEqual(select(seen, missed, 3, 0.3), ['SJC'])
        self.assertEqual(select(seen, missed, 4, 0.3), [])

    def test_builder_reads_only_the_fit_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / 'asr.csv'
            with table.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, ['id', 'reference', 'gipformer_asr', 'zipformer_asr'])
                writer.writeheader()
                for i in range(3):
                    writer.writerow({'id': f'f{i}', 'reference': 'Vàng SJC bán ra', 'gipformer_asr': 'VÀNG SC BÁN RA', 'zipformer_asr': 'VÀNG SJC BÁN RA'})
                    writer.writerow({'id': f't{i}', 'reference': 'Mở Zalo', 'gipformer_asr': 'MỞ DA LÔ', 'zipformer_asr': 'MỞ DA LÔ'})
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps({**{f'f{i}': {'split': 'fit'} for i in range(3)}, **{f't{i}': {'split': 'test'} for i in range(3)}}))
            out = Path(tmp) / 'hot'
            subprocess.run([sys.executable, 'build_hotwords.py', '--asr', str(table), '--manifest', str(manifest), '--out', str(out)],
                           cwd=ROOT, check=True, capture_output=True)
            self.assertEqual((out / 'gipformer.txt').read_text(), 'SJC\n')
            self.assertEqual((out / 'zipformer.txt').read_text(), '')  # never misses SJC; ZALO is only in test
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['utterances'], 3)


class EvalSpecTests(unittest.TestCase):
    def test_parse_spec(self):
        self.assertEqual(parse_spec('gipformer=models/gip'), ('gipformer', 'models/gip', {}, ''))
        with tempfile.TemporaryDirectory() as tmp:
            words = Path(tmp) / 'gipformer.txt'
            words.write_text('SJC\n')
            kind, directory, options, label = parse_spec(f'zipformer=models/zip@decoding=modified_beam_search,hotwords={words},hotwords_score=2')
            self.assertEqual((kind, directory, options['hotwords_score'], options['max_active_paths']), ('zipformer', 'models/zip', 2.0, 4))
            self.assertEqual(label, 'decoding-modified_beam_search_hotwords-gipformer_hotwords_score-2')
        for bad in ('whisper=models/w', 'gipformer=models/gip@beam_size=4', 'gipformer=models/gip@hotwords=x'):
            with self.assertRaises((ValueError, FileNotFoundError)):
                parse_spec(bad)

    def test_keyword_scores(self):
        said, recalled, extra = keyword_scores('VÀNG SJC SJC'.split(), 'VÀNG SC SJC ZALO'.split())
        self.assertEqual((said, recalled, extra), (2, 1, 2))  # SC for SJC is a miss and a false alarm, as in keyword spotting


@unittest.skipUnless(HAS_TORCH, 'torch not installed')
class LoraScaleTests(unittest.TestCase):
    def test_scale_interpolates_between_base_and_adapter(self):
        import torch
        import torch.nn as nn
        from asr_lora.lora import add_lora, merge_lora, scale_lora

        class FeedforwardModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj, self.out_proj = nn.Linear(6, 8), nn.Linear(8, 6)

            def forward(self, x):
                return self.out_proj(torch.relu(self.in_proj(x)))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = FeedforwardModule()

        torch.manual_seed(0)
        model = Model()
        x = torch.randn(3, 6)
        base = model.encoder(x).detach()
        add_lora(model, r=2, alpha=4.0)
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.data.normal_()
        full = model.encoder(x).detach()
        self.assertEqual(scale_lora(model, 0.0), 2)
        self.assertTrue(torch.allclose(model.encoder(x), base, atol=1e-6))
        scale_lora(model, 0.5)
        half = model.encoder(x).detach()
        self.assertFalse(torch.allclose(half, base) or torch.allclose(half, full))
        merge_lora(model)
        self.assertTrue(torch.allclose(model.encoder(x), half, atol=1e-5))  # the export merges what was scaled
        with self.assertRaises(ValueError):
            scale_lora(model, -1)

    def test_default_lr_follows_icefall_lora(self):
        from train_asr_lora import default_lr, get_parser
        self.assertEqual((default_lr('scaled_adam'), default_lr('adamw'), default_lr('scaled_adam', 0.01)), (0.045, 5e-4, 0.01))
        args = get_parser().parse_args(['--model', 'gipformer', '--model-dir', 'm', '--train-manifest', 't', '--valid-manifest', 'v',
                                        '--exp-dir', 'e', '--lora-joiner'])
        self.assertIsNone(args.base_lr)  # resolved per optimizer in main()
        self.assertTrue(args.lora_joiner)

    def test_joiner_only_targets(self):
        import torch.nn as nn
        from asr_lora.lora import add_lora, lora_targets

        class FeedforwardModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj, self.out_proj = nn.Linear(4, 4), nn.Linear(4, 4)

        class Joiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_proj, self.decoder_proj, self.output_linear = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 9)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.joiner = FeedforwardModule(), Joiner()

        wrapped = add_lora(Model(), 2, 4.0, 0.0, *lora_targets({'targets': 'joiner'}))
        self.assertEqual(sorted(wrapped), ['joiner.decoder_proj', 'joiner.encoder_proj', 'joiner.output_linear'])
        from train_asr_lora import get_parser
        args = get_parser().parse_args(['--model', 'zipformer', '--model-dir', 'm', '--train-manifest', 't', '--valid-manifest', 'v',
                                        '--exp-dir', 'e', '--lora-targets', 'joiner', '--real-repeat', '3', '--tts-speed-perturb',
                                        '--tts-reverb-prob', '0.3'])
        self.assertEqual((args.lora_targets, args.real_repeat, args.tts_reverb_prob, args.fixed_tts_subset), ('joiner', 3, 0.3, False))

    def test_joiner_is_opt_in_and_saved_in_the_config(self):
        import torch.nn as nn
        from asr_lora.lora import LoRALinear, add_lora, lora_targets

        class FeedforwardModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj, self.out_proj = nn.Linear(4, 4), nn.Linear(4, 4)

        class Joiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_proj, self.decoder_proj, self.output_linear = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 9)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.joiner = FeedforwardModule(), Joiner()

        plain = Model()
        add_lora(plain, 2, 4.0, 0.0, *lora_targets({'targets': 'icefall'}))
        self.assertNotIsInstance(plain.joiner.output_linear, LoRALinear)
        joint = Model()
        wrapped = add_lora(joint, 2, 4.0, 0.0, *lora_targets({'targets': 'icefall', 'joiner': True}))
        self.assertEqual(sorted(wrapped), ['encoder.in_proj', 'encoder.out_proj', 'joiner.decoder_proj',
                                           'joiner.encoder_proj', 'joiner.output_linear'])
        self.assertTrue(all(p.requires_grad == ('lora_' in n) for n, p in joint.named_parameters()))


@unittest.skipUnless(HAS_PEFT, 'peft not installed')
class QwenMergeScaleTests(unittest.TestCase):
    def test_scale_adapter(self):
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
        from merge_qwen_lora import scale_adapter

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                return self.proj(x)

        torch.manual_seed(0)
        x = torch.randn(2, 4)
        tiny = Tiny()
        base = tiny(x).detach()
        model = get_peft_model(tiny, LoraConfig(r=2, lora_alpha=4, target_modules=['proj'], init_lora_weights=False))
        full = model(x).detach()
        self.assertEqual(scale_adapter(model, 1.0), 0)
        self.assertEqual(scale_adapter(model, 0.5), 1)
        self.assertTrue(torch.allclose(model(x), base + 0.5 * (full - base), atol=1e-5))
        merged = model.merge_and_unload()
        self.assertTrue(torch.allclose(merged(x), base + 0.5 * (full - base), atol=1e-5))
        dora = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=['proj'], use_dora=True))
        with self.assertRaises(ValueError):
            scale_adapter(dora, 0.5)


@unittest.skipUnless(HAS_LHOTSE, 'lhotse/soundfile not installed')
class EpochMixTests(unittest.TestCase):
    """Few hours of real speech next to much TTS: real share per epoch, fresh TTS subsets, TTS-only augmentation."""

    def setUp(self):
        import numpy as np
        import soundfile
        from asr_lora.data import to_cuts
        self.tmp = tempfile.TemporaryDirectory()
        self.records = []
        for i in range(9):
            path = Path(self.tmp.name) / f'{i}.wav'
            soundfile.write(str(path), (0.1 * np.random.RandomState(i).randn(16000)).astype('float32'), 16000)
            self.records.append({'id': f'{"r" if i < 3 else "t"}{i}', 'audio': str(path), 'text': 'BẬT ĐÈN',
                                 'source': 'train_audio' if i < 3 else 'tts'})
        self.cuts = to_cuts(self.records)

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_share_and_resampling(self):
        from asr_lora.data import epoch_cuts
        cuts, mix = epoch_cuts(self.cuts, self.records, 1, real_repeat=2, max_tts_ratio=0.5, seed=1)
        self.assertEqual((mix['real'], mix['real_copies'], mix['tts'], mix['tts_pool'], mix['real_share_percent']), (3, 3, 3, 6, 66.7))
        ids = [c.id for c in cuts]
        self.assertEqual(len(ids), len(set(ids)))
        copy = next(c for c in cuts if c.id.endswith('#r1'))
        self.assertAlmostEqual(copy.duration, 1 / 0.9, places=2)  # the second copy is the 0.9x speed version
        self.assertEqual(copy.load_audio().shape, (1, 17778))
        tts_ids = lambda epoch, resample=True: sorted(c.id for c in epoch_cuts(self.cuts, self.records, epoch, 2, 0.5, 1, resample)[0]
                                                      if c.id.startswith('t'))
        self.assertGreater(len({tuple(tts_ids(e)) for e in range(6)}), 1)  # a fresh TTS subset per epoch
        self.assertEqual(len({tuple(tts_ids(e, False)) for e in range(6)}), 1)
        self.assertEqual(epoch_cuts(self.cuts, self.records, 0, 1, -1)[1]['tts'], 6)
        self.assertEqual(epoch_cuts(self.cuts, self.records, 0, 1, 0)[1]['tts'], 0)
        with self.assertRaises(ValueError):
            epoch_cuts(self.cuts, self.records, 0, 0)

    def test_tts_only_augmentation(self):
        from asr_lora.data import epoch_cuts
        cuts, mix = epoch_cuts(self.cuts, self.records, 3, 1, -1, seed=2, tts_speed=True, tts_reverb_prob=1.0)
        self.assertEqual(mix['tts_reverberated'], 6)
        tts = [c for c in cuts if c.id.startswith('t')]
        self.assertTrue(all('_rvb' in c.id for c in tts))
        self.assertFalse(any('_rvb' in c.id or '_sp' in c.id for c in cuts if c.id.startswith('r')))
        reverberated = epoch_cuts(self.cuts, self.records, 3, 1, -1, seed=2, tts_reverb_prob=1.0)[0]
        try:
            audio = next(c for c in reverberated if c.id.startswith('t')).load_audio()  # simulated RIR, no RIR corpus needed
        except RuntimeError as error:
            if 'torchaudio' in str(error):
                self.skipTest('this lhotse build needs torchaudio for the transform (icefall machines have it)')
            raise
        self.assertEqual(audio.shape, (1, 16000))


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


class RejectionSamplingTests(unittest.TestCase):
    def test_self_source_keeps_verified_own_trajectories_of_fit_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            train = root / 'train'
            (train / 'zaic_tools').mkdir(parents=True)
            (train / 'zaic_tools' / '__init__.py').write_text(FAKE_DEVKIT)
            gold = [{'name': 'notes.list', 'arguments': {}}, {'name': 'notes.add', 'arguments': {'text': 'mua sữa'}}]
            items = [{'id': sid, 'audio_text': 'Ghi chú mua sữa.', 'context': {}, 'history': [],
                      'label': {'tool_calls': gold, 'final_state': {'notes': {'items': ['mua sữa']}}}} for sid in ('s1', 's2', 's3')]
            (train / 'train.jsonl').write_text('\n'.join(json.dumps(i, ensure_ascii=False) for i in items))
            run = Path(tmp) / 'run'
            for item in items:
                folder = train / 'train_environment' / item['id']
                folder.mkdir(parents=True)
                (folder / 'state.json').write_text(json.dumps({'_meta': {'tools': ['notes.list', 'notes.add', 'sys.ask_missing_info']}}))
                sample = run / 'samples' / item['id']
                sample.mkdir(parents=True)
                agent_input = {'normalized_text': 'Ghi chú mua xữa.', 'context': {}, 'history': []}
                (sample / 'trace.jsonl').write_text(json.dumps({'stage': 'agent', 'input': agent_input}, ensure_ascii=False) + '\n')
            own = [{'name': 'notes.add', 'arguments': {'text': 'mua sữa'}}]  # skips the list call: another valid path
            wrong = [{'name': 'notes.add', 'arguments': {'text': 'mua xữa'}}]
            results = [{'id': 's1', 'state_match': True, 'predicted_tool_calls': own},
                       {'id': 's2', 'state_match': True, 'predicted_tool_calls': own},     # test split: stays gold-only
                       {'id': 's3', 'state_match': False, 'predicted_tool_calls': wrong}]  # unsolved: not used
            (run / 'results.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in results))
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps({'s1': {'split': 'fit'}, 's2': {'split': 'test'}, 's3': {'split': 'fit'}}))
            out = Path(tmp) / 'sft'
            env = {**os.environ, 'DATA_ROOT': str(root), 'AGENT_SYSTEM_PROMPT': 'AGENT', 'NORMALIZE_SYSTEM_PROMPT': 'NORM'}
            command = [sys.executable, 'build_agent_sft.py', '--source', f'run:{run}', '--source', f'run:{run}', '--source', f'self:{run}',
                       '--drop-examples', '--manifest', str(manifest), '--out', str(out)]
            subprocess.run(command, env=env, check=True, capture_output=True, text=True, cwd=ROOT)
            summary = json.loads((out / 'summary.json').read_text())
            # s1: gold once (second run source is identical) + its own path; s2: gold once; s3: gold once
            self.assertEqual((summary['kept_trajectories'], summary['skipped_duplicate'], summary['kept_self_trajectories']), (4, 3, 1))
            self.assertEqual(summary['kept_self_differs_from_gold'], 1)
            train_records = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
            own_steps = [json.loads(r['messages'][2]['content']) for r in train_records if r['source'].startswith('self:')]
            self.assertEqual([s.get('name', s['action']) for s in own_steps], ['notes.add', 'finish'])
            test_records = [json.loads(line) for line in (out / 'test.jsonl').read_text().splitlines()]
            self.assertFalse(any(r['source'].startswith('self:') for r in test_records))
            capped = subprocess.run(command + ['--max-trajectories-per-sample', '1', '--out', str(Path(tmp) / 'capped')],
                                    env=env, check=True, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(json.loads((Path(tmp) / 'capped' / 'summary.json').read_text())['kept_trajectories'], 3, capped.stdout)


class SignificanceTests(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(4, 1), 0.375)
        self.assertEqual(mcnemar_exact(10, 2), 0.0386)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)
        base = [{'id': str(i), 'state_match': i < 5, 'exact_calls': False} for i in range(20)]
        new = [{'id': str(i), 'state_match': i < 5 or i >= 15, 'exact_calls': False} for i in range(20)]
        _, summary = compare(base, new, {}, {str(i): 'test' for i in range(20)})
        self.assertEqual(summary['state_outcome_changes'], {'both_correct': 5, 'both_failed': 10, 'improved': 5})
        self.assertEqual(summary['mcnemar_exact_p'], 0.0625)
        self.assertEqual(summary['by_split']['test']['mcnemar_exact_p'], 0.0625)


if __name__ == '__main__':
    unittest.main()
