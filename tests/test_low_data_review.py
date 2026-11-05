import importlib.util
import json
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
import os
import shutil
import subprocess
from unittest.mock import patch

from asr_lora.data import check_tts_reviewed


class ReviewGateTests(unittest.TestCase):
    def test_similar_names_and_different_drugs_do_not_become_the_written_label(self):
        from prepare_asr_finetune_data import tts_label
        self.assertEqual(tts_label('Nhắn cho Linh.', 'Nhắn cho Lan.'), 'NHẮN CHO LINH')
        self.assertEqual(tts_label('Nhắn cho Lính.', 'Nhắn cho Linh.'), 'NHẮN CHO LÍNH')
        self.assertEqual(tts_label('Uống Panadol.', 'Uống Paracetamol.'), 'UỐNG PANADOL')
        self.assertEqual(tts_label('Búc xe giúp tôi.', 'Book xe giúp tôi.'), 'BOOK XE GIÚP TÔI')
        self.assertEqual(tts_label('Mở kênh ét cộng.', 'Mở kênh S+.'), 'MỞ KÊNH S CỘNG')
        self.assertEqual(tts_label('Ét jay xi trên hai triệu.', 'SJC hai triệu.'), 'SJC TRÊN HAI TRIỆU')

    def _review_fixture(self, folder):
        root = Path(folder); audio = root/'audio.wav'; audio.write_bytes(b'fixture audio')
        record = {'id': 'hard', 'audio': str(audio), 'text': 'UỐNG LORATADIN'}
        path = root/'review.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'label', 'audio_sha256', 'nghe_dung_nhan'])
            writer.writeheader()
            writer.writerow({'id': record['id'], 'label': record['text'],
                             'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(), 'nghe_dung_nhan': 'đúng'})
        return record, path

    def test_listening_can_keep_a_hard_word_even_when_both_asrs_disagree_with_the_label(self):
        from qc_tts_asr import manual_overrides, judge, KEEP
        with tempfile.TemporaryDirectory() as folder:
            record, path = self._review_fixture(folder)
            self.assertNotIn(judge(record['text'], {'gipformer': 'UỐNG LÀ RA', 'zipformer': 'UỐNG LÀ RA'})[0], KEEP)
            self.assertEqual(manual_overrides([record], path), {'hard': True})

    def test_listening_approval_is_not_reused_for_changed_audio_or_label(self):
        from qc_tts_asr import manual_overrides
        with tempfile.TemporaryDirectory() as folder:
            record, path = self._review_fixture(folder)
            with self.assertRaisesRegex(ValueError, 'changed'):
                manual_overrides([{**record, 'text': 'UỐNG THUỐC KHÁC'}], path)
            Path(record['audio']).write_bytes(b'different audio')
            with self.assertRaisesRegex(ValueError, 'changed'):
                manual_overrides([record], path)

    def test_real_first_row_does_not_hide_unchecked_tts_later_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mixed.jsonl'
            path.write_text('\n'.join(json.dumps({'source': source}) for source in ['train_audio', 'tts']))
            with self.assertRaisesRegex(ValueError, 'not checked'):
                check_tts_reviewed([path])

    def test_qc_cache_is_invalidated_when_audio_or_model_changes(self):
        from qc_tts_asr import transcribe_all
        class ASR:
            calls = 0
            def __init__(self, *args): pass
            def transcribe(self, audio):
                ASR.calls += 1
                return Path(audio).read_text()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); model = root/'model'; model.mkdir()
            weights = model/'weights.pt'; weights.write_text('model1')
            audio = root/'a.wav'; audio.write_text('audio1')
            cache = root/'hyps.jsonl'; cache.write_text(json.dumps({'id': 'a', 'hyp': 'stale'})+'\n')
            rows = [{'id': 'a', 'audio': str(audio)}]
            with patch('asr.gipformer.GipFormer', ASR):
                self.assertEqual(transcribe_all(rows, 'gipformer', str(model), cache, 'cpu'), {'a': 'audio1'})
                transcribe_all(rows, 'gipformer', str(model), cache, 'cpu')
                self.assertEqual(ASR.calls, 1)
                audio.write_text('audio2')
                self.assertEqual(transcribe_all(rows, 'gipformer', str(model), cache, 'cpu'), {'a': 'audio2'})
                weights.write_text('changed-model2')
                transcribe_all(rows, 'gipformer', str(model), cache, 'cpu')
                self.assertEqual(ASR.calls, 3)

    def test_joiner_stage_still_runs_when_real_encoder_stage_does_not_improve(self):
        source = Path(__file__).resolve().parents[1]/'run_finetune.sh'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); shutil.copyfile(source, root/'run_finetune.sh')
            tts = root/'outputs/asr_finetune/tts'; tts.mkdir(parents=True)
            (tts/'train_qc.jsonl').write_text('{"source":"tts"}\n')
            fake = root/'python'; log = root/'calls'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$SMOKE_CALL_LOG"\n')
            fake.chmod(0o755)
            env = {**os.environ, 'PYTHON_BIN': str(fake), 'SMOKE_CALL_LOG': str(log)}
            subprocess.run(['bash', str(root/'run_finetune.sh'), 'asr2', 'gipformer'], env=env,
                           check=True, capture_output=True)
            calls = log.read_text().splitlines()
            self.assertEqual([line.split()[0] for line in calls],
                             ['train_asr_lora.py', 'train_asr_lora.py', 'eval_asr_onnx.py'])
            self.assertIn('--lora-targets joiner', calls[1])
            self.assertIn('--tts-ratio-unit seconds', calls[1])


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lhotse', 'soundfile', 'numpy')),
                     'lhotse/soundfile/numpy not installed')
class EpochExposureTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import soundfile as sf
        from asr_lora.data import to_cuts
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.records = []
        # One real 2-second clip and seven TTS fixtures of 1.5 seconds, plus an oversized clip.
        for i, seconds in enumerate([2]+[1.5]*7+[4]):
            path = Path(self.tmp.name)/f'{i}.wav'
            sf.write(path, np.zeros(round(seconds*16000), dtype='float32'), 16000)
            self.records.append({'id': str(i), 'audio': str(path), 'text': 'BẬT ĐÈN',
                                 'source': 'train_audio' if i == 0 else 'tts'})
        self.cuts = to_cuts(self.records)

    def test_duration_cap_survives_speed_augmentation(self):
        from asr_lora.data import epoch_cuts
        cuts, mix = epoch_cuts(self.cuts, self.records, 1, tts_speed=True, ratio_unit='seconds')
        self.assertEqual(mix['tts_over_duration_budget'], 1)
        self.assertLessEqual(mix['tts_seconds'], mix['real_seconds'])
        self.assertGreaterEqual(mix['real_share_seconds_percent'], 50)
        self.assertEqual(len(cuts), 2)

    def test_count_sampling_covers_the_pool_before_leaving_examples_unseen(self):
        from asr_lora.data import epoch_cuts
        seen = set()
        for epoch in range(1, 5):
            cuts, _ = epoch_cuts(self.cuts, self.records, epoch, max_tts_ratio=2)
            current = [cut.recording_id for cut in cuts if cut.recording_id != '0']
            self.assertEqual(len(current), len(set(current)))
            seen.update(current)
        self.assertEqual(seen, {str(i) for i in range(1, 9)})

    def test_duration_sampling_covers_every_fitting_clip_in_one_cycle(self):
        from asr_lora.data import epoch_cuts
        seen = set()
        for epoch in range(1, 8):
            cuts, mix = epoch_cuts(self.cuts, self.records, epoch, ratio_unit='seconds')
            self.assertEqual(mix['duration_cycle_epochs'], 7)
            seen.update(cut.recording_id for cut in cuts if cut.recording_id != '0')
        self.assertEqual(seen, {str(i) for i in range(1, 8)})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch not installed')
class ValidationReviewTests(unittest.TestCase):
    def test_empty_validation_cannot_report_zero_wer(self):
        from contextlib import nullcontext
        from train_asr_lora import validate
        class Model:
            def eval(self): pass
        with self.assertRaisesRegex(RuntimeError, 'empty'):
            validate(Model(), None, None, [], 'cpu', None, None, nullcontext)

    def test_nonfinite_encoder_cannot_be_decoded_as_an_accuracy_measurement(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from train_asr_lora import validate
        class Model:
            def eval(self): pass
            def forward_encoder(self, *args): return torch.full((1, 3, 4), float('nan')), torch.tensor([3])
        batch = {'inputs': torch.zeros(1, 10, 80), 'supervisions': {'num_frames': torch.tensor([10]), 'text': ['BẬT ĐÈN']}}
        with patch('train_asr_lora.compute_loss', return_value=(torch.tensor(0.), 3, {})):
            with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                validate(Model(), None, SimpleNamespace(decode=lambda _: ''), [batch], 'cpu', None, None, nullcontext)


if __name__ == '__main__':
    unittest.main()
