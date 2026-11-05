"""Follow-ups to the 2026-11-05 review: TTS labels follow the audio, evidence order follows the runtime."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_agent_sft import asr_order, ordered, run_base, saved_hypotheses
from prepare_asr_finetune_data import tts_label

try:
    import torch  # noqa: F401  (train_asr_lora imports asr_lora.lora, which needs torch)
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


class TtsLabelTests(unittest.TestCase):
    def test_label_is_what_the_tts_says_with_latin_spelling(self):
        cases = [
            ('Nhắn ét em ét cho Lan.', 'Nhắn SMS cho Lan.', 'NHẮN SMS CHO LAN'),
            ('Giá xăng ròn chín lăm.', 'Giá xăng RON 95.', 'GIÁ XĂNG RON CHÍN LĂM'),            # no extra MƯƠI
            ('Báo thức sáu giờ rưỡi nhé.', 'Báo thức 6 rưỡi nhé.', 'BÁO THỨC SÁU GIỜ RƯỠI NHÉ'),  # GIỜ is said
            ('Mở kênh S cộng Sport.', 'Mở kênh S+ Sport.', 'MỞ KÊNH S CỘNG SPORT'),              # CỘNG is said
            ('Tắt bếp lúc bảy giờ bốn mươi lăm.', 'Tắt bếp lúc 7h45.', 'TẮT BẾP LÚC BẢY GIỜ BỐN MƯƠI LĂM'),
            ('Vàng ét giê xê bữa nay.', 'Vàng SJC hôm nay.', 'VÀNG SJC BỮA NAY'),                # mixed span
            ('Vàng sờ jê xê.', 'Vàng SJC.', 'VÀNG SJC'),                                         # English letter names
            ('Nhắn da lô cho mẹ.', 'Nhắn Zalo cho mẹ.', 'NHẮN ZALO CHO MẸ'),
        ]
        for spoken, written, expected in cases:
            self.assertEqual(tts_label(spoken, written), expected, spoken)

    def test_a_different_word_is_not_relabelled(self):
        self.assertEqual(tts_label('Nhắn tin cho Lan nhé.', 'Nhắn SMS cho Lan nhé.'), 'NHẮN TIN CHO LAN NHÉ')
        self.assertEqual(tts_label('Tám giờ nhé.', '8h nhé.'), 'TÁM GIỜ NHÉ')

    def test_committed_requests_only_differ_from_the_audio_by_latin_spellings(self):
        import difflib
        from prepare_asr_finetune_data import LATIN, target_text
        for path in ('outputs/asr_finetune/tts_requests.jsonl', 'outputs/synthetic_agent/tts_requests.jsonl'):
            for line in (ROOT / path).read_text(encoding='utf-8').splitlines():
                request = json.loads(line)
                said, label = target_text(request['tts_text']).split(), request['text'].split()
                for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=said, b=label, autojunk=False).get_opcodes():
                    if tag != 'equal':
                        self.assertEqual(tag, 'replace', request['id'])
                        self.assertTrue(all(LATIN.fullmatch(w) for w in label[j1:j2]), (request['id'], said[i1:i2], label[j1:j2]))


class EvidenceOrderTests(unittest.TestCase):
    def make_run(self, root, mode='dual'):
        run = Path(root) / 'run'
        folder = run / 'samples' / 's'
        folder.mkdir(parents=True)
        config = {'ASR_MODE': mode, 'ASR_PRIMARY': 'zipformer', 'ASR_SECONDARY': 'gipformer'}
        (run / 'run_metadata.json').write_text(json.dumps({'identity': {'configuration': config}}))
        for name, value in [('asr_gipformer', ' NHẮN MÌNH'), ('asr_zipformer', ' NHẮN MINH'),
                            ('history_0_gipformer', ' GỌI LÀN'), ('history_0_zipformer', ' GỌI LAN'), ('normalize', 'Nhắn Minh.')]:
            (folder / f'{name}.cache.json').write_text(json.dumps({'key': 'k', 'value': value}, ensure_ascii=False))
        agent_input = {'normalized_text': 'Nhắn Minh.', 'history': [{'role': 'user', 'text': 'Gọi Lan.'}], 'context': {}}
        (folder / 'trace.jsonl').write_text(json.dumps({'stage': 'agent', 'input': agent_input}, ensure_ascii=False) + '\n')
        return run, folder

    def test_primary_asr_first_like_main_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, folder = self.make_run(tmp)
            self.assertEqual(asr_order(run), ['zipformer', 'gipformer'])
            self.assertEqual(list(saved_hypotheses(folder, 'asr_', asr_order(run))), ['zipformer', 'gipformer'])
            with patch.dict(os.environ, {'AGENT_ASR_EVIDENCE': '1'}):
                start = run_base(run, {'id': 's'}, True)
            self.assertEqual(list(start['asr_transcripts']), ['zipformer', 'gipformer'])
            self.assertEqual(list(start['history'][0]['asr_alternatives']), ['zipformer', 'gipformer'])
        self.assertEqual(asr_order(Path('/nonexistent')), [])
        self.assertEqual(list(ordered({'b': 1, 'a': 2, 'c': 3}, ['c'])), ['c', 'a', 'b'])

    def test_repair_script_only_reorders(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.make_run(tmp)
            sft = Path(tmp) / 'sft'
            sft.mkdir()
            payload = {'normalized_text': 'x', 'asr_transcripts': {'gipformer': 'A', 'zipformer': 'B'},
                       'history': [{'role': 'user', 'text': 't', 'asr_alternatives': {'gipformer': 'C', 'zipformer': 'D'}}]}
            record = {'id': 'r', 'source': f'run:{run}', 'step': 0, 'messages': [
                {'role': 'system', 'content': 'S'}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
                {'role': 'assistant', 'content': '{"action": "finish"}'}]}
            other = dict(record, id='o', source='reference')
            (sft / 'train.jsonl').write_text(json.dumps(record, ensure_ascii=False) + '\n' + json.dumps(other, ensure_ascii=False) + '\n')
            subprocess.run([sys.executable, str(ROOT / 'scripts' / 'reorder_sft_asr_evidence.py'), '--sft-dir', str(sft), '--run', str(run)],
                           check=True, capture_output=True, cwd=tmp)
            fixed, untouched = [json.loads(line) for line in (sft / 'train.jsonl').read_text().splitlines()]
            new = json.loads(fixed['messages'][1]['content'])
            self.assertEqual(new, payload)  # same values
            self.assertEqual(list(new['asr_transcripts']), ['zipformer', 'gipformer'])
            self.assertEqual(list(new['history'][0]['asr_alternatives']), ['zipformer', 'gipformer'])
            self.assertEqual(untouched, other)


@unittest.skipIf(torch is None, 'torch not installed')
class SkipGuardTests(unittest.TestCase):
    def test_default_limit(self):
        from train_asr_lora import get_parser
        args = get_parser().parse_args(['--model', 'gipformer', '--model-dir', 'm', '--train-manifest', 't', '--valid-manifest', 'v', '--exp-dir', 'e'])
        self.assertEqual(args.max_skip_fraction, 0.05)


if __name__ == '__main__':
    unittest.main()
