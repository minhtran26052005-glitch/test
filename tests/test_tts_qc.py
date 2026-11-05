import json
import tempfile
import unittest
from pathlib import Path

from asr_lora.data import cap_tts, check_tts_reviewed
from eval_asr_onnx import drift
from prepare_asr_finetune_data import unlabeled_public_test
from qc_tts_asr import judge, profile, real_baseline

LABEL = 'NHẮN SMS CHO LAN'


class JudgeTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(judge(LABEL, {'gipformer': ' NHẮN SMS CHO LAN', 'zipformer': 'NHẮN SMS CHO LAN'})[0], 'asr_already_right')
        self.assertEqual(judge(LABEL, {'gipformer': ' NHẮN ÉT EM ÉT CHO LAN', 'zipformer': 'NHẮN SMS CHO LAN'})[0], 'asr_learnable')
        self.assertEqual(judge(LABEL, {'gipformer': ' MỞ ĐÈN PHÒNG', 'zipformer': ' BẬT TIVI'})[0], 'tts_failed')
        # both listeners hear the same other sentence: the TTS said something else
        self.assertEqual(judge(LABEL, {'gipformer': ' NHẮN TIN CHO LAN', 'zipformer': ' NHẮN TIN CHO LAN'})[0], 'tts_misread')
        self.assertEqual(judge('MỘT HAI BA BỐN NĂM SÁU BẢY TÁM',
                               {'gipformer': 'MỘT HAI BA BỐN NĂM XÁU BẢY TẮM', 'zipformer': 'MỘT HAI BA BỐN NĂM SÁU BẢY'})[0], 'asr_learnable')

    def test_profile_and_real_baseline(self):
        stats = profile([(LABEL, {'gipformer': 'NHẮN TIN CHO LAN', 'zipformer': 'NHẮN TIN CHO LAN'}), (LABEL, {'gipformer': LABEL, 'zipformer': LABEL})])
        self.assertEqual((stats['wer_gipformer'], stats['percent_all_wrong_identically']), (12.5, 50.0))
        real = real_baseline()
        self.assertEqual(real['utterances'], 5036)
        self.assertAlmostEqual(sum(real['categories_percent'].values()), 100, delta=0.1)
        self.assertLess(real['categories_percent']['tts_failed'] + real['categories_percent']['tts_misread'], 10)


class TrainingMixTests(unittest.TestCase):
    def test_tts_never_outnumbers_real_speech(self):
        records = [{'id': f'r{i}', 'source': 'train_audio'} for i in range(10)] + [{'id': f't{i}', 'source': 'tts'} for i in range(25)]
        kept, mix = cap_tts(records, 1.0, seed=1)
        self.assertEqual(mix, {'real': 10, 'tts': 10, 'tts_dropped': 15})
        self.assertEqual(sum(r['source'] == 'tts' for r in kept), 10)
        self.assertEqual(cap_tts(records, 1.0, seed=1)[0], kept)  # deterministic
        self.assertEqual(len(cap_tts(records, -1)[0]), 35)
        self.assertEqual(cap_tts(records, 0.0)[1]['tts'], 0)

    def test_unchecked_tts_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, checked, real = Path(tmp) / 'train.jsonl', Path(tmp) / 'train_qc.jsonl', Path(tmp) / 'real.jsonl'
            raw.write_text(json.dumps({'id': 't', 'source': 'tts'}) + '\n')
            checked.write_text(raw.read_text())
            real.write_text(json.dumps({'id': 'r', 'source': 'train_audio'}) + '\n')
            check_tts_reviewed([checked, real])
            with self.assertRaisesRegex(ValueError, 'qc_tts_asr'):
                check_tts_reviewed([real, raw])
            check_tts_reviewed([raw], allow_unchecked=True)


class DriftTests(unittest.TestCase):
    def test_fixed_broken_and_unlabeled(self):
        records = [{'id': 'a', 'text': 'BẬT ĐÈN'}, {'id': 'b', 'text': 'TẮT QUẠT'}, {'id': 'c'}]
        outputs = {'base': {'a': ['BẬT', 'ĐÈN'], 'b': ['TẮT', 'QUẠ'], 'c': ['XIN', 'CHÀO']},
                   'lora': {'a': ['BẬT', 'ĐEN'], 'b': ['TẮT', 'QUẠT'], 'c': ['XIN', 'CHÀO', 'BẠN']}}
        report = drift(records, outputs, ['base', 'lora'])['lora']
        self.assertEqual((report['fixed_utterances'], report['broken_utterances']), (1, 1))
        self.assertEqual(report['changed_utterances_percent'], 100.0)
        self.assertEqual(report['changed_words_per_100'], 50.0)

    def test_public_test_manifest_has_no_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'public_test'
            folder.mkdir()
            item = {'id': 'p1', 'audio': 'audio/p1/p1.wav', 'history': [{'role': 'user', 'audio': 'audio/p1/p1_u1.wav'}, {'role': 'assistant', 'text': 'Dạ'}]}
            (folder / 'public_test.jsonl').write_text(json.dumps(item) + '\n')
            records = unlabeled_public_test(tmp)
            self.assertEqual([r['id'] for r in records], ['p1:last', 'p1:history:0'])
            self.assertTrue(all('text' not in r for r in records))
            self.assertEqual(unlabeled_public_test(Path(tmp) / 'missing'), [])


if __name__ == '__main__':
    unittest.main()
