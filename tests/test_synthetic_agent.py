import json
import unittest
from collections import Counter
from pathlib import Path

from build_agent_sft import agent_records, failed
from build_synthetic_agent import build, check, numbers, references
from audit_finetune_data import gold_labels
from env_adapter import MockEnvironment

SPLITS = 'outputs/timestamp_word_fusion/split_manifest.json'


class NumberTests(unittest.TestCase):
    def test_colloquial_values(self):
        cases = {'hai lăm ngàn rưỡi': 25500, 'hai triệu tám': 2_800_000, 'nghìn rưỡi bước': 1500, 'bốn lăm phút': 45,
                 'bốn mươi lăm phút': 45, '127 triệu': 127_000_000, 'chín giờ rưỡi': 9.5, 'năm chục nghìn': 50_000,
                 'chục hột gà': 10, 'số một tám năm sáu không không': 185600}
        for text, value in cases.items():
            self.assertEqual(numbers(text), Counter({value: 1}), text)

    def test_words_that_are_not_numbers(self):
        self.assertEqual(numbers('không mưa thì thôi'), Counter())
        self.assertEqual(numbers('xem tỷ giá ơ-rô'), Counter())
        self.assertEqual(numbers('thứ bảy, ba rưỡi chiều'), Counter({7: 1, 3.5: 1}))


class CheckTests(unittest.TestCase):
    source = 'Chuyển cho chủ nhà Phạm Hùng hai triệu tám, đón sớm 45 phút nhé.'
    calls = [{'name': 'banking.transfer', 'arguments': {'beneficiary': 'Phạm Hùng', 'amount': 2800000}}]

    def test_equivalent_paraphrase_passes(self):
        self.assertTrue(check('Em chuyển hai triệu tám cho chủ nhà Phạm Hùng, đón sớm bốn lăm phút nghen.', self.source, self.calls)['check_ok'])

    def test_changed_name_or_number_is_flagged(self):
        renamed = check('Chuyển cho chủ nhà Phạm Hưng hai triệu tám, đón sớm 45 phút nhé.', self.source, self.calls)
        self.assertFalse(renamed['check_ok'])
        self.assertIn('hùng', renamed['missing_anchor_tokens'])
        changed = check('Chuyển cho chủ nhà Phạm Hùng hai triệu, đón sớm 45 phút nhé.', self.source, self.calls)
        self.assertFalse(changed['check_ok'])
        dropped = check('Chuyển cho chủ nhà Phạm Hùng hai triệu tám nhé.', self.source, self.calls)
        self.assertEqual(dropped['number_mismatch'], ['45'])

    def test_article_mot_and_tone_placement_tolerated(self):
        calls = [{'name': 'calendar.cancel_event', 'arguments': {'event_id': 'e1', 'reason': 'hủy'}}]
        self.assertTrue(check('Em huỷ một cái lịch đó nhé.', 'Em hủy cái lịch đó nhé.', calls)['check_ok'])


class AuthoredDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        splits = {sid: item['split'] for sid, item in json.loads(Path(SPLITS).read_text()).items()}
        cls.gold = gold_labels()
        cls.records = build(cls.gold, references(), splits)

    def test_counts_split_and_checks(self):
        self.assertEqual(Counter(r['kind'] for r in self.records), {'paraphrase': 600, 'missing': 40, 'cancel': 150})
        self.assertTrue(all(r['split'] == 'fit' for r in self.records))  # never a validation/test seed
        self.assertEqual([r['id'] for r in self.records if not r['check_ok']], [])
        self.assertEqual(len({r['id'] for r in self.records}), len(self.records))

    def test_labels_follow_the_rules(self):
        for r in self.records:
            if r['kind'] == 'paraphrase':
                self.assertEqual(r['tool_calls'], self.gold[r['sample_id']])
                self.assertTrue(r['verify_state'])
            elif r['kind'] == 'missing':
                (ask,) = r['tool_calls']
                self.assertEqual(ask['name'], 'sys.ask_missing_info')
                self.assertEqual(ask['arguments']['params'], [r['removed']])
                self.assertIn(ask['arguments']['tool_name'], {c['name'] for c in self.gold[r['sample_id']]})
            else:
                self.assertEqual(r['tool_calls'], [])
                self.assertTrue(r['text'].startswith(r['source_text'].rstrip()))
            self.assertNotEqual(r['text'], r['source_text'])


class SyntheticReplayTests(unittest.TestCase):
    def test_own_label_and_tool_errors(self):
        item = {'id': 'case', 'context': {}, 'history': [], 'label': {'tool_calls': [{'name': 'smart_home.get_status', 'arguments': {}}]}}
        good = {'text': 'đặt xe', 'context': {}, 'history': [], 'asr_transcripts': None, 'examples': None, 'verify_state': False,
                'tool_calls': [{'name': 'ride.book', 'arguments': {'pickup': 'a', 'destination': 'b'}}]}
        broken = dict(good, tool_calls=[{'name': 'smart_home.climate_set', 'arguments': {'device_id': 'nope', 'temperature': 20}}])
        records, outcome = agent_records(item, [('synthetic:good', good), ('synthetic:broken', broken)],
                                         lambda item: MockEnvironment({}), 'not-checked', 'SYSTEM')
        self.assertEqual(outcome['kept_trajectories'], 1)
        self.assertEqual(outcome['skipped_tool_error'], 1)
        self.assertEqual([json.loads(r['messages'][2]['content']).get('name', 'finish') for r in records], ['ride.book', 'finish'])
        self.assertTrue(failed([{'observation': {'ok': False}}]))
        self.assertFalse(failed([{'observation': [{'device_id': 'x'}]}]))


if __name__ == '__main__':
    unittest.main()
