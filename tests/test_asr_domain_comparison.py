import unittest
from compare_asr_domains import (
    aligned_target, candidate_occurrences, cardinal_value, matches, numeric_difference,
)
from asr_error_audit import align


class ASRDomainComparisonTests(unittest.TestCase):
    def test_excludes_recoverable_surface_variants(self):
        for target, heard in [('siro ho', 'siroho'), ('peter hansen', 'peter han sen'),
                              ('email', 'mail'), ('mạ', 'mẹ'), ('sếp dũng', 'xếp dũng'),
                              ('mười lăm phút', 'mười năm phút')]:
            self.assertTrue(matches(target, heard), (target, heard))
        for target, heard in [('Minh'.lower(), 'mình'), ('vitamin d', 'vitamin t'),
                              ('chưa uống', 'chú uống'), ('sáu mươi phút', 'sáu phút'),
                              ('sjc', 'sgc'), ('sếp dũng', 'xếp dũ')]:
            self.assertFalse(matches(target, heard), (target, heard))

    def test_distinguishes_wrong_value_from_ambiguous_short_number(self):
        self.assertEqual(cardinal_value('một trăm hai mươi ba triệu'.split()), 123000000)
        self.assertEqual(cardinal_value('một trăm hai triệu'.split()), 102000000)
        self.assertIsNone(cardinal_value('bốn năm'.split()))
        self.assertEqual(numeric_difference('bốn mươi phần trăm', 'bốn phần trăm', ''),
                         'different_explicit_numeric_value')
        self.assertEqual(numeric_difference('bốn mươi lăm phút', 'bốn năm phút', ''),
                         'ambiguous_short_number_not_confirmed_value_change')
        self.assertEqual(numeric_difference('hai mươi lăm độ', 'hai lâm độ', ''),
                         'uncertain_spoken_number_review_context')
        self.assertEqual(numeric_difference('một trăm nghìn', 'mấy trăm nghìn', ''),
                         'uncertain_spoken_number_review_context')

    def test_person_mai_and_nested_numbers_are_not_false_targets(self):
        index = {'mai': [('mai', {'term': 'mai', 'category': 'thoi_gian_quan_he', 'tokens': ['mai']})],
                 'ba': [('ba triệu', {'term': 'ba triệu', 'category': 'gia_tri_so', 'tokens': ['ba', 'triệu']})]}
        self.assertEqual(list(candidate_occurrences('cô giáo mai'.split(), index)), [])
        self.assertEqual(list(candidate_occurrences('một trăm hai mươi ba triệu'.split(), index)), [])
        self.assertEqual(len(list(candidate_occurrences('mai gửi ba triệu'.split(), index))), 2)

    def test_a_correct_first_mention_does_not_hide_a_second_error(self):
        ref = 'bật phòng khách rồi tắt phòng khách'
        hyp = 'bật phòng khách rồi tắt phòng ngủ'
        _, h, operations, _ = align(ref, hyp)
        self.assertTrue(matches('phòng khách', aligned_target(h, operations, 1, 3)))
        self.assertFalse(matches('phòng khách', aligned_target(h, operations, 5, 7)))
