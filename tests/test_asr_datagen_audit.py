import unittest

from asr_datagen_audit import semantic_text


class DatagenSemanticTests(unittest.TestCase):
    def test_do_not_prioritize_format_or_dialect_alone(self):
        for reference, prediction in [
            ('45 phút', 'bốn lăm phút'),
            ('HTV7', 'HTV bảy'),
            ('KTX', 'ký túc xá'),
            ('Zalo', 'gia lô'),
            ('uống Omega-3', 'uống Omega ba'),
            ('hông', 'không'),
            ('điều hoà', 'điều hòa'),
        ]:
            with self.subTest(reference=reference):
                self.assertEqual(semantic_text(reference), semantic_text(prediction))

    def test_preserve_meaning_bearing_errors(self):
        for reference, prediction in [
            ('30 phần trăm', 'năm mươi phần trăm'),
            ('thuốc huyết áp', 'thuốc'),
            ('vitamin D', 'vitamin T'),
            ('Omeprazole', 'Omezon'),
            ('Minh', 'mình'),
            ('đừng bật', 'được bật'),
            ('mai', 'nay'),
        ]:
            with self.subTest(reference=reference):
                self.assertNotEqual(semantic_text(reference), semantic_text(prediction))
