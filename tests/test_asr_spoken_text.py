import unittest

from asr_spoken_text import spoken_text


class SpokenTextTests(unittest.TestCase):
    def assertEquivalent(self, written, spoken, **kwargs):
        self.assertEqual(spoken_text(written, **kwargs), spoken_text(spoken, **kwargs))

    def test_digits_and_punctuation(self):
        self.assertEquivalent('Đặt 8. Đèn 30%, điều hòa 26 độ.', 'đặt tám đèn ba mươi phần trăm điều hòa hai mươi sáu độ')

    def test_phone_grouping_and_missing_zero(self):
        spoken = 'không chín tám ba hai một bốn sáu bảy năm năm'
        self.assertEquivalent('0983 214 6755', spoken)
        self.assertEquivalent('09832146755', spoken)
        self.assertNotEqual(spoken_text('09832146755'), spoken_text('không chín tám ba hai một bốn sáu bảy năm'))
        self.assertNotEqual(spoken_text('09889990002'), spoken_text('không chín tám tám chín chín không không hai'))

    def test_cardinal_and_digit_codes(self):
        self.assertEquivalent('VTV3, HTV7, RON95, E5.', 'VTV ba HTV bảy RON chín mươi lăm E năm')
        self.assertEquivalent('RV221 và A4', 'RV hai hai một và A bốn')
        self.assertEquivalent('Omega-3', 'Omega ba')
        self.assertNotEqual(spoken_text('VTV3'), spoken_text('VTV sáu'))
        self.assertNotEqual(spoken_text('RV221'), spoken_text('RV hai một'))
        self.assertNotEqual(spoken_text('SJC'), spoken_text('SC'))

    def test_ticket_and_money(self):
        self.assertEquivalent('Vé anh số 412750', 'Vé anh số bốn một hai bảy năm không')
        self.assertEquivalent('1.200.000 đồng', 'một triệu hai trăm nghìn đồng')
        self.assertEquivalent('50đ', 'năm mươi đồng')
        self.assertEquivalent('1,5 kg', 'một phẩy năm kg')
        self.assertEquivalent('2kg', 'hai kg')

    def test_date_and_clock(self):
        self.assertEquivalent('ngày 2/11/2026', 'ngày hai tháng mười một năm hai nghìn lẻ hai mươi sáu')
        self.assertEquivalent('lúc 07:30', 'lúc bảy giờ ba mươi phút')
        self.assertNotEqual(spoken_text('07:30'), spoken_text('08:30'))

    def test_context_aliases_are_explicit_and_value_preserving(self):
        self.assertEquivalent('RON95', 'ron chín lăm', contextual_aliases=True)
        self.assertEquivalent('26 độ', 'hai sáu độ', contextual_aliases=True)
        self.assertEquivalent('24h', 'hai tư giờ', contextual_aliases=True)
        self.assertNotEqual(spoken_text('26 độ'), spoken_text('hai sáu độ'))
        self.assertNotEqual(spoken_text('30%', contextual_aliases=True), spoken_text('năm mươi phần trăm', contextual_aliases=True))

    def test_no_semantic_correction(self):
        self.assertNotEqual(spoken_text('đừng bật'), spoken_text('được bật'))
        self.assertNotEqual(spoken_text('Minh'), spoken_text('mình'))
        self.assertNotEqual(spoken_text('Omeprazole'), spoken_text('Omezon'))
        self.assertNotEqual(spoken_text('mai'), spoken_text('nay'))
        self.assertNotEqual(spoken_text('chưa trả'), spoken_text('trả'))


if __name__ == '__main__':
    unittest.main()
