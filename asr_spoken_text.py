"""Symmetric written-to-spoken normalization for local Vietnamese ASR scoring.

The normalizer never sees the other transcript, tool labels, or environment.
Keep numeric-format conversion separate from optional contextual aliases.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from num2words import num2words

VERSION = 'vi-spoken-v1'
WORD = re.compile(r'[^\W_]+', re.UNICODE)
DIGITS = ('không', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín')
# Codes which are normally read as a number rather than a digit sequence.
CARDINAL_PREFIXES = {'ron', 'e', 'vtv', 'htv', 'sctv', 'vov'}


def raw_tokens(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize('NFC', text).casefold())


def digit_sequence(value: str) -> str:
    return ' '.join(DIGITS[int(char)] for char in value if char.isdigit())


@lru_cache(maxsize=4096)
def cardinal(value: str) -> str:
    # Leading zeroes are meaningful for phone numbers, PINs, and identifiers.
    if len(value) > 1 and value.startswith('0'):
        return digit_sequence(value)
    if len(value) > 15:
        return digit_sequence(value)
    return num2words(int(value), lang='vi')


def spoken_text(text: str, *, contextual_aliases: bool = False) -> str:
    """Return a deterministic spoken rendering; never correct lexical mistakes.

    Formats: cardinals, leading-zero/grouped phones, lottery digits after 'vé',
    alphanumeric codes, HH:MM[:SS], d/m/yyyy, VI thousands/decimal separators,
    percent, h, and currency symbols. Other units/letters are preserved.
    Dates/times use one fixed verbalization, not all possible spoken variants.
    """
    text = unicodedata.normalize('NFC', text).casefold()

    # Protect composite expressions by replacing them with words first.
    def phone(match: re.Match[str]) -> str:
        value = ''.join(char for char in match.group() if char.isdigit())
        return digit_sequence(value) if 9 <= len(value) <= 12 else match.group()

    text = re.sub(r'(?<!\w)0\d{2,3}(?:[ .-]\d{2,4}){1,4}(?!\w)', phone, text)
    text = re.sub(r'(?<!\w)0\d{8,11}(?!\w)', phone, text)

    def date(match: re.Match[str]) -> str:
        day, month, year = match.groups()
        if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
            return match.group()
        return f'{cardinal(str(int(day)))} tháng {cardinal(str(int(month)))} năm {cardinal(year)}'

    text = re.sub(r'(?<!\w)(\d{1,2})/(\d{1,2})/(\d{4})(?!\w)', date, text)

    def clock(match: re.Match[str]) -> str:
        hours, minutes, seconds = match.groups()
        if int(hours) > 23 or int(minutes) > 59 or (seconds and int(seconds) > 59):
            return match.group()
        result = f'{cardinal(str(int(hours)))} giờ'
        if int(minutes):
            result += f' {cardinal(str(int(minutes)))} phút'
        if seconds and int(seconds):
            result += f' {cardinal(str(int(seconds)))} giây'
        return result

    text = re.sub(r'(?<!\w)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\w)', clock, text)

    # A ticket's serial number is not a count or monetary amount.
    def ticket(match: re.Match[str]) -> str:
        return match.group(1) + digit_sequence(match.group(2))

    text = re.sub(r'(\bvé(?:\s+(?:anh|chị|em|tôi|mình|số|là|của)){0,4}\s+)(\d{4,8})(?!\w)', ticket, text)

    def mixed(match: re.Match[str]) -> str:
        value = match.group()
        if not (any(c.isdigit() for c in value) and any(c.isalpha() for c in value)):
            return value
        pieces = re.findall(r'\d+|[^\W\d_]+', value)
        if len(pieces) == 2 and pieces[0].isdigit() and pieces[1] in {'h', 'đ', 'vnd', 'usd', 'kg', 'km', 'mg', 'cm', 'mm', 'ml'}:
            unit = {'h': 'giờ', 'đ': 'đồng', 'vnd': 'đồng'}.get(pieces[1], pieces[1])
            return cardinal(pieces[0]) + ' ' + unit
        result = []
        for index, piece in enumerate(pieces):
            if piece.isdigit():
                prior = pieces[index - 1] if index else ''
                result.append(cardinal(piece) if prior in CARDINAL_PREFIXES or len(piece) == 1 else digit_sequence(piece))
            else:
                result.append(piece)
        return ' '.join(result)

    text = re.sub(r'(?<!\w)\d{1,3}(?:\.\d{3})+(?!\w)', lambda m: cardinal(m.group().replace('.', '')), text)
    text = re.sub(r'(?<!\w)(\d+),(\d+)(?!\w)', lambda m: cardinal(m.group(1)) + ' phẩy ' + digit_sequence(m.group(2)), text)
    text = re.sub(r'\b[^\W_]+(?:-[^\W_]+)*\b', mixed, text)
    text = text.replace('%', ' phần trăm ').replace('₫', ' đồng ')
    text = re.sub(r'(?<=\d)đ\b', ' đồng ', text)
    text = re.sub(r'\d+', lambda m: cardinal(m.group()), text)

    result = ' '.join(raw_tokens(text))
    if contextual_aliases:
        # Fixed rules apply equally on both sides; never use a best-matching
        # reference. Preserve genuine wrong values such as 30 -> 50.
        result = re.sub(r'\bmươi tư\b', 'mươi bốn', result)
        result = re.sub(r'\bmươi năm\b', 'mươi lăm', result)
        result = re.sub(r'\bmươi một\b', 'mươi mốt', result)
        result = re.sub(r'\b(ron|rôn) chín (lăm|năm)\b', 'ron chín mươi lăm', result)
        # Temperatures in the 20s: 'hai sáu độ' is a common shorthand for 26.
        unit_digits = 'một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín'
        def temperature(match: re.Match[str]) -> str:
            word = {'tư': 'bốn', 'năm': 'lăm', 'một': 'mốt'}.get(match.group(1), match.group(1))
            return f'hai mươi {word} độ'
        result = re.sub(rf'\bhai ({unit_digits}) độ\b', temperature, result)
        result = re.sub(r'\bhai (tư|bốn) giờ\b', 'hai mươi bốn giờ', result)
    return result


def spoken_tokens(text: str, *, contextual_aliases: bool = False) -> list[str]:
    return spoken_text(text, contextual_aliases=contextual_aliases).split()
