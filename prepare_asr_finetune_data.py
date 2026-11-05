"""Build ASR fine-tuning manifests for GipFormer/ZipFormer from data already in the repo; no model is run.

real audio   every train utterance (final turn + user history, 5,036) -> its reference transcript, written the
             way both models emit text: uppercase syllables, numbers/dates/times read out, no punctuation.
tts requests generated variants that passed audit_finetune_data.py for ASR (asr_use=ok) from seeds in the
             fit split only, with the seed's own train recording as the voice to clone (synthesize_tts.py).

Splits follow the split manifest: fit -> train, validation, test. Never train on test: it is the held-out
check for both the ASR WER and the agent benchmark.

Usage: python prepare_asr_finetune_data.py [--out outputs/asr_finetune]
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
from asr_spoken_text import spoken_tokens

UTTERANCES = 'outputs/asr_error_audit/utterances.csv'
VARIANTS = 'outputs/finetune_data_audit/variants.csv'
GENERATED = 'outputs/asr_datagen_api/generated.csv'
MANIFEST = 'outputs/timestamp_word_fusion/split_manifest.json'
SPLIT_FILE = {'fit': 'train', 'validation': 'validation', 'test': 'test'}


def target_text(text):
    """Reference -> model output style (both checkpoints emit uppercase spoken syllables without punctuation)."""
    return ' '.join(spoken_tokens(text)).upper()


LATIN = re.compile(r"[A-Z0-9][A-Z0-9'\-.]*")


# Vietnamese and English readings of letters and digits, as TTS and speakers say them.
LETTER_NAMES = {'a': ['a', 'ây'], 'b': ['bê', 'bi'], 'c': ['xê', 'xi', 'si', 'cờ'], 'd': ['đê', 'di'], 'e': ['e', 'i'],
                'f': ['ép'], 'g': ['giê', 'gi', 'dzi'], 'h': ['hát', 'ếch'], 'i': ['i', 'ai'], 'j': ['giê', 'jê', 'jay', 'giây'],
                'k': ['ca', 'cây'], 'l': ['lờ', 'e lờ', 'eo'], 'm': ['em', 'mờ'], 'n': ['en', 'nờ'], 'o': ['ô', 'âu'],
                'p': ['pê', 'pi'], 'q': ['quy', 'kiu'], 'r': ['rờ', 'e rờ', 'a'], 's': ['ét', 'sờ', 'ét xì'], 't': ['tê', 'ti'],
                'u': ['u', 'diu'], 'v': ['vê', 'vi'], 'w': ['vê kép', 'đáp liu'], 'x': ['ích', 'ếch'], 'y': ['i', 'oai'],
                'z': ['dét', 'zét', 'di'], '0': ['không'], '1': ['một'], '2': ['hai'], '3': ['ba'], '4': ['bốn'], '5': ['năm'],
                '6': ['sáu'], '7': ['bảy'], '8': ['tám'], '9': ['chín']}


def _plain(text):
    text = unicodedata.normalize('NFD', text.lower().replace('đ', 'd'))
    return ''.join(c for c in text if c.isalnum() and not unicodedata.combining(c))


def _pronunciation_fits(read, spelled):
    """The audio span must sound like the Latin word, read as a word ('RÒN' = RON, 'DA LÔ' = ZALO) or letter by
    letter ('ÉT EM ÉT' = SMS, 'A LỜ A RỜ MỜ' = ALARM). A different word ('TIN' for SMS) keeps the audio word."""
    if not 1 <= len(read) <= max(3 * len(spelled) + 1, sum(len(w) for w in spelled)):
        return False
    heard = _plain(' '.join(read))
    word = _plain(' '.join(spelled))
    spellings = [word]
    if len(word) <= 6:  # letter by letter, with Vietnamese or English letter names ('ÉT GIÊ XÊ', 'SỜ JÊ XÊ', 'ÉT JAY XI')
        spellings += [_plain(' '.join(names)) for names in product(*(LETTER_NAMES.get(c, [c]) for c in word))]
    return max(SequenceMatcher(a=heard, b=candidate).ratio() for candidate in spellings) >= 0.5


def _spell(read, spelled):
    """Replace span: Latin spelling where the written side is Latin, the audio words everywhere else."""
    is_latin = [bool(LATIN.fullmatch(w)) for w in spelled]
    if all(is_latin):
        return spelled if _pronunciation_fits(read, spelled) else read
    m = next(i for i, latin in enumerate(is_latin) if not latin)  # Latin prefix ('SJC' in 'SJC HÔM' vs 'ÉT GIÊ XÊ BỮA')
    n = next(i for i, latin in enumerate(reversed(is_latin)) if not latin)  # Latin suffix
    if m and not any(is_latin[m:]):
        tail = len(spelled) - m
        if len(read) > tail and _pronunciation_fits(read[:-tail], spelled[:m]):
            return spelled[:m] + read[-tail:]
    if n and not any(is_latin[:-n]):
        head = len(spelled) - n
        if len(read) > head and _pronunciation_fits(read[head:], spelled[-n:]):
            return read[:head] + spelled[-n:]
    return read


def tts_label(spoken, written):
    """ASR target for synthesized speech: exactly the words the TTS reads (`spoken`), except that a span the
    written text spells in Latin letters (SMS, VTV, RV183, Omeprazole) replaces its Vietnamese pronunciation.

    Labelling from `written` alone adds or drops words the audio does not say ("bốn lăm" -> BỐN MƯƠI LĂM,
    "sáu giờ rưỡi" -> SÁU RƯỠI, "S+ Sport" loses CỘNG); labelling from `spoken` alone loses the acronym spelling.
    """
    said, wrote = target_text(spoken).split(), target_text(written).split()
    label = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=said, b=wrote, autojunk=False).get_opcodes():
        label += _spell(said[i1:i2], wrote[j1:j2]) if tag == 'replace' else said[i1:i2]
    return ' '.join(label)


def relative_audio(path):
    """Source-machine absolute paths -> path relative to the repo root (data/train+public_test/...)."""
    marker = 'data/train+public_test/'
    return path[path.index(marker):] if marker in path else path


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def real_audio_records(utterances, splits):
    for row in utterances:
        index = row['history_index']
        yield {'id': f"{row['id']}:{row['turn']}{':' + index if index else ''}", 'audio': relative_audio(row['audio']),
               'text': target_text(row['reference']), 'reference': row['reference'], 'sample_id': row['id'],
               'turn': row['turn'], 'source': 'train_audio', 'split': SPLIT_FILE.get(splits.get(row['id']), 'unsplit')}


def tts_requests(variants, generated, references, splits):
    generated_by_id = {row['variant_id']: row for row in generated}
    for row in variants:
        if row['asr_use'] != 'ok' or splits.get(row['sample_id']) != 'fit':
            continue
        original = generated_by_id[row['variant_id']]
        audio = relative_audio(original.get('source_audio', ''))
        # The audio reads spoken_text; Latin spellings (SMS / VTV / foreign names) come from the written text.
        yield {'id': row['variant_id'], 'tts_text': row['spoken_text'], 'text': tts_label(row['spoken_text'], original['text']),
               'speaker_audio': audio, 'speaker_text': references.get(audio, ''), 'sample_id': row['sample_id'],
               'turn': row['turn'], 'target_model': row['target_model'], 'domain': row['domain'], 'source': 'tts', 'split': 'train'}


def unlabeled_public_test(data_root):
    """public_test audio (final turn + user history) without text: only used to measure drift after fine-tuning."""
    source = Path(data_root) / 'public_test' / 'public_test.jsonl'
    if not source.is_file():
        return []
    records = []
    for line in source.read_text(encoding='utf-8').splitlines():
        item = json.loads(line) if line.strip() else None
        if not item:
            continue
        audios = [(f"{item['id']}:last", item.get('audio'))]
        audios += [(f"{item['id']}:history:{i}", turn.get('audio')) for i, turn in enumerate(item.get('history', []))
                   if turn.get('role') == 'user' and turn.get('audio')]
        for uid, audio in audios:
            if audio:
                path = Path(audio) if Path(audio).is_absolute() else Path(data_root) / 'public_test' / audio
                records.append({'id': uid, 'audio': str(path), 'sample_id': item['id'], 'source': 'public_test_unlabeled'})
    return records


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='outputs/asr_finetune')
    parser.add_argument('--data-root', default='data/train+public_test', help='for the unlabeled public_test drift manifest')
    args = parser.parse_args()
    out = Path(args.out)
    splits = {sid: item['split'] for sid, item in json.loads(Path(MANIFEST).read_text()).items()}
    utterances = rows(UTTERANCES)
    real = list(real_audio_records(utterances, splits))
    by_split = {}
    for record in real:
        by_split.setdefault(record['split'], []).append(record)
    for split, records in by_split.items():
        write(out / f'{split}.jsonl', records)
    references = {relative_audio(row['audio']): row['reference'] for row in utterances}
    tts = list(tts_requests(rows(VARIANTS), rows(GENERATED), references, splits))
    write(out / 'tts_requests.jsonl', tts)
    drift = unlabeled_public_test(args.data_root)
    if drift:
        write(out / 'public_test_unlabeled.jsonl', drift)
    summary = {'real_audio': dict(Counter(r['split'] for r in real)), 'public_test_unlabeled': len(drift),
               'tts_requests_fit_only': len(tts), 'tts_by_turn': dict(Counter(r['turn'] for r in tts)),
               'tts_without_speaker_audio': sum(not r['speaker_audio'] for r in tts),
               'target_format': 'real audio: reference with numbers read out; TTS: the words the TTS reads, Latin spellings kept from the written text',
               'note': 'Paths are relative to the repo root. tts_requests need synthesize_tts.py before training.'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
