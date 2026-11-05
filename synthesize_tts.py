"""Synthesize tts_requests.jsonl into 16 kHz mono wavs + an ASR training manifest (VieNeu-TTS, Apache-2.0).

Each request uses its seed's own train recording (`speaker_audio`) and transcript (`speaker_text`) as a
voice-cloning reference. Listen to the results before training. Resumable: completed ids are skipped.
Run on a GPU machine:

  pip install vieneu soundfile soxr
  python synthesize_tts.py --requests outputs/asr_finetune/tts_requests.jsonl --out outputs/asr_finetune/tts

Output: OUT/audio/<id>.wav and OUT/train.jsonl in the same format as the real-audio manifests (`text` is the ASR target).
Mix it with real audio, not instead of it (see train_asr_lora.py --train-manifest, which accepts several files).
"""
import argparse
import json
from pathlib import Path

TARGET_RATE = 16000


def synthesize_request(tts, request, speaker, watermark=False):
    reference = request.get('speaker_text', '').strip()
    if not reference:
        raise ValueError(f"{request['id']}: voice cloning requires speaker_text matching speaker_audio")
    return tts.infer(request['tts_text'], ref_audio=str(speaker), ref_text=reference,
                     apply_watermark=watermark)


def to_16k_mono(audio, rate):
    import numpy as np
    audio = np.asarray(audio, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if rate != TARGET_RATE:
        import soxr
        audio = soxr.resample(audio, rate, TARGET_RATE)
    peak = float(abs(audio).max()) if audio.size else 0.0
    return audio / peak * 0.95 if peak > 1.0 else audio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--requests', default='outputs/asr_finetune/tts_requests.jsonl')
    parser.add_argument('--out', default='outputs/asr_finetune/tts')
    parser.add_argument('--audio-root', default='.', help='directory that speaker_audio paths are relative to')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--watermark', action='store_true', help='keep VieNeu inaudible watermark (off by default)')
    parser.add_argument('--min-seconds', type=float, default=0.8)
    parser.add_argument('--max-seconds', type=float, default=20.0)
    args = parser.parse_args()
    import soundfile as sf
    from vieneu import Vieneu
    tts = Vieneu()
    out = Path(args.out)
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    requests = [json.loads(line) for line in Path(args.requests).read_text(encoding='utf-8').splitlines() if line.strip()][:args.limit]
    manifest = out / 'train.jsonl'
    done = {json.loads(line)['id'] for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()} if manifest.exists() else set()
    skipped = 0
    with manifest.open('a', encoding='utf-8') as handle:
        for index, request in enumerate(requests, 1):
            if request['id'] in done:
                continue
            wav = out / 'audio' / f"{request['id']}.wav"
            speaker = Path(args.audio_root) / request['speaker_audio']
            try:
                audio = synthesize_request(tts, request, speaker, args.watermark)
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                print(f"[TTS] lỗi {request['id']}: {exc}", flush=True)
                skipped += 1
                continue
            audio = to_16k_mono(audio, tts.sample_rate)
            seconds = len(audio) / TARGET_RATE
            if not args.min_seconds <= seconds <= args.max_seconds:
                skipped += 1  # truncated or runaway synthesis
                continue
            sf.write(str(wav), audio, TARGET_RATE, subtype='PCM_16')
            handle.write(json.dumps({'id': request['id'], 'audio': str(wav), 'text': request['text'], 'reference': request['tts_text'],
                                     'sample_id': request['sample_id'], 'turn': request['turn'], 'source': 'tts',
                                     'target_model': request['target_model'],
                                     'speaker_audio': request['speaker_audio'], 'split': 'train', 'seconds': round(seconds, 3)},
                                    ensure_ascii=False) + '\n')
            handle.flush()
            if index % 100 == 0:
                print(f'[TTS] {index}/{len(requests)}', flush=True)
    print(json.dumps({'requests': len(requests), 'already_done': len(done), 'skipped': skipped, 'manifest': str(manifest)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
