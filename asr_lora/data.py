"""Manifests (prepare_asr_finetune_data.py / synthesize_tts.py) -> lhotse dataloaders, icefall-style.

Features are 80-dim kaldi fbank computed on the fly (lhotse FbankConfig defaults: dither 0, snip_edges False,
high_freq -400), the same front end sherpa-onnx uses at inference.
"""
import json
import random
from pathlib import Path


def read_manifests(paths, audio_root='.', target_model=None):
    records = []
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                if target_model and record.get('source') == 'tts':
                    if not record.get('target_model'):
                        raise ValueError(f"{record['id']}: TTS manifest lacks target_model; regenerate its metadata from the TTS requests")
                    if record['target_model'] not in {target_model, 'both'}:
                        continue
                audio = Path(record['audio'])
                record['audio'] = str(audio if audio.is_absolute() else Path(audio_root) / audio)
                records.append(record)
    ids = [record['id'] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate utterance ids across manifests')
    return records


def check_tts_reviewed(paths, allow_unchecked=False):
    """TTS audio enters training only after qc_tts_asr.py (its kept manifest is named *_qc.jsonl)."""
    for path in paths:
        if Path(path).name.endswith('_qc.jsonl') or allow_unchecked:
            continue
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip() and json.loads(line).get('source') == 'tts':
                raise ValueError(f'{path} contains TTS audio that was not checked: run qc_tts_asr.py and train on its '
                                 '*_qc.jsonl, or pass --allow-unchecked-tts')
            break


def cap_tts(records, max_ratio, seed=42):
    """Keep real speech the majority: at most max_ratio TTS utterances per real one (deterministic subsample).
    A small in-domain set is otherwise dominated by one synthetic voice/style; max_ratio < 0 disables the cap."""
    real = [r for r in records if r.get('source') != 'tts']
    tts = [r for r in records if r.get('source') == 'tts']
    if max_ratio < 0 or len(tts) <= max_ratio * len(real):
        return records, {'real': len(real), 'tts': len(tts), 'tts_dropped': 0}
    keep = set(id(r) for r in random.Random(seed).sample(tts, int(max_ratio * len(real))))
    kept = [r for r in records if r.get('source') != 'tts' or id(r) in keep]
    return kept, {'real': len(real), 'tts': len(keep), 'tts_dropped': len(tts) - len(keep)}


def to_cuts(records, sample_rate=16000):
    from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
    recordings, supervisions = [], []
    for record in records:
        recording = Recording.from_file(record['audio'], recording_id=record['id'])
        if recording.num_channels != 1:
            raise ValueError(f"{record['audio']} has {recording.num_channels} channels; expected mono")
        recordings.append(recording)
        supervisions.append(SupervisionSegment(id=record['id'], recording_id=record['id'], start=0.0,
                                               duration=recording.duration, channel=0, text=record['text'], language='vi'))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(recordings),
                                 supervisions=SupervisionSet.from_segments(supervisions))
    return cuts.resample(sample_rate)


def icefall_spec_augment():
    """SpecAugment of icefall's asr_datamodule (used in the dataset when CR-CTC does not do it in the model)."""
    from lhotse.dataset import SpecAugment
    return SpecAugment(time_warp_factor=80, num_frame_masks=10, features_mask_size=27, num_feature_masks=2, frames_mask_size=100)


def dataloader(cuts, max_duration, train, spec_augment_in_dataset, num_workers=2, speed_perturb=False, seed=42):
    import torch
    from lhotse import Fbank, FbankConfig
    from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset, SimpleCutSampler
    from lhotse.dataset.input_strategies import OnTheFlyFeatures
    if train and speed_perturb:
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        input_transforms=[icefall_spec_augment()] if train and spec_augment_in_dataset else [],
        return_cuts=True,
    )
    if train:
        sampler = DynamicBucketingSampler(cuts, max_duration=max_duration, shuffle=True, num_buckets=min(30, max(2, len(cuts) // 50)),
                                          drop_last=False, seed=seed)
    else:
        sampler = SimpleCutSampler(cuts, max_duration=max_duration, shuffle=False)
    return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=num_workers,
                                       persistent_workers=False)
