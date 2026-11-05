"""Manifests (prepare_asr_finetune_data.py / synthesize_tts.py) -> lhotse dataloaders, icefall-style.

Features are 80-dim kaldi fbank computed on the fly (lhotse FbankConfig defaults: dither 0, snip_edges False,
high_freq -400), the same front end sherpa-onnx uses at inference.
"""
import json
import math
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


REPEAT_SPEEDS = (0.9, 1.1, 0.95, 1.05)  # speed of the 2nd, 3rd, ... copy of a real utterance in one epoch
TTS_SPEEDS = (0.9, 1.0, 1.1)


def epoch_cuts(cuts, records, epoch, real_repeat=1, max_tts_ratio=1.0, seed=42, resample_tts=True,
               tts_speed=False, tts_reverb_prob=0.0, ratio_unit='utterances'):
    """Training cuts of one epoch when real speech is a few hours and TTS is most of the data.

    Real: every utterance `real_repeat` times; copies are speed-perturbed (0.9, 1.1, 0.95, 1.05, ...) so they
    are not identical. TTS: at most max_tts_ratio x (real cuts of the epoch), all of it when < 0; with
    resample_tts cycles through a seeded permutation of the pool. With ratio_unit='seconds', disjoint groups
    fit a duration budget; slowest allowed speed is budgeted before augmentation. Overlong cuts are reported.
    The duration budget controls audio exposure, not the fraction of gradient contributed by TTS.
    tts_speed / tts_reverb_prob add a random speed (0.9 / 1.0 / 1.1) and simulated room
    reverberation to TTS only, so cloned voices do not all carry one clean synthetic channel.
    Returns (CutSet, mix counts)."""
    from lhotse import CutSet
    if real_repeat < 1:
        raise ValueError('--real-repeat must be >= 1')
    if not math.isfinite(max_tts_ratio) or not 0 <= tts_reverb_prob <= 1:
        raise ValueError('TTS ratio must be finite and reverb probability must be in [0, 1]')
    if ratio_unit not in {'utterances', 'seconds'}:
        raise ValueError('TTS ratio unit must be utterances or seconds')
    source = {record['id']: record.get('source') for record in records}  # to_cuts: recording id = record id
    real = [cut for cut in cuts if source[cut.recording_id] != 'tts']
    tts = [cut for cut in cuts if source[cut.recording_id] == 'tts']
    out = list(real)
    for k in range(1, real_repeat):
        factor = REPEAT_SPEEDS[(k - 1) % len(REPEAT_SPEEDS)]
        out += [cut.perturb_speed(factor).with_id(f'{cut.id}#r{k}') for cut in real]
    real_seconds = sum(cut.duration for cut in out)
    order = list(range(len(tts)))
    random.Random(f'{seed}:tts-pool').shuffle(order)
    epoch_index = max(epoch - 1, 0) if resample_tts else 0
    oversized = 0
    groups = []
    if max_tts_ratio < 0:
        chosen = order
    elif ratio_unit == 'utterances':
        budget = min(len(tts), int(max_tts_ratio * len(out)))
        start = epoch_index * budget % max(len(tts), 1)
        chosen = (order[start:] + order[:start])[:budget]
    else:
        budget_seconds = max_tts_ratio * real_seconds
        group, seconds = [], 0.0
        for index in order:
            bound = tts[index].duration / (min(TTS_SPEEDS) if tts_speed else 1.0)
            if bound > budget_seconds:
                oversized += 1
                continue
            if group and seconds + bound > budget_seconds:
                groups.append(group)
                group, seconds = [], 0.0
            group.append(index)
            seconds += bound
        if group:
            groups.append(group)
        chosen = groups[epoch_index % len(groups)] if groups else []
    augment = random.Random(f'{seed}:augment:{epoch}')
    reverberated = 0
    for index in chosen:
        cut = tts[index]
        factor = augment.choice(TTS_SPEEDS) if tts_speed else 1.0
        if factor != 1.0:
            cut = cut.perturb_speed(factor)
        if augment.random() < tts_reverb_prob:
            cut = cut.reverb_rir(None, room_rng_seed=augment.randrange(2 ** 31), source_rng_seed=augment.randrange(2 ** 31))
            reverberated += 1
        out.append(cut)
    tts_seconds = sum(cut.duration for cut in out[len(out)-len(chosen):]) if chosen else 0.0
    mix = {'real': len(real), 'real_copies': len(out) - len(real) - len(chosen), 'tts': len(chosen), 'tts_pool': len(tts),
           'tts_reverberated': reverberated, 'real_share_percent': round(100 * (len(out) - len(chosen)) / max(len(out), 1), 1),
           'ratio_unit': ratio_unit, 'real_seconds': round(real_seconds, 3), 'tts_seconds': round(tts_seconds, 3),
           'real_share_seconds_percent': round(100 * real_seconds / max(real_seconds + tts_seconds, 1e-9), 1),
           'tts_over_duration_budget': oversized, 'duration_cycle_epochs': len(groups)}
    return CutSet.from_cuts(out), mix


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
