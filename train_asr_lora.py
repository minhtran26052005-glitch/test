"""LoRA fine-tuning of the two Zipformer ASR checkpoints with icefall losses.

  gipformer  GipFormer 1.5 (model.pt): pruned RNN-T, SpecAugment in the dataset, as icefall trains transducers.
  zipformer  hynt ZipFormer-150M (jit_script.pt): its training objective CR-CTC + CTC + pruned RNN-T; the model
             duplicates each batch and masks the two copies independently (icefall AsrModel use_cr_ctc=True).

LoRA placement/initialisation/optimiser follow icefall egs/librispeech/ASR/zipformer_lora (see asr_lora/lora.py):
only lora_A/lora_B train, the base network runs in eval mode, ScaledAdam + Eden with base lr 0.0045.

Needs a CUDA machine with icefall (pinned commit in asr_lora/models.py), k2, lhotse, sentencepiece:

  python train_asr_lora.py --model gipformer --model-dir models/gipformer1.5-68M-rnnt \
      --train-manifest outputs/asr_finetune/train.jsonl --train-manifest outputs/asr_finetune/tts/train.jsonl \
      --valid-manifest outputs/asr_finetune/validation.jsonl --exp-dir outputs/asr_lora/gipformer
  python export_asr_lora.py --model gipformer --model-dir models/gipformer1.5-68M-rnnt \
      --lora outputs/asr_lora/gipformer/lora-best.pt --out-dir models/gipformer1.5-68M-rnnt-lora
"""
import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path

from asr_lora.lora import DEFAULT_TARGETS, EXTENDED_TARGETS, add_lora, lora_state_dict, set_lora_train_mode, trainable_summary
from asr_lora.models import ICEFALL_COMMIT, load_model, setup_icefall

PRESETS = {
    'gipformer': {'use_cr_ctc': False, 'spec_augment_in_dataset': True},
    'zipformer': {'use_cr_ctc': True, 'spec_augment_in_dataset': False},
}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(PRESETS), required=True)
    parser.add_argument('--model-dir', required=True, help='gipformer: model.pt, tokens.txt, bpe.model; zipformer: jit_script.pt, tokens.txt, bpe.model')
    parser.add_argument('--model-config', help='JSON of icefall model arguments overriding the recovered architecture')
    parser.add_argument('--icefall-dir', default='~/.cache/icefall')
    parser.add_argument('--train-manifest', action='append', required=True)
    parser.add_argument('--valid-manifest', action='append', required=True)
    parser.add_argument('--audio-root', default='.')
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=float, default=32.0)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-targets', choices=['icefall', 'extended'], default='icefall',
                        help='icefall: attention in_proj + feed-forward (zipformer_lora); extended: + nonlin attention, conv, attn out_proj')
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--max-duration', type=float, default=300.0, help='seconds of audio per batch')
    parser.add_argument('--optimizer', choices=['scaled_adam', 'adamw'], default='scaled_adam')
    parser.add_argument('--base-lr', type=float, default=0.0045, help='0.0045 for ScaledAdam (icefall); ~5e-4 for AdamW')
    parser.add_argument('--lr-batches', type=float, default=100000.0)
    parser.add_argument('--lr-epochs', type=float, default=100.0)
    parser.add_argument('--warmup-batches', type=float, default=100.0)
    parser.add_argument('--prune-range', type=int, default=5)
    parser.add_argument('--lm-scale', type=float, default=0.25)
    parser.add_argument('--am-scale', type=float, default=0.0)
    parser.add_argument('--simple-loss-scale', type=float, default=0.5)
    parser.add_argument('--ctc-loss-scale', type=float, default=0.2)
    parser.add_argument('--cr-loss-scale', type=float, default=0.2)
    parser.add_argument('--time-mask-ratio', type=float, default=2.5)
    parser.add_argument('--warm-step', type=int, default=0,
                        help='icefall ramps loss scales over warm_step batches from scratch; 0 = start at the final scales (fine-tuning)')
    parser.add_argument('--speed-perturb', action='store_true')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def loss_scales(args, batch_idx):
    """icefall compute_loss schedule; with warm_step=0 the post-warmup values are used from the first batch."""
    ramp = 1.0 if args.warm_step <= 0 else min(batch_idx / args.warm_step, 1.0)
    return {'simple': 1.0 - ramp * (1.0 - args.simple_loss_scale), 'pruned': 0.1 + 0.9 * ramp,
            'ctc': args.ctc_loss_scale, 'cr': ramp * args.cr_loss_scale}


def combine_losses(losses, params, scales):
    """Weighted sum as icefall compute_loss; losses = model(...) output tuple."""
    simple, pruned, ctc, attention, cr = losses
    weighted = []
    if params.use_transducer:
        weighted += [(scales['simple'], simple), (scales['pruned'], pruned)]
    if params.use_ctc:
        weighted.append((scales['ctc'], ctc))
        weighted.append((scales['cr'], cr))  # empty tensor when CR-CTC is off (validation): skipped below
    if params.use_attention_decoder:
        weighted.append((params.attention_decoder_loss_scale, attention))
    terms = [scale * value for scale, value in weighted if value.numel() and scale]
    if not terms:
        raise RuntimeError('No loss term is active for this model configuration')
    return sum(terms[1:], terms[0])


def compute_loss(model, params, sp, batch, device, args, batch_idx, spec_augment, is_training):
    import k2
    import torch
    features = batch['inputs'].to(device)
    supervisions = batch['supervisions']
    feature_lens = supervisions['num_frames'].to(device)
    y = k2.RaggedTensor(sp.encode(supervisions['text'], out_type=int))
    use_spec_aug = params.use_cr_ctc and is_training
    segments = torch.stack([supervisions['sequence_idx'], supervisions['start_frame'], supervisions['num_frames']], dim=1) if use_spec_aug else None
    with torch.set_grad_enabled(is_training):
        losses = model(x=features, x_lens=feature_lens, y=y, prune_range=args.prune_range, am_scale=args.am_scale,
                       lm_scale=args.lm_scale, use_cr_ctc=params.use_cr_ctc and is_training, use_spec_aug=use_spec_aug,
                       spec_augment=spec_augment, supervision_segments=segments, time_warp_factor=80)
        loss = combine_losses(losses, params, loss_scales(args, batch_idx))
    frames = int((feature_lens // params.subsampling_factor).sum())
    parts = {name: float(value.detach()) for name, value in zip(('simple', 'pruned', 'ctc', 'att', 'cr'), losses) if value.numel()}
    return loss, frames, parts


def word_errors(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1], len(ref)


def validate(model, params, sp, loader, device, args, greedy_search_batch, amp):
    import torch
    model.eval()
    loss_sum = frames_sum = errors = words = 0
    with torch.no_grad():
        for batch in loader:
            with amp():
                loss, frames, _ = compute_loss(model, params, sp, batch, device, args, 10**9, None, False)
                features = batch['inputs'].to(device)
                lengths = batch['supervisions']['num_frames'].to(device)
                encoder_out, encoder_out_lens = model.forward_encoder(features, lengths)
            hyps = greedy_search_batch(model=model, encoder_out=encoder_out.float(), encoder_out_lens=encoder_out_lens)
            for reference, tokens in zip(batch['supervisions']['text'], hyps):
                e, n = word_errors(reference, sp.decode(tokens))
                errors, words = errors + e, words + n
            loss_sum, frames_sum = loss_sum + float(loss), frames_sum + frames
    return {'loss_per_frame': loss_sum / max(frames_sum, 1), 'wer_percent': round(100 * errors / max(words, 1), 3), 'words': words}


def main():
    args = get_parser().parse_args()
    import random
    import sentencepiece as spm
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(exp_dir / 'train.log')])
    train = setup_icefall(args.icefall_dir)
    from beam_search import greedy_search_batch  # icefall zipformer recipe
    from optim import Eden, ScaledAdam
    from asr_lora.data import dataloader, read_manifests, to_cuts

    override = json.loads(args.model_config) if args.model_config else None
    model, params, report = load_model(train, args.model, args.model_dir, override)
    logging.info('Loaded %s: %s', args.model, json.dumps(report, ensure_ascii=False))
    preset = PRESETS[args.model]
    params.use_cr_ctc = preset['use_cr_ctc'] and params.use_ctc
    if preset['use_cr_ctc'] and not params.use_ctc:
        logging.warning('Checkpoint has no CTC head: CR-CTC disabled, training with RNN-T only')
    params.update(time_mask_ratio=args.time_mask_ratio, attention_decoder_loss_scale=0.8)
    targets = DEFAULT_TARGETS if args.lora_targets == 'icefall' else EXTENDED_TARGETS
    wrapped = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, targets)
    logging.info('LoRA on %d projections; %s', len(wrapped), trainable_summary(model))
    device = torch.device(args.device)
    model.to(device)
    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(args.model_dir) / 'bpe.model'))

    train_cuts = to_cuts(read_manifests(args.train_manifest, args.audio_root))
    valid_cuts = to_cuts(read_manifests(args.valid_manifest, args.audio_root))
    train_loader = dataloader(train_cuts, args.max_duration, True, preset['spec_augment_in_dataset'] or not params.use_cr_ctc,
                              args.num_workers, args.speed_perturb, args.seed)
    valid_loader = dataloader(valid_cuts, args.max_duration, False, False, args.num_workers)
    spec_augment = train.get_spec_augment(params) if params.use_cr_ctc else None

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if args.optimizer == 'scaled_adam':
        optimizer = ScaledAdam([{'named_params': trainable, 'lr': args.base_lr}], lr=args.base_lr, clipping_scale=2.0)
        scheduler = Eden(optimizer, args.lr_batches, args.lr_epochs, warmup_batches=args.warmup_batches)
    else:
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.base_lr, weight_decay=0.0)
        scheduler = None
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(args.precision)
    def amp():
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == 'fp16')

    config = {'r': args.lora_r, 'alpha': args.lora_alpha, 'dropout': args.lora_dropout, 'targets': args.lora_targets}
    history = [{'epoch': 0, **validate(model, params, sp, valid_loader, device, args, greedy_search_batch, amp)}]
    logging.info('Before fine-tuning: %s', history[-1])
    best = history[-1]['wer_percent']
    batch_idx = 0
    for epoch in range(1, args.num_epochs + 1):
        if scheduler:
            scheduler.step_epoch(epoch - 1)
        train_loader.sampler.set_epoch(epoch)
        set_lora_train_mode(model)
        started, seen_frames, seen_loss = time.time(), 0, 0.0
        for batch in train_loader:
            batch_idx += 1
            set_lora_train_mode(model)
            with amp():
                loss, frames, parts = compute_loss(model, params, sp, batch, device, args, batch_idx, spec_augment, True)
            if not math.isfinite(float(loss)):
                logging.warning('Skipping batch %d with non-finite loss %s', batch_idx, parts)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if scheduler:
                scheduler.step_batch(batch_idx)
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            seen_frames, seen_loss = seen_frames + frames, seen_loss + float(loss)
            if batch_idx % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                logging.info('epoch %d batch %d loss/frame %.4f %s lr %.2e', epoch, batch_idx,
                             seen_loss / max(seen_frames, 1), {k: round(v / max(frames, 1), 4) for k, v in parts.items()}, lr)
        result = {'epoch': epoch, 'train_loss_per_frame': seen_loss / max(seen_frames, 1), 'seconds': round(time.time() - started, 1),
                  **validate(model, params, sp, valid_loader, device, args, greedy_search_batch, amp)}
        history.append(result)
        logging.info('Epoch %d: %s', epoch, result)
        checkpoint = {'lora': lora_state_dict(model), 'lora_config': config, 'model_kind': args.model,
                      'model_dir': str(args.model_dir), 'model_config': report['config'], 'epoch': epoch, 'valid': result,
                      'icefall_commit': ICEFALL_COMMIT}
        torch.save(checkpoint, exp_dir / f'lora-epoch-{epoch}.pt')
        if result['wer_percent'] < best:
            best = result['wer_percent']
            shutil.copyfile(exp_dir / f'lora-epoch-{epoch}.pt', exp_dir / 'lora-best.pt')
        (exp_dir / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2))
    logging.info('Best validation WER %.3f%% (before fine-tuning %.3f%%)', best, history[0]['wer_percent'])
    if not (exp_dir / 'lora-best.pt').exists():
        logging.warning('No epoch beat the original model on validation; keep the original checkpoint')


if __name__ == '__main__':
    main()
