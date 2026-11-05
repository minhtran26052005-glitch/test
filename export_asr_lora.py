"""Merge a Zipformer LoRA adapter into its checkpoint and export a drop-in model directory.

The output directory replaces GIPFORMER_MODEL_PATH / ZIPFORMER_MODEL_PATH in run.sh as is:
  encoder.onnx, decoder.onnx, joiner.onnx   sherpa-onnx transducer (icefall export-onnx.py, same metadata)
  tokens.txt, bpe.model                     copied from the base model
  model.pt (gipformer) / jit_script.pt (zipformer)  what llm_qwen.check_competition counts parameters from
  merge_report.json                         parameter counts and the numerical checks below
After merging, the architecture and parameter count are exactly the base model's. The script checks that the
merged model reproduces the adapter model, and that the ONNX encoder reproduces PyTorch, before writing the report.

  python export_asr_lora.py --model zipformer --model-dir models/zipformer-150M-6000h \
      --lora outputs/asr_lora/zipformer/lora-best.pt --out-dir models/zipformer-150M-6000h-lora
"""
import argparse
import copy
import importlib.util
import json
import shutil
from pathlib import Path

from asr_lora.lora import DEFAULT_TARGETS, EXTENDED_TARGETS, add_lora, load_lora_state_dict, merge_lora
from asr_lora.models import load_model, setup_icefall


def recipe_module(train, filename, name):
    path = Path(train.__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoder_output(model, features, lengths):
    import torch
    with torch.no_grad():
        out, out_lens = model.forward_encoder(features, lengths)
        return model.joiner.encoder_proj(out), out_lens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['gipformer', 'zipformer'], required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--lora', required=True, help='lora-*.pt from train_asr_lora.py')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--icefall-dir', default='~/.cache/icefall')
    parser.add_argument('--model-config', help='same JSON override used for training, if any')
    parser.add_argument('--tolerance', type=float, default=2e-3)
    parser.add_argument('--num-threads', type=int, default=2, help='CPU threads for ONNX verification')
    args = parser.parse_args()
    import torch
    train = setup_icefall(args.icefall_dir)
    from scaling_converter import convert_scaled_to_non_scaled
    checkpoint = torch.load(args.lora, map_location='cpu', weights_only=False)
    if checkpoint['model_kind'] != args.model:
        raise ValueError(f"Adapter was trained for {checkpoint['model_kind']}, not {args.model}")
    override = json.loads(args.model_config) if args.model_config else None
    model, params, report = load_model(train, args.model, args.model_dir, override)
    base_parameters = sum(p.numel() for p in model.parameters())
    config = checkpoint['lora_config']
    add_lora(model, config['r'], config['alpha'], 0.0, DEFAULT_TARGETS if config['targets'] == 'icefall' else EXTENDED_TARGETS)
    load_lora_state_dict(model, checkpoint['lora'])
    model.eval()

    generator = torch.Generator().manual_seed(0)
    features = torch.randn(2, 300, 80, generator=generator)
    lengths = torch.tensor([300, 220])
    with_adapter, _ = encoder_output(model, features, lengths)
    merged = merge_lora(model)
    merged_out, merged_lens = encoder_output(model, features, lengths)
    if not torch.isfinite(with_adapter).all() or not torch.isfinite(merged_out).all():
        raise RuntimeError('Non-finite output before/after merging the ASR adapter')
    merge_diff = float((with_adapter - merged_out).abs().max())
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != base_parameters:
        raise RuntimeError(f'Merged model has {parameters} parameters, base had {base_parameters}')
    if merge_diff > args.tolerance * max(1.0, float(with_adapter.abs().max())):
        raise RuntimeError(f'Merged model differs from the adapter model by {merge_diff}')

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('tokens.txt', 'bpe.model'):
        shutil.copyfile(Path(args.model_dir) / name, out / name)
    if args.model == 'gipformer':
        torch.save({'model': model.state_dict()}, out / 'model.pt')
    else:
        # Same procedure as icefall zipformer/export.py --jit 1, which produced the original jit_script.pt.
        export = recipe_module(train, 'export.py', 'zipformer_export')
        scripted = copy.deepcopy(model)
        convert_scaled_to_non_scaled(scripted, inplace=True)
        scripted.__class__.forward = torch.jit.ignore(scripted.__class__.forward)
        scripted.encoder = export.EncoderModel(scripted.encoder, scripted.encoder_embed)
        torch.jit.script(scripted).save(str(out / 'jit_script.pt'))

    onnx_export = recipe_module(train, 'export-onnx.py', 'zipformer_export_onnx')
    onnx_model = copy.deepcopy(model)
    convert_scaled_to_non_scaled(onnx_model, inplace=True, is_onnx=True)
    encoder = onnx_export.OnnxEncoder(encoder=onnx_model.encoder, encoder_embed=onnx_model.encoder_embed,
                                      encoder_proj=onnx_model.joiner.encoder_proj)
    decoder = onnx_export.OnnxDecoder(decoder=onnx_model.decoder, decoder_proj=onnx_model.joiner.decoder_proj)
    joiner = onnx_export.OnnxJoiner(output_linear=onnx_model.joiner.output_linear)
    onnx_export.export_encoder_model_onnx(encoder, str(out / 'encoder.onnx'), max_len=-1, dynamic_axes=1,
                                          use_int32_inputs=0, keep_x_lens=1, opset_version=13)
    onnx_export.export_decoder_model_onnx(decoder, str(out / 'decoder.onnx'), use_int32_inputs=0, dynamic_axes=1, opset_version=13)
    onnx_export.export_joiner_model_onnx(joiner, str(out / 'joiner.onnx'), dynamic_axes=1, opset_version=13)

    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.num_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(out / 'encoder.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    onnx_out, onnx_lens = session.run(None, {'x': features.numpy(), 'x_lens': lengths.numpy()})
    if not torch.isfinite(torch.from_numpy(onnx_out)).all():
        raise RuntimeError('Non-finite ONNX encoder output')
    onnx_diff = float((torch.from_numpy(onnx_out) - merged_out).abs().max())
    if onnx_diff > args.tolerance * max(1.0, float(merged_out.abs().max())) or not (torch.from_numpy(onnx_lens) == merged_lens).all():
        raise RuntimeError(f'ONNX encoder differs from PyTorch by {onnx_diff}')
    summary = {'model': args.model, 'base_model_dir': args.model_dir, 'lora': args.lora, 'merged_projections': merged,
               'parameters': parameters, 'merge_max_abs_diff': merge_diff, 'onnx_encoder_max_abs_diff': onnx_diff,
               'adapter_epoch': checkpoint.get('epoch'), 'adapter_valid': checkpoint.get('valid'), 'model_config': report['config']}
    (out / 'merge_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
