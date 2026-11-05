"""Exercise real ASR checkpoints through two tiny training/export stages without CUDA.

The TTS-labelled rows are REAL AUDIO reused as pipeline fixtures, not synthesized speech.
Epoch adapters are exported even when validation does not improve, solely to test reload.
No production model directory is modified. This is not an accuracy experiment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='outputs/low_data_review_20261105/two_stage')
    parser.add_argument('--icefall-dir', default='.review-deps/icefall')
    parser.add_argument('--models', default='gipformer,zipformer')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit("Set CUDA_VISIBLE_DEVICES='' before running this CPU-only smoke test")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = [json.loads(line) for line in Path('outputs/asr_finetune/train.jsonl').open()][:2]
    valid = [json.loads(line) for line in Path('outputs/asr_finetune/validation.jsonl').open()][:1]
    fixtures = [{**row, 'id': f'smoke_tts_fixture_{i}', 'source': 'tts', 'target_model': 'both'}
                for i, row in enumerate(train)]
    for name, rows in [('train', train), ('validation', valid), ('tts_fixture', fixtures)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    summary = {'cuda_visible_devices': '', 'accuracy_experiment': False,
               'tts_audio_is_real_audio_reused_for_pipeline_testing': True, 'steps': []}
    def run(label, command):
        started = time.monotonic()
        with (out / f'{label}.log').open('w') as log:
            process = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
        summary['steps'].append({'label': label, 'command': command, 'exit_code': process.returncode,
                                 'seconds': round(time.monotonic()-started, 2)})
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary['steps'][-1]), flush=True)
        process.check_returncode()
    model_dirs = {'gipformer': 'models/gipformer1.5-68M-rnnt', 'zipformer': 'models/zipformer-150M-6000h'}
    for kind in args.models.split(','):
        common = ['--model', kind, '--icefall-dir', args.icefall_dir]
        training = ['--train-manifest', str(out/'train.jsonl'), '--valid-manifest', str(out/'validation.jsonl'),
                    '--device', 'cpu', '--precision', 'fp32', '--num-workers', '0', '--num-epochs', '1',
                    '--max-duration', '30', '--lora-r', '4', '--lora-alpha', '8', '--log-interval', '1']
        base = model_dirs[kind]
        stage1, export1 = out/f'{kind}_stage1', out/f'{kind}_export1'
        stage2, export2 = out/f'{kind}_stage2', out/f'{kind}_export2'
        run(f'{kind}_train1', ['train_asr_lora.py', *common, '--model-dir', base, *training,
                              '--max-tts-ratio', '0', '--speed-perturb', '--exp-dir', str(stage1)])
        run(f'{kind}_export1', ['export_asr_lora.py', *common, '--model-dir', base,
                               '--lora', str(stage1/'lora-epoch-1.pt'), '--out-dir', str(export1)])
        run(f'{kind}_train2', ['train_asr_lora.py', *common, '--model-dir', str(export1), *training,
                              '--train-manifest', str(out/'tts_fixture.jsonl'), '--allow-unchecked-tts',
                              '--lora-targets', 'joiner', '--real-repeat', '2', '--max-tts-ratio', '1',
                              '--tts-ratio-unit', 'seconds',
                              '--tts-speed-perturb', '--tts-reverb-prob', '1', '--exp-dir', str(stage2)])
        run(f'{kind}_export2', ['export_asr_lora.py', *common, '--model-dir', str(export1),
                               '--lora', str(stage2/'lora-epoch-1.pt'), '--out-dir', str(export2)])
        run(f'{kind}_decode', ['eval_asr_onnx.py', '--manifest', str(out/'validation.jsonl'),
                              '--model', f'{kind}={base}', '--model', f'{kind}={export1}',
                              '--model', f'{kind}={export2}', '--limit', '1', '--out', str(out/f'{kind}_decode')])


if __name__ == '__main__':
    main()
