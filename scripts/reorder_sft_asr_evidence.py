"""Put the restored ASR evidence of already built agent SFT records in runtime order (primary, then secondary).

Records built before build_agent_sft.asr_order existed listed hypotheses alphabetically (gipformer before
zipformer), while main.process puts the primary ASR first; the dual run's primary is zipformer. This rewrites
`asr_transcripts` and history `asr_alternatives` of the given run's records in place and refreshes SHA256SUMS.
No devkit is needed: only the key order changes, never a value.

  python scripts/reorder_sft_asr_evidence.py --sft-dir outputs/agent_sft --run outputs/dual_asr_qwen/benchmark_seed42_n1545
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_agent_sft import asr_order, ordered  # noqa: E402


def reorder_record(record, source, order):
    """True if the record's user payload changed."""
    if record.get('source') != source:
        return False
    message = record['messages'][1]
    payload = json.loads(message['content'])
    changed = False
    if isinstance(payload.get('asr_transcripts'), dict):
        new = ordered(payload['asr_transcripts'], order)
        changed |= list(new) != list(payload['asr_transcripts'])
        payload['asr_transcripts'] = new
    for turn in payload.get('history', []):
        if isinstance(turn.get('asr_alternatives'), dict):
            new = ordered(turn['asr_alternatives'], order)
            changed |= list(new) != list(turn['asr_alternatives'])
            turn['asr_alternatives'] = new
    if changed:
        message['content'] = json.dumps(payload, ensure_ascii=False)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sft-dir', default='outputs/agent_sft')
    parser.add_argument('--run', action='append', required=True, help='run directory as passed to --source run:DIR')
    args = parser.parse_args()
    sft = Path(args.sft_dir)
    counts = Counter()
    for path in sorted(sft.glob('*.jsonl')):
        lines = path.read_text(encoding='utf-8').splitlines()
        out = []
        for line in lines:
            record = json.loads(line)
            for run in args.run:
                if reorder_record(record, f'run:{run}', asr_order(run)):
                    counts[path.name] += 1
            out.append(json.dumps(record, ensure_ascii=False))
        path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    sums = sft / 'SHA256SUMS'
    if sums.exists():
        # Entries are relative to the repo root (sha256sum -c runs there); fall back to the file next to SHA256SUMS.
        names = [line.split(None, 1)[1].strip().lstrip('*') for line in sums.read_text().splitlines() if line.strip()]
        files = [Path(name) if Path(name).is_file() else sft / Path(name).name for name in names]
        sums.write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {name}\n' for f, name in zip(files, names)))
    print(json.dumps({'reordered_records': dict(counts), 'orders': {run: asr_order(run) for run in args.run}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
