#!/usr/bin/env python3
"""Prepare an isolated share snapshot, excluding credentials, weights and source audio.

Copies all saved outputs to a compressed archive and readable result tables to a
Git checkout. Does not mutate live outputs or call any generation/inference API.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from generate_asr_data_api import export, load_keys

ROOT = Path(__file__).resolve().parent
TEXT_SUFFIXES = {'.py','.sh','.json','.jsonl','.csv','.md','.txt','.log','.jinja'}
PATTERNS = [re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
            re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
            re.compile(rb'hf_[A-Za-z0-9]{30,}'),
            re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
DIRECT_DIRS = {'asr_datagen_quality_20261104','asr_datagen_targets','asr_error_audit',
               'asr_spoken_eval','timestamp_fusion_plan'}
DIRECT_TOP = {'benchmark_seed42_n100','benchmark_seed42_n1545','zipformer_150m',
              'dual_asr_qwen','timestamp_word_fusion','asr_datagen_api','asr_timed_v1'}


def excluded(path: Path) -> bool:
    return any(p in {'__pycache__','.cache','.git'} for p in path.parts) or path.name.endswith(('.pyc','.tmp','.lock'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkout', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    args = ap.parse_args()
    checkout, assets = args.checkout.resolve(), args.assets.resolve()
    full = assets/'full'
    if full.exists():
        raise SystemExit('Snapshot already exists; use a fresh --assets directory')
    full.mkdir(parents=True)
    keys = [k.encode() for k in load_keys(ROOT/'API_gg.txt')]
    token = subprocess.run(['gh','auth','token'],capture_output=True,check=False).stdout.strip()
    if token:
        keys.append(token)
    checked, records = 0, []

    def safe_copy(src: Path, dst: Path) -> None:
        nonlocal checked
        data = src.read_bytes()
        if any(k and k in data for k in keys) or any(pattern.search(data) for pattern in PATTERNS):
            raise RuntimeError(f'Credential-like value detected in {src.relative_to(ROOT)}; publishing blocked')
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(data)
        shutil.copymode(src,dst)
        checked += 1

    # All project source and prompts. Never copy credential files or local dependencies.
    source_files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.md','.txt','.json'}
                    and p.name != 'API_gg.txt'] + [ROOT/'.gitignore']
    for dirname in ('asr','tests','examples','prompts'):
        source_files += [p for p in (ROOT/dirname).rglob('*') if p.is_file() and not excluded(p)]
    for src in source_files:
        safe_copy(src,checkout/src.relative_to(ROOT))
    print(f'Copied {len(source_files)} source/prompt files',flush=True)

    # Weight provenance and model-owned prompt templates, without checkpoint tensors.
    for src in (ROOT/'models').rglob('*'):
        if src.is_file() and not src.is_symlink() and not excluded(src) and src.suffix in {'.md','.json','.txt','.jinja'}:
            safe_copy(src,checkout/'model_metadata'/src.relative_to(ROOT/'models'))

    paths = sorted((ROOT/'outputs').rglob('*'))
    symlinks = 0
    for src in paths:
        rel = src.relative_to(ROOT)
        if excluded(rel):
            continue
        dst = full/rel
        if src.is_symlink():
            target = src.resolve()
            if not target.is_relative_to(ROOT/'outputs'):
                raise RuntimeError(f'Output symlink leaves the output tree: {rel}')
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.symlink_to(os.path.relpath(full/target.relative_to(ROOT),dst.parent))
            symlinks += 1
        elif src.is_file():
            safe_copy(src,dst)
    print(f'Copied outputs; scanned {checked} files; normalized {symlinks} relative symlinks',flush=True)

    # Merge exactly the immutable per-seed records copied above, rather than a stale live CSV.
    old_cwd = Path.cwd()
    os.chdir(full)
    try:
        out = Path('outputs/asr_datagen_api')
        seeds = [json.loads(line) for line in (out/'seeds.jsonl').read_text().splitlines()]
        generated = export(out,seeds)
    finally:
        os.chdir(old_cwd)
    failures = [p for p in (full/'outputs/asr_datagen_api/failed').glob('*.json') if '.' not in p.stem]
    snapshot = {'snapshot_utc':datetime.now(timezone.utc).isoformat(),
                'source_workspace':str(ROOT),'generation_finished':generated['completed_seeds']==len(seeds),
                'generation':{**generated,'failed_seeds':len(failures)},
                'expected_generation_variants':len(seeds)*10,
                'included':'All saved outputs, including per-request prompts, raw responses, ASR caches and traces.',
                'excluded':['API_gg.txt and authentication material','model weights','source training/public audio and zip',
                            'local Python environments/dependency folders','temporary lock/cache bytecode files'],
                'output_symlinks_rewritten_relative':symlinks,
                'security_scan':{'files_scanned':checked,'known_credential_leaks':0,'credential_pattern_hits':0}}
    (full/'SHARE_SNAPSHOT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')

    # Readable tables retain the familiar outputs/... paths in Git.
    for src in (full/'outputs').rglob('*'):
        if not src.is_file() or src.is_symlink():
            continue
        rel = src.relative_to(full/'outputs')
        keep = rel.parts[0] in DIRECT_DIRS
        keep |= len(rel.parts)==2 and rel.parts[0] in DIRECT_TOP and src.suffix in {'.json','.jsonl','.csv','.md'}
        keep |= len(rel.parts)>=3 and rel.parts[0]=='timestamp_word_fusion' and rel.parts[1]=='comparison'
        keep |= len(rel.parts)>=3 and rel.parts[0]=='zipformer_150m' and rel.parts[1]=='domain_comparison'
        keep |= len(rel.parts)==3 and rel.parts[1]=='benchmark_seed42_n1545' and src.name in {'summary.json','results.jsonl','run_metadata.json','progress.json'}
        if keep:
            dst = checkout/'outputs'/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)

    for src in sorted(full.rglob('*')):
        if src.is_symlink():
            records.append({'path':src.relative_to(full).as_posix(),'symlink':str(src.readlink())})
        elif src.is_file():
            data = src.read_bytes()
            records.append({'path':src.relative_to(full).as_posix(),'bytes':len(data),
                            'sha256':hashlib.sha256(data).hexdigest()})
    manifest = assets/'results-file-manifest.jsonl.gz'
    with gzip.open(manifest,'wt',encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    archive = assets/'lifemate-results-2026-11-05.tar.gz'
    print(f'Compressing {len(records)} result entries...',flush=True)
    with tarfile.open(archive,'w:gz',compresslevel=6,dereference=False) as tar:
        for name in ('SHARE_SNAPSHOT.json','outputs'):
            tar.add(full/name,arcname=name)
    for path in (archive,manifest):
        snapshot.setdefault('release_assets',[]).append({'name':path.name,'bytes':path.stat().st_size,
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (assets/'SHARE_SNAPSHOT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    (checkout/'SHARE_SNAPSHOT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    (assets/'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in snapshot['release_assets']))
    print(json.dumps(snapshot,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
