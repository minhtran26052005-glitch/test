#!/usr/bin/env python3
"""Download the pinned full result snapshot, verify SHA-256, optionally restore it."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extract', action='store_true', help='Restore outputs/ into this checkout')
    p.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((Path(__file__).resolve().parents[1]/'SHARE_SNAPSHOT.json').read_text())
    cache = root/'.downloads'
    cache.mkdir(parents=True,exist_ok=True)
    base = 'https://github.com/minhtran26052005-glitch/test/releases/download/snapshot-2026-11-05/'
    for asset in manifest['release_assets']:
        target = cache/asset['name']
        valid = target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==asset['sha256']
        if not valid:
            tmp=target.with_suffix(target.suffix+'.part')
            print('Downloading',asset['name'],flush=True)
            urllib.request.urlretrieve(base+asset['name'],tmp)
            if hashlib.sha256(tmp.read_bytes()).hexdigest()!=asset['sha256']:
                raise SystemExit('SHA-256 mismatch: '+asset['name'])
            tmp.replace(target)
        print('Verified',asset['name'],flush=True)
    if args.extract:
        if not hasattr(tarfile,'data_filter'):
            raise SystemExit('Use Python 3.12+ for safe archive extraction')
        archive=cache/'lifemate-results-2026-11-05.tar.gz'
        with tarfile.open(archive,'r:gz') as tar:
            tar.extractall(root,filter='data')
        print('Restored',root/'outputs')


if __name__=='__main__':
    main()
