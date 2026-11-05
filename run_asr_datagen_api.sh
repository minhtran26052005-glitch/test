#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/venv/bin/python}"
export PYTHONUNBUFFERED=1
if [[ $# == 0 ]]; then
  set -- run
fi
exec "$PYTHON_BIN" generate_asr_data_api.py "$@"
