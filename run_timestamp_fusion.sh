#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON_BIN" cache_timed_asr.py
if [[ "${FUSION_REFIT:-0}" == 1 || ! -f outputs/timestamp_word_fusion/selector.json ]]; then
  "$PYTHON_BIN" train_word_fusion.py
fi
"$PYTHON_BIN" apply_word_fusion.py
"$PYTHON_BIN" compare_word_fusion.py
