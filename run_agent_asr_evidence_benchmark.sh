#!/usr/bin/env bash
# Ablation of run_dual_asr_benchmark.sh: same ASR caches, normalizer and history review;
# the only change is AGENT_ASR_EVIDENCE=1 (raw hypotheses + ASR-evidence rules for the agent).
set -euo pipefail
cd "$(dirname "$0")"
export ASR_MODE=dual ASR_PRIMARY=zipformer ASR_SECONDARY=gipformer ASR_COMBINE=llm_review
export ASR_DEVICE=cpu ASR_CACHE_ONLY=1 ASR_REVIEW_HISTORY=1 AGENT_ASR_EVIDENCE=1
export ZIPFORMER_MODEL_PATH="$PWD/models/zipformer-150M-6000h"
export GIPFORMER_MODEL_PATH="$PWD/models/gipformer1.5-68M-rnnt"
export OUTPUT_DIR="$PWD/outputs/agent_asr_evidence"
# Reuse the 10.072 transcripts already verified by prepare_dual_asr.py; no ASR model is run.
export ASR_CACHE_DIR="$PWD/outputs/dual_asr_qwen/asr_cache"
export ENV_MODE=official DATA_SPLIT=train DEBUG=0 SAVE_INTERMEDIATE=1 RESUME=1
export LLM_DEVICE_MAP=cuda:0 LLM_GPU_MAX_MEMORY='' PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/pc/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN=python3; fi
export PYTHON_BIN
if [[ ! -d "$ASR_CACHE_DIR" ]]; then
  echo "Thiếu $ASR_CACHE_DIR: chạy prepare_dual_asr.py (bước đầu của run_dual_asr_benchmark.sh) trước." >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
trap 'status=$?; date -Is > "$OUTPUT_DIR/ended_at.txt"; echo "$status" > "$OUTPUT_DIR/exit_code.txt"' EXIT
date -Is > "$OUTPUT_DIR/started_at.txt"
./run.sh all --benchmark 1545 --seed 42 --resume
"$PYTHON_BIN" compare_benchmark_runs.py outputs/dual_asr_qwen/benchmark_seed42_n1545 \
  "$OUTPUT_DIR/benchmark_seed42_n1545" "$OUTPUT_DIR/comparison"
