#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export ASR_MODE=single ASR_PRIMARY=zipformer ASR_COMBINE=primary ASR_DEVICE=cpu
export ZIPFORMER_MODEL_PATH="$PWD/models/zipformer-150M-6000h"
export OUTPUT_DIR="$PWD/outputs/zipformer_150m"
export ASR_CACHE_DIR="$OUTPUT_DIR/asr_cache"
export ENV_MODE=official DATA_SPLIT=train DEBUG=0 SAVE_INTERMEDIATE=1 RESUME=1
export LLM_DEVICE_MAP=cuda:0 LLM_GPU_MAX_MEMORY='' PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
trap 'status=$?; date -Is > "$OUTPUT_DIR/ended_at.txt"; echo "$status" > "$OUTPUT_DIR/exit_code.txt"' EXIT
date -Is > "$OUTPUT_DIR/started_at.txt"
./run.sh asr --precache-asr
./run.sh all --benchmark 1545 --seed 42 --resume
/home/pc/venv/bin/python export_debug_csv.py "$OUTPUT_DIR/benchmark_seed42_n1545"
/home/pc/venv/bin/python compare_asr_runs.py "$OUTPUT_DIR"
