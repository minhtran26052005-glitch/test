#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export ASR_MODE=dual ASR_PRIMARY=zipformer ASR_SECONDARY=gipformer ASR_COMBINE=llm_review
export ASR_DEVICE=cpu ASR_CACHE_ONLY=1 ASR_REVIEW_HISTORY=1
export ZIPFORMER_MODEL_PATH="$PWD/models/zipformer-150M-6000h"
export GIPFORMER_MODEL_PATH="$PWD/models/gipformer1.5-68M-rnnt"
export OUTPUT_DIR="$PWD/outputs/dual_asr_qwen"
export ASR_CACHE_DIR="$OUTPUT_DIR/asr_cache"
export ENV_MODE=official DATA_SPLIT=train DEBUG=0 SAVE_INTERMEDIATE=1 RESUME=1
export LLM_DEVICE_MAP=cuda:0 LLM_GPU_MAX_MEMORY='' PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
trap 'status=$?; date -Is > "$OUTPUT_DIR/ended_at.txt"; echo "$status" > "$OUTPUT_DIR/exit_code.txt"' EXIT
date -Is > "$OUTPUT_DIR/started_at.txt"
/home/pc/venv/bin/python prepare_dual_asr.py
./run.sh all --benchmark 1545 --seed 42 --resume
/home/pc/venv/bin/python compare_dual_asr.py
