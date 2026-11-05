#!/usr/bin/env bash
# Agent v2 on the dual-ASR setup of run_dual_asr_benchmark.sh (same ASR caches, normalizer, history review), plus:
# ASR evidence, task rules, repeat guard, 3 few-shot examples (same-sample/group excluded). Every option can be
# switched off from the environment for ablations, e.g. AGENT_FEWSHOT_K=0 OUTPUT_DIR=outputs/agent_v2_no_fewshot.
# AGENT_THINKING=1 also works but is several times slower; it raises the decision token budget to 2048.
set -euo pipefail
cd "$(dirname "$0")"
export ASR_MODE=dual ASR_PRIMARY=zipformer ASR_SECONDARY=gipformer ASR_COMBINE=llm_review
export ASR_DEVICE=cpu ASR_CACHE_ONLY=1 ASR_REVIEW_HISTORY=1
export AGENT_ASR_EVIDENCE="${AGENT_ASR_EVIDENCE:-1}" AGENT_TASK_RULES="${AGENT_TASK_RULES:-1}" AGENT_REPEAT_GUARD="${AGENT_REPEAT_GUARD:-1}"
export AGENT_FEWSHOT_K="${AGENT_FEWSHOT_K:-3}" AGENT_FEWSHOT_POOL="${AGENT_FEWSHOT_POOL:-$PWD/outputs/fewshot/pool.jsonl}"
export AGENT_FEWSHOT_EXCLUDE="${AGENT_FEWSHOT_EXCLUDE:-$PWD/outputs/timestamp_word_fusion/split_manifest.json}"
export AGENT_THINKING="${AGENT_THINKING:-0}"
if [[ "$AGENT_THINKING" == 1 ]]; then export AGENT_MAX_NEW_TOKENS="${AGENT_MAX_NEW_TOKENS:-2048}"; fi
export ZIPFORMER_MODEL_PATH="$PWD/models/zipformer-150M-6000h"
export GIPFORMER_MODEL_PATH="$PWD/models/gipformer1.5-68M-rnnt"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/agent_v2}"
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
if [[ "$AGENT_FEWSHOT_K" != 0 && ! -f "$AGENT_FEWSHOT_POOL" ]]; then
  "$PYTHON_BIN" build_fewshot_pool.py --out "$AGENT_FEWSHOT_POOL"
fi
mkdir -p "$OUTPUT_DIR"
trap 'status=$?; date -Is > "$OUTPUT_DIR/ended_at.txt"; echo "$status" > "$OUTPUT_DIR/exit_code.txt"' EXIT
date -Is > "$OUTPUT_DIR/started_at.txt"
./run.sh all --benchmark 1545 --seed 42 --resume
"$PYTHON_BIN" compare_benchmark_runs.py outputs/dual_asr_qwen/benchmark_seed42_n1545 \
  "$OUTPUT_DIR/benchmark_seed42_n1545" "$OUTPUT_DIR/comparison"
