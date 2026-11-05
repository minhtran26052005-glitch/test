#!/usr/bin/env bash
# Fine-tuning steps for the three models, run one at a time on the GPU machine that has data/, models/ and icefall.
#   ./run_finetune.sh data        rebuild ASR manifests + normalizer SFT (already in the repo) and the agent SFT (devkit)
#   ./run_finetune.sh tts         synthesize the generated sentences with the seed speakers' voices (VieNeu-TTS)
#   ./run_finetune.sh gipformer   LoRA RNN-T fine-tune, export, test-split WER vs the base model
#   ./run_finetune.sh zipformer   LoRA CR-CTC + CTC + RNN-T fine-tune, export, test-split WER vs the base model
#   ./run_finetune.sh qwen        LoRA SFT of Qwen3.5-4B on agent + normalizer records, merge to bf16
# Afterwards point run.sh at the exported dirs (GIPFORMER_MODEL_PATH / ZIPFORMER_MODEL_PATH / LLM_MODEL_PATH) and
# rerun the agent benchmark with the same AGENT_* flags used to build the SFT data; compare on split=test only.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN=python3; fi
export PYTHON_BIN
ICEFALL_DIR="${ICEFALL_DIR:-$HOME/.cache/icefall}"
GIP_DIR="${GIPFORMER_MODEL_PATH:-models/gipformer1.5-68M-rnnt}"
ZIP_DIR="${ZIPFORMER_MODEL_PATH:-models/zipformer-150M-6000h}"
QWEN_DIR="${LLM_MODEL_PATH:-models/Qwen3.5-4B}"
# Prompt flags the fine-tuned Qwen will run with; the SFT system prompt is taken from run.sh with these values.
export AGENT_ASR_EVIDENCE="${AGENT_ASR_EVIDENCE:-1}" AGENT_TASK_RULES="${AGENT_TASK_RULES:-1}" AGENT_REPEAT_GUARD="${AGENT_REPEAT_GUARD:-1}"
export AGENT_FEWSHOT_K="${AGENT_FEWSHOT_K:-0}"
ASR_TRAIN=(--train-manifest outputs/asr_finetune/train.jsonl)
if [[ -f outputs/asr_finetune/tts/train.jsonl ]]; then ASR_TRAIN+=(--train-manifest outputs/asr_finetune/tts/train.jsonl); fi
if [[ -f outputs/synthetic_agent/tts/train.jsonl ]]; then ASR_TRAIN+=(--train-manifest outputs/synthetic_agent/tts/train.jsonl); fi

need() { [[ -e "$1" ]] || { echo "Thiếu $1: $2" >&2; exit 1; }; }

case "${1:-}" in
  data)
    "$PYTHON_BIN" audit_finetune_data.py
    "$PYTHON_BIN" prepare_asr_finetune_data.py
    "$PYTHON_BIN" build_synthetic_agent.py
    ./run.sh build_agent_sft.py --normalize-csv outputs/zipformer_150m/asr_comparison.csv --normalize-single --out outputs/qwen_sft/normalize
    need data/train+public_test/train/train.jsonl "giải nén devkit vào data/"
    need outputs/dual_asr_qwen/benchmark_seed42_n1545/samples "python scripts/fetch_results.py --extract"
    ENV_MODE=official ./run.sh build_agent_sft.py --source reference \
      --source run:outputs/benchmark_seed42_n1545 \
      --source run:outputs/zipformer_150m/benchmark_seed42_n1545 \
      --source run:outputs/dual_asr_qwen/benchmark_seed42_n1545 \
      --source generated --source synthetic --drop-examples --out outputs/agent_sft
    ;;
  tts)
    "$PYTHON_BIN" synthesize_tts.py --requests outputs/asr_finetune/tts_requests.jsonl --out outputs/asr_finetune/tts
    "$PYTHON_BIN" synthesize_tts.py --requests outputs/synthetic_agent/tts_requests.jsonl --out outputs/synthetic_agent/tts
    ;;
  gipformer)
    need "$GIP_DIR/model.pt" "huggingface-cli download g-group-ai-lab/gipformer1.5-68M-rnnt model.pt bpe.model --local-dir $GIP_DIR"
    need "$GIP_DIR/bpe.model" "huggingface-cli download g-group-ai-lab/gipformer1.5-68M-rnnt bpe.model --local-dir $GIP_DIR"
    "$PYTHON_BIN" train_asr_lora.py --model gipformer --model-dir "$GIP_DIR" --icefall-dir "$ICEFALL_DIR" "${ASR_TRAIN[@]}" \
      --valid-manifest outputs/asr_finetune/validation.jsonl --exp-dir outputs/asr_lora/gipformer
    if [[ ! -f outputs/asr_lora/gipformer/lora-best.pt ]]; then
      echo 'No improved GipFormer adapter; keeping the base model.'; exit 0
    fi
    "$PYTHON_BIN" export_asr_lora.py --model gipformer --model-dir "$GIP_DIR" --icefall-dir "$ICEFALL_DIR" \
      --lora outputs/asr_lora/gipformer/lora-best.pt --out-dir "${GIP_DIR}-lora"
    "$PYTHON_BIN" eval_asr_onnx.py --manifest outputs/asr_finetune/test.jsonl --model "gipformer=$GIP_DIR" --model "gipformer=${GIP_DIR}-lora" \
      --out outputs/asr_lora/gipformer/test_eval
    ;;
  zipformer)
    need "$ZIP_DIR/jit_script.pt" "tải hynt/ZipFormer-150M-CR-CTC-RNNT-6000h (jit_script.pt, bpe.model)"
    "$PYTHON_BIN" train_asr_lora.py --model zipformer --model-dir "$ZIP_DIR" --icefall-dir "$ICEFALL_DIR" "${ASR_TRAIN[@]}" \
      --valid-manifest outputs/asr_finetune/validation.jsonl --exp-dir outputs/asr_lora/zipformer --max-duration 200
    if [[ ! -f outputs/asr_lora/zipformer/lora-best.pt ]]; then
      echo 'No improved ZipFormer adapter; keeping the base model.'; exit 0
    fi
    "$PYTHON_BIN" export_asr_lora.py --model zipformer --model-dir "$ZIP_DIR" --icefall-dir "$ICEFALL_DIR" \
      --lora outputs/asr_lora/zipformer/lora-best.pt --out-dir "${ZIP_DIR}-lora"
    "$PYTHON_BIN" eval_asr_onnx.py --manifest outputs/asr_finetune/test.jsonl --model "zipformer=$ZIP_DIR" --model "zipformer=${ZIP_DIR}-lora" \
      --out outputs/asr_lora/zipformer/test_eval
    ;;
  qwen)
    need outputs/agent_sft/train.jsonl "./run_finetune.sh data"
    "$PYTHON_BIN" train_qwen_lora.py --model-path "$QWEN_DIR" \
      --train outputs/agent_sft/train.jsonl --train outputs/qwen_sft/normalize/train.jsonl \
      --valid outputs/agent_sft/validation.jsonl --valid outputs/qwen_sft/normalize/validation.jsonl \
      --task-weights normalize=0.5 --out-dir outputs/qwen_lora ${QWEN_ARGS:-}
    "$PYTHON_BIN" merge_qwen_lora.py --base "$QWEN_DIR" --adapter outputs/qwen_lora/adapter --out "${QWEN_DIR}-lora"
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
