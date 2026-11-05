#!/usr/bin/env bash
# Fine-tuning steps for the three models, run one at a time on the GPU machine that has data/, models/ and icefall.
#   ./run_finetune.sh data        rebuild ASR manifests + normalizer SFT (already in the repo) and the agent SFT (devkit)
#   ./run_finetune.sh tts         synthesize the generated sentences with the seed speakers' voices (VieNeu-TTS)
#   ./run_finetune.sh qc          both ASRs listen to the TTS audio; only what says its label is kept (*_qc.jsonl)
#   ./run_finetune.sh hotwords    no training: domain hotword lists from the fit split, greedy vs beam vs hotwords on validation
#   ./run_finetune.sh gipformer   LoRA RNN-T fine-tune, export, test-split WER vs the base model
#   ./run_finetune.sh zipformer   LoRA CR-CTC + CTC + RNN-T fine-tune, export, test-split WER vs the base model
#   ./run_finetune.sh scales KIND merge 1.0/0.7/0.5 of the trained ASR adapter (LORA_SCALES), compare on validation
#   ./run_finetune.sh asr2 KIND   few hours of real speech + much TTS: encoder LoRA on real audio only, then joiner-only
#                                 LoRA on real (REAL_REPEAT copies) + TTS (speed/reverb), compared on validation
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
# TTS audio is used only after './run_finetune.sh qc' kept it (*_qc.jsonl).
if [[ -s outputs/asr_finetune/tts/train_qc.jsonl ]]; then ASR_TRAIN+=(--train-manifest outputs/asr_finetune/tts/train_qc.jsonl); fi
if [[ -s outputs/synthetic_agent/tts/train_qc.jsonl ]]; then ASR_TRAIN+=(--train-manifest outputs/synthetic_agent/tts/train_qc.jsonl); fi
EVAL_MANIFESTS=(--manifest outputs/asr_finetune/test.jsonl)
if [[ -f outputs/asr_finetune/public_test_unlabeled.jsonl ]]; then EVAL_MANIFESTS+=(--manifest outputs/asr_finetune/public_test_unlabeled.jsonl); fi

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
  qc)
    # Both ASRs listen to every TTS wav; keep only audio that says its label, then listen to listen_sample.csv.
    for manifest in outputs/asr_finetune/tts/train.jsonl outputs/synthetic_agent/tts/train.jsonl; do
      [[ -f "$manifest" ]] || continue
      "$PYTHON_BIN" qc_tts_asr.py --manifest "$manifest" --model "gipformer=$GIP_DIR" --model "zipformer=$ZIP_DIR"
    done
    ;;
  hotwords)
    # Contextual biasing needs no training and leaves the weights alone; check it before any fine-tuning.
    "$PYTHON_BIN" build_hotwords.py --out outputs/asr_hotwords
    for kind in gipformer zipformer; do
      dir="$GIP_DIR"; [[ "$kind" == zipformer ]] && dir="$ZIP_DIR"
      "$PYTHON_BIN" eval_asr_onnx.py --manifest outputs/asr_finetune/validation.jsonl --model "$kind=$dir" \
        --model "$kind=$dir@decoding=modified_beam_search" \
        --model "$kind=$dir@decoding=modified_beam_search,hotwords=outputs/asr_hotwords/$kind.txt,hotwords_score=${HOTWORDS_SCORE:-1.5}" \
        --out "outputs/asr_hotwords/validation_$kind"
    done
    ;;
  scales)
    # WiSE-FT interpolation of a trained adapter; choose the scale on validation, then score only that one on test.
    kind="${2:-}"; [[ "$kind" == gipformer || "$kind" == zipformer ]] || { echo 'usage: ./run_finetune.sh scales gipformer|zipformer' >&2; exit 1; }
    dir="$GIP_DIR"; [[ "$kind" == zipformer ]] && dir="$ZIP_DIR"
    need "outputs/asr_lora/$kind/lora-best.pt" "./run_finetune.sh $kind"
    models=(--model "$kind=$dir")
    for scale in ${LORA_SCALES:-1.0 0.7 0.5}; do
      "$PYTHON_BIN" export_asr_lora.py --model "$kind" --model-dir "$dir" --icefall-dir "$ICEFALL_DIR" \
        --lora "outputs/asr_lora/$kind/lora-best.pt" --scale "$scale" --out-dir "${dir}-lora-s$scale"
      models+=(--model "$kind=${dir}-lora-s$scale")
    done
    "$PYTHON_BIN" eval_asr_onnx.py --manifest outputs/asr_finetune/validation.jsonl "${models[@]}" --out "outputs/asr_lora/$kind/scales_validation"
    ;;
  asr2)
    # TTS teaches words, not acoustics: it never updates the encoder (adapting only prediction/joint networks with TTS
    # is the customization result of Li et al. 2020); see docs/RESEARCH_LOW_DATA_DOMAIN_FINETUNE.md section 3.6.
    kind="${2:-}"; [[ "$kind" == gipformer || "$kind" == zipformer ]] || { echo 'usage: ./run_finetune.sh asr2 gipformer|zipformer' >&2; exit 1; }
    if [[ ${#ASR_TRAIN[@]} -le 2 ]]; then
      echo 'asr2 needs nonempty QC-passed TTS manifests: synthesize audio and run ./run_finetune.sh qc first.' >&2; exit 1
    fi
    dir="$GIP_DIR"; batch=(); [[ "$kind" == zipformer ]] && { dir="$ZIP_DIR"; batch=(--max-duration 200); }
    "$PYTHON_BIN" train_asr_lora.py --model "$kind" --model-dir "$dir" --icefall-dir "$ICEFALL_DIR" ${batch[@]+"${batch[@]}"} \
      --train-manifest outputs/asr_finetune/train.jsonl --max-tts-ratio 0 --speed-perturb \
      --valid-manifest outputs/asr_finetune/validation.jsonl --exp-dir "outputs/asr_lora/${kind}_real"
    stage1_dir="$dir"
    models=(--model "$kind=$dir")
    if [[ -f "outputs/asr_lora/${kind}_real/lora-best.pt" ]]; then
      "$PYTHON_BIN" export_asr_lora.py --model "$kind" --model-dir "$dir" --icefall-dir "$ICEFALL_DIR" \
        --lora "outputs/asr_lora/${kind}_real/lora-best.pt" --out-dir "${dir}-lora-real"
      stage1_dir="${dir}-lora-real"
      models+=(--model "$kind=$stage1_dir")
    else
      echo "Stage 1 did not beat the base $kind; stage 2 will adapt the base model's joiner."
    fi
    "$PYTHON_BIN" train_asr_lora.py --model "$kind" --model-dir "$stage1_dir" --icefall-dir "$ICEFALL_DIR" ${batch[@]+"${batch[@]}"} \
      "${ASR_TRAIN[@]}" --lora-targets joiner --real-repeat "${REAL_REPEAT:-2}" --max-tts-ratio "${MAX_TTS_RATIO:-1}" \
      --tts-ratio-unit "${TTS_RATIO_UNIT:-seconds}" \
      --tts-speed-perturb --tts-reverb-prob "${TTS_REVERB_PROB:-0.3}" \
      --valid-manifest outputs/asr_finetune/validation.jsonl --exp-dir "outputs/asr_lora/${kind}_joiner"
    if [[ -f "outputs/asr_lora/${kind}_joiner/lora-best.pt" ]]; then
      "$PYTHON_BIN" export_asr_lora.py --model "$kind" --model-dir "$stage1_dir" --icefall-dir "$ICEFALL_DIR" \
        --lora "outputs/asr_lora/${kind}_joiner/lora-best.pt" --out-dir "${dir}-lora-2stage"
      models+=(--model "$kind=${dir}-lora-2stage")
    else
      echo "Stage 2 did not beat its starting checkpoint; $stage1_dir remains the candidate."
    fi
    "$PYTHON_BIN" eval_asr_onnx.py --manifest outputs/asr_finetune/validation.jsonl "${models[@]}" --out "outputs/asr_lora/${kind}_2stage_validation"
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
    "$PYTHON_BIN" eval_asr_onnx.py "${EVAL_MANIFESTS[@]}" --model "gipformer=$GIP_DIR" --model "gipformer=${GIP_DIR}-lora" \
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
    "$PYTHON_BIN" eval_asr_onnx.py "${EVAL_MANIFESTS[@]}" --model "zipformer=$ZIP_DIR" --model "zipformer=${ZIP_DIR}-lora" \
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
