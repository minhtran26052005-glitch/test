#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export ASR_MODE="${ASR_MODE:-single}" ASR_PRIMARY="${ASR_PRIMARY:-gipformer}" ASR_SECONDARY="${ASR_SECONDARY:-phowhisper_large}" ASR_COMBINE="${ASR_COMBINE:-primary}"
export GIPFORMER_MODEL_PATH="${GIPFORMER_MODEL_PATH:-models/gipformer1.5-68M-rnnt}" PHOWHISPER_MODEL_PATH="${PHOWHISPER_MODEL_PATH:-vinai/PhoWhisper-large}"
export ZIPFORMER_MODEL_PATH="${ZIPFORMER_MODEL_PATH:-models/zipformer-150M-6000h}" ASR_CACHE_DIR="${ASR_CACHE_DIR:-}"
export ASR_CACHE_ONLY="${ASR_CACHE_ONLY:-0}" ASR_REVIEW_HISTORY="${ASR_REVIEW_HISTORY:-0}"
export FUSION_SELECTOR_PATH="${FUSION_SELECTOR_PATH:-outputs/timestamp_word_fusion/selector.json}" ASR_TIMED_CACHE_DIR="${ASR_TIMED_CACHE_DIR:-outputs/asr_timed_v1}"
export LLM_MODEL_PATH="${LLM_MODEL_PATH:-models/Qwen3.5-4B}" LLM_BACKEND="${LLM_BACKEND:-transformers}" LLM_DTYPE="${LLM_DTYPE:-bfloat16}" LLM_QUANTIZATION="${LLM_QUANTIZATION:-none}" LLM_CONTEXT_TOKENS="${LLM_CONTEXT_TOKENS:-16384}" LLM_TEMPERATURE="${LLM_TEMPERATURE:-0}" NORMALIZE_MAX_NEW_TOKENS="${NORMALIZE_MAX_NEW_TOKENS:-512}" AGENT_MAX_NEW_TOKENS="${AGENT_MAX_NEW_TOKENS:-1024}"
export LLM_GPU_MAX_MEMORY="${LLM_GPU_MAX_MEMORY-7GiB}" LLM_CPU_MAX_MEMORY="${LLM_CPU_MAX_MEMORY:-24GiB}"
export DATA_ROOT="${DATA_ROOT:-data/train+public_test}" DATA_SPLIT="${DATA_SPLIT:-train}"
export RUN_MODE="${RUN_MODE:-all}" DEBUG="${DEBUG:-1}" SAVE_INTERMEDIATE="${SAVE_INTERMEDIATE:-1}" RESUME="${RESUME:-0}" ENV_MODE="${ENV_MODE:-mock}" MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-12}" OUTPUT_DIR="${OUTPUT_DIR:-outputs}" STRICT_COMPETITION="${STRICT_COMPETITION:-1}" LLM_DEVICE_MAP="${LLM_DEVICE_MAP:-auto}" ASR_DEVICE="${ASR_DEVICE:-cpu}"
export NORMALIZE_SYSTEM_PROMPT="$(cat <<'PROMPT'
Bạn là hệ thống hiệu chỉnh và chuẩn hóa transcript tiếng Việt dành cho Voice Agent.
Nhiệm vụ:
- Sửa lỗi ASR rõ ràng khi có đủ bằng chứng.
- Chuẩn hóa số lượng, tiền tệ, thời gian, ngày tháng, nhiệt độ, địa chỉ và đơn vị.
- Chỉnh dấu câu và cách viết cho dễ hiểu.
- Giữ nguyên ý định, tên người, tên thiết bị và nội dung yêu cầu.
- Không tự thêm hành động.
- Không tự suy diễn các thông tin không xuất hiện trong audio.
- Không tự điền thông tin từ user profile vào transcript.
- Nếu không chắc chắn, giữ nguyên nội dung thay vì đoán.
- Nếu được cung cấp nhiều transcript ASR, sử dụng chúng làm bằng chứng, không mặc định transcript nào cũng đúng.
Chỉ trả về JSON có khóa normalized_text. Không giải thích.
PROMPT
)"
export AGENT_SYSTEM_PROMPT="$(cat <<'PROMPT'
Bạn là LifeMate, một AI Agent thực hiện yêu cầu của người dùng bằng các công cụ có sẵn.
Mục tiêu là đạt đúng trạng thái cuối của môi trường.
Quy tắc:
- Chỉ sử dụng tool có trong available_tools.
- Arguments phải tuân thủ đúng JSON Schema.
- Không tự tạo tool name, device_id, account_id, contact_id hoặc tham số không có căn cứ.
- Phân biệt yêu cầu rõ ràng, thiếu thông tin và ngoài phạm vi.
- Nếu cần thông tin từ một tool đọc trạng thái, phải gọi tool đó trước.
- Sau mỗi observation, quyết định bước tiếp theo.
- Có thể thực hiện nhiều hành động liên tiếp.
- Không gọi lại các hành động đã thành công nếu không cần thiết.
- Nếu thiếu tham số bắt buộc và không thể suy ra từ context, sử dụng sys.ask_missing_info theo đúng schema.
- Nếu không có ứng dụng nào đáp ứng yêu cầu, kết thúc với danh sách tool rỗng.
- Khi đã hoàn thành nhiệm vụ thì kết thúc.
- Không sinh lời giải thích ngoài JSON.
Mỗi lượt chỉ trả về {"action":"call","name":"tool_name","arguments":{...}} hoặc {"action":"finish"}.
PROMPT
)"
if [[ -d .deps521 && -d .deps ]]; then
  export PYTHONPATH="$PWD/.deps521:$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
fi
PYTHON_BIN="${PYTHON_BIN:-/home/pc/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN=python3; fi
exec "$PYTHON_BIN" main.py "$@"
