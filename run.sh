#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export ASR_MODE="${ASR_MODE:-single}" ASR_PRIMARY="${ASR_PRIMARY:-gipformer}" ASR_SECONDARY="${ASR_SECONDARY:-phowhisper_large}" ASR_COMBINE="${ASR_COMBINE:-primary}"
export GIPFORMER_MODEL_PATH="${GIPFORMER_MODEL_PATH:-models/gipformer1.5-68M-rnnt}" PHOWHISPER_MODEL_PATH="${PHOWHISPER_MODEL_PATH:-vinai/PhoWhisper-large}"
export ZIPFORMER_MODEL_PATH="${ZIPFORMER_MODEL_PATH:-models/zipformer-150M-6000h}" ASR_CACHE_DIR="${ASR_CACHE_DIR:-}"
export ASR_CACHE_ONLY="${ASR_CACHE_ONLY:-0}" ASR_REVIEW_HISTORY="${ASR_REVIEW_HISTORY:-0}"
# 1: agent also sees the raw ASR hypotheses (final turn + history) and gets the ASR-evidence rules below.
export AGENT_ASR_EVIDENCE="${AGENT_ASR_EVIDENCE:-0}"
# Agent options, all off by default (prompt and payload then match the original benchmark):
# AGENT_TASK_RULES=1 rules for multi-step completeness and when to ask; AGENT_REPEAT_GUARD=1 warn instead of re-running an identical call;
# AGENT_FEWSHOT_K=N add N solved train requests similar to this one (pool from build_fewshot_pool.py; AGENT_FEWSHOT_EXCLUDE=split manifest
# drops same-group examples when benchmarking on train); AGENT_THINKING=1 Qwen reasons before each decision (raise AGENT_MAX_NEW_TOKENS).
export AGENT_TASK_RULES="${AGENT_TASK_RULES:-0}" AGENT_REPEAT_GUARD="${AGENT_REPEAT_GUARD:-0}" AGENT_THINKING="${AGENT_THINKING:-0}"
export AGENT_FEWSHOT_K="${AGENT_FEWSHOT_K:-0}" AGENT_FEWSHOT_POOL="${AGENT_FEWSHOT_POOL:-outputs/fewshot/pool.jsonl}" AGENT_FEWSHOT_EXCLUDE="${AGENT_FEWSHOT_EXCLUDE:-}"
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
agent_rules="$(cat <<'PROMPT'
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
PROMPT
)"
agent_asr_evidence_rules="$(cat <<'PROMPT'
Bằng chứng ASR:
- normalized_text là bản đã chuẩn hóa của lượt nói cuối, có thể đã sửa sai hoặc làm rơi mất từ. asr_transcripts là các bản nhận dạng thô của chính lượt nói đó; lượt user trong history có thể có asr_alternatives.
- Bản thô có thể viết hoa toàn bộ, thiếu dấu câu, đọc số bằng chữ, nghe nhầm từ gần âm, tên riêng hoặc từ tiếng Anh. Không mặc định bản nào đúng; đối chiếu các bản với nhau, với history, context và observation.
- Chi tiết như tên người, số tiền, giờ, số lượng, điều kiện, phủ định hoặc yêu cầu phụ chỉ xuất hiện trong một bản vẫn là bằng chứng; không bỏ qua chỉ vì normalized_text thiếu nó.
- Tên người, liên hệ, quán, món, thiết bị, phòng, kênh, khóa học, thuốc có thể bị nghe sai theo âm. Khi observation trả về danh sách, chọn mục có cách đọc gần nhất với các bản ASR và dùng đúng id, tên trong observation; không dùng nguyên chữ ASR sai làm id.
- Nếu tìm theo tên nghe được không có kết quả, thử lại một lần với cách viết từ bản ASR khác hoặc cách viết gần âm trước khi kết luận thiếu thông tin.
- Với nội dung tự do như tin nhắn hoặc nhắc nhở, viết theo ý đã đối chiếu, không chép nguyên lỗi ASR.
- Không hỏi lại chỉ vì các bản ASR khác nhau; sys.ask_missing_info vẫn theo quy tắc thiếu tham số bắt buộc ở trên.
PROMPT
)"
agent_task_rules="$(cat <<'PROMPT'
Làm đủ yêu cầu:
- Xác định mọi việc người dùng yêu cầu trong câu, kể cả việc phụ nối bằng "với", "rồi", "luôn", "tiện thì", "sau đó", và mọi nhánh "nếu ... thì ..., không thì ...".
- Với điều kiện, gọi tool đọc cần thiết để kiểm tra trước, rồi chỉ làm nhánh đúng theo observation; nhánh không xảy ra thì bỏ qua.
- Nếu người dùng đổi ý ("à mà thôi", "khoan", "đừng ... nữa"), chỉ làm theo ý sau cùng. Câu chỉ hỏi thông tin thì chỉ gọi tool đọc.
- Chỉ finish khi mọi việc đã thực hiện, đã xác định là không cần làm, hoặc đã gọi sys.ask_missing_info.
- Không tự chọn giá trị cho tham số bắt buộc mà người dùng không nói và context, history, observation đều không có, ví dụ số tiền, mức âm lượng, giờ, người nhận; khi đó dùng sys.ask_missing_info.
- Nếu tìm kiếm trả về nhiều mục cùng khớp mà không phân biệt được, hoặc không có mục nào khớp, hỏi lại đúng tham số id đó bằng sys.ask_missing_info.
- sys.ask_missing_info là lời gọi cuối cùng: thực hiện trước mọi phần đã đủ thông tin, rồi mới hỏi phần còn thiếu.
- Không gọi lại tool đọc với cùng tham số; dùng lại observation đã có.
PROMPT
)"
agent_fewshot_rules="$(cat <<'PROMPT'
Ví dụ tham khảo:
- examples là các yêu cầu tương tự đã được giải đúng trong dữ liệu huấn luyện; mỗi ví dụ có lời người dùng, phần cuối history và chuỗi tool_calls đúng.
- Dùng ví dụ để học cách tách việc, thứ tự gọi tool đọc trước tool ghi, khi nào hỏi lại và khi nào không gọi tool nào.
- Ví dụ thuộc môi trường khác: không chép id, tên, số hay nội dung từ ví dụ; mọi giá trị phải lấy từ yêu cầu hiện tại, context, history hoặc observation hiện tại.
PROMPT
)"
agent_repeat_rule='- Observation có repeated_call nghĩa là lời gọi trước của bạn trùng một lời gọi đã thực hiện nên chưa được chạy; đọc previous_observation thay vì gọi lại.'
agent_thinking_rule='- Khi suy nghĩ, viết ngắn: các việc cần làm, việc đã xong, bước kế tiếp.'
if [[ "$AGENT_ASR_EVIDENCE" == 1 ]]; then agent_rules+=$'\n'"$agent_asr_evidence_rules"; fi
if [[ "$AGENT_TASK_RULES" == 1 ]]; then agent_rules+=$'\n'"$agent_task_rules"; fi
if [[ "$AGENT_FEWSHOT_K" != 0 ]]; then agent_rules+=$'\n'"$agent_fewshot_rules"; fi
if [[ "$AGENT_REPEAT_GUARD" == 1 ]]; then agent_rules+=$'\n'"$agent_repeat_rule"; fi
if [[ "$AGENT_THINKING" == 1 ]]; then agent_rules+=$'\n'"$agent_thinking_rule"; fi
export AGENT_SYSTEM_PROMPT="$agent_rules"$'\n''Mỗi lượt chỉ trả về {"action":"call","name":"tool_name","arguments":{...}} hoặc {"action":"finish"}.'
if [[ -d .deps521 && -d .deps ]]; then
  export PYTHONPATH="$PWD/.deps521:$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
fi
PYTHON_BIN="${PYTHON_BIN:-/home/pc/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN=python3; fi
# ./run.sh some_script.py ... runs an offline tool (e.g. build_agent_sft.py) with this exact configuration and prompts.
if [[ "${1:-}" == *.py ]]; then exec "$PYTHON_BIN" "$@"; fi
exec "$PYTHON_BIN" main.py "$@"
