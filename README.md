# LifeMate: ASR tiếng Việt, ghép từ theo timestamp và sinh dữ liệu

Code, prompt và snapshot kết quả nghiên cứu cho Zalo AI Challenge LifeMate. Luồng chính:

`audio → GipFormer / ZipFormer → ghép từ (tuỳ chọn) → Qwen chuẩn hoá → Qwen gọi tool → môi trường mô phỏng`

## Bắt đầu đọc ở đâu

- [Kết quả và giới hạn của từng thử nghiệm](docs/RESULTS_CURRENT.md).
- [Danh mục CSV/báo cáo](outputs/README.md).
- [Khảo sát lỗi ASR để gen data](outputs/asr_datagen_targets/KHAO_SAT_DE_GEN_DATA.md).
- [So sánh từ khoá khi ghép hai ASR](outputs/timestamp_word_fusion/comparison/BAO_CAO_GHEP_TU.md).
- [Kiểm tra chất lượng câu sinh](outputs/asr_datagen_quality_20261104/BAO_CAO_CHAT_LUONG.md) và [252 câu đã chấm](outputs/asr_datagen_quality_20261104/reviewed_examples.csv).
- [Data để fine-tune: data đã gen dùng được cho ASR/agent đến đâu, data có sẵn, nguồn ngoài](docs/DATA_FINETUNE.md).
- [Text sinh hiện tại](outputs/asr_datagen_api/generated.csv): **10.980 câu / 1.098 seed**, tại snapshot 09:06 ngày 05/11/2026 (UTC+7). Đây là dữ liệu thô cần rà, chưa phải bộ audio hoặc ASR đã fine-tune.

## Code và prompt

| Phần | File |
| --- | --- |
| Điều phối, config, chuẩn hoá và tool agent | `main.py`, `run.sh`, `agent.py`, `llm_qwen.py`, `env_adapter.py`, `fewshot.py` |
| ASR, timestamp và ghép từ | `asr/`, `cache_timed_asr.py`, `train_word_fusion.py`, `apply_word_fusion.py` |
| Benchmark và khảo sát lỗi | `benchmark.py`, `compare_benchmark_runs.py`, `compare_asr_domains.py`, `compare_word_fusion.py`, `asr_datagen_audit.py` |
| Sinh text qua Gemini/Gemma, quota và resume | `generate_asr_data_api.py`, `asr_datagen_api_config.json`, `run_asr_datagen_api.sh` |
| Kiểm tra chất lượng text đã sinh | `audit_generated_asr_data.py`, `finalize_asr_datagen_quality.py`, `audit_finetune_data.py` |
| Data fine-tune Qwen (agent + normalizer), few-shot pool | `build_agent_sft.py`, `build_fewshot_pool.py`, `eval_fewshot_retrieval.py`; xem [docs/DATA_FINETUNE.md](docs/DATA_FINETUNE.md) |
| Prompt sinh dữ liệu | [`prompts/asr_datagen_system.txt`](prompts/asr_datagen_system.txt) |
| Prompt chuẩn hoá | [`prompts/normalization_system.txt`](prompts/normalization_system.txt) |
| Prompt gọi tool | [`prompts/tool_agent_system.txt`](prompts/tool_agent_system.txt); khối bật theo cờ: [`tool_agent_asr_evidence.txt`](prompts/tool_agent_asr_evidence.txt), [`tool_agent_task_rules.txt`](prompts/tool_agent_task_rules.txt), [`tool_agent_fewshot.txt`](prompts/tool_agent_fewshot.txt) |

Hai prompt Qwen trong `prompts/` là bản tách để đọc; **`run.sh` là nguồn đang được thực thi**. Prompt và payload thực tế của từng lần gọi API/Qwen được lưu trong archive kết quả đầy đủ.

## Cài và chạy

Dùng Python 3.12. Cài dependencies vào môi trường riêng:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHON_BIN="$PWD/.venv/bin/python"
python -m unittest discover -s tests -v
```

71 unit tests dùng mock, không yêu cầu tải trọng số hay gọi API. Để chỉ chạy tests:

```bash
pip install requests jsonschema num2words==0.5.14
python -m unittest discover -s tests -v
```

Để inference, tải checkpoint tương ứng theo [`model_metadata/`](model_metadata/) vào:

- `models/gipformer1.5-68M-rnnt`: `g-group-ai-lab/gipformer1.5-68M-rnnt`.
- `models/zipformer-150M-6000h`: `hynt/ZipFormer-150M-CR-CTC-RNNT-6000h`.
- `models/Qwen3.5-4B`: `Qwen/Qwen3.5-4B`, đầy đủ, bf16, không tự quantize.

Giải nén bộ dữ liệu/devkit vào `data/train+public_test/` khi chạy official benchmark. Repo không kèm trọng số, audio/zip gốc hoặc môi trường Python của máy đã chạy.

```bash
./run.sh asr --audio sample.wav
ENV_MODE=official DATA_SPLIT=train ./run.sh all --id train_0002
ASR_MODE=dual ASR_PRIMARY=zipformer ASR_SECONDARY=gipformer ASR_COMBINE=timestamp_word ASR_CACHE_ONLY=1 ./run.sh asr --id train_0001
```

Đưa bản ASR thô cho agent: `AGENT_ASR_EVIDENCE=1` (mặc định 0). Agent nhận thêm `asr_transcripts` của lượt cuối và `asr_alternatives` cho lượt user trong history, kèm khối quy tắc đối chiếu/khớp tên gần âm. Mục đích là giữ lại từ khoá mà normalizer làm rơi hoặc chỉ một ASR nghe đúng. Khi tắt, prompt và payload giống hệt trước. Benchmark so cặp với lượt dual cũ, không chạy lại ASR:

```bash
bash run_agent_asr_evidence_benchmark.sh
```

Các cờ cải thiện agent (đều mặc định 0, tắt thì prompt/payload như cũ):

| Cờ | Tác dụng |
| --- | --- |
| `AGENT_TASK_RULES=1` | Quy tắc làm đủ mọi việc trong câu, nhánh điều kiện, đổi ý, khi nào hỏi lại; `sys.ask_missing_info` là bước cuối |
| `AGENT_REPEAT_GUARD=1` | Lời gọi trùng y hệt lần trước không chạy lại mà trả kết quả cũ kèm cảnh báo; gửi lại ngay thì mới chạy (tối đa 2 lần, như `redeem` hai lần trong gold) |
| `AGENT_FEWSHOT_K=3` | Thêm 3 yêu cầu train đã giải đúng, gần giống nhất (`outputs/fewshot/pool.jsonl` từ `build_fewshot_pool.py`); `AGENT_FEWSHOT_EXCLUDE=<split manifest>` loại cùng mẫu/nhóm khi benchmark trên train |
| `AGENT_THINKING=1` | Qwen suy nghĩ trước mỗi quyết định; chậm hơn nhiều, cần `AGENT_MAX_NEW_TOKENS` lớn hơn |

Hết bước mà agent chưa `finish` thì giữ các lời gọi đã chạy và state hash thật, không còn báo lỗi mất mẫu (trace `agent_step_limit`). Chạy toàn bộ cấu hình mới: `bash run_agent_v2_benchmark.sh` (ghi vào `outputs/agent_v2/`; tắt từng cờ bằng biến môi trường để ablation). `./run.sh <script>.py ...` chạy script offline với đúng cấu hình/prompt này.

Lệnh benchmark ghi vào `outputs/agent_asr_evidence/` và cần `outputs/dual_asr_qwen/asr_cache` (từ `prepare_dual_asr.py`). Lệnh `timestamp_word` ở trên cần khôi phục archive cache trước; fingerprint của model và đường dẫn audio phải khớp cấu hình. Đọc [hướng dẫn chi tiết của workspace](docs/WORKSPACE_README.md) để xem các mode, cấu hình bộ nhớ, benchmark và giới hạn cuộc thi.

## Khôi phục đầy đủ kết quả

Git chứa các bảng kết quả và báo cáo để đọc trực tiếp. Gần 100.000 file log/cache/prompt/response gốc nằm trong [Release snapshot-2026-11-05](https://github.com/minhtran26052005-glitch/test/releases/tag/snapshot-2026-11-05):

```bash
python scripts/fetch_results.py --extract
```

Script tải archive, kiểm tra SHA-256 theo [`SHARE_SNAPSHOT.json`](SHARE_SNAPSHOT.json), rồi khôi phục `outputs/`. Archive có cache ASR của 5.036 audio × 2 model, quyết định ghép từ, trace Qwen/tool, prompt và response API theo từng seed, kể cả pilot và lỗi. Các symlink giữa kết quả đã chuyển sang đường dẫn tương đối.

Các đường dẫn nằm **bên trong nội dung metadata** vẫn ghi máy nguồn để bảo toàn provenance. Khi replay trên máy khác, cần điều chỉnh đường dẫn audio/model và kiểm tra lại fingerprint; không coi mọi cache đều tự chuyển được sang máy khác chỉ bằng giải nén.

## Sinh dữ liệu bằng API

Tạo `API_gg.txt` cục bộ theo [`API_gg.example.txt`](API_gg.example.txt), điền key của chính bạn. File key và `.env` được Git bỏ qua.

```bash
./run_asr_datagen_api.sh status
./run_asr_datagen_api.sh run --limit 1
./run_asr_datagen_api.sh run
```

**`run` gọi API thật và sử dụng quota.** Worker theo thứ tự model trong config, có backoff, fallback, lưu từng response và bỏ qua seed đã hoàn thành. Quota là cấu hình quan sát tại máy nguồn; cần kiểm tra lại quota của tài khoản mới. Nếu đổi prompt/config/input, dùng output directory mới để giữ lịch sử kết quả.

Mỗi lượt user trong train là một seed: 5.036 seed × 10 biến thể = 50.360 câu dự kiến. Tại snapshot mới xong 1.098 seed, còn 26 seed lỗi kiểm tra; không đánh dấu corpus hoàn tất. Kết quả sinh chưa TTS, chưa fine-tune ASR và có các lỗi ngữ nghĩa/bản đọc đã ghi trong báo cáo chất lượng.

## Cách hiểu kết quả

- Benchmark agent là kiểm tra local trên 1.545 mẫu train có nhãn; state hash và exact calls là proxy, không phải điểm leaderboard.
- Bộ ghép từ giảm cờ lệch từ khoá trong kiểm tra kỹ thuật; chưa có benchmark full agent cho bộ ghép từ này.
- Phân vùng fit/validation/test của thử nghiệm ghép nằm trong train đã từng được khảo sát; không phải tập test hoàn toàn chưa thấy.
- Giữ riêng ASR thô, text chuẩn hoá và tool calls. Các nhãn/reference chỉ dùng offline cho đánh giá và sinh dữ liệu.

[`SHARE_SNAPSHOT.json`](SHARE_SNAPSHOT.json) ghi thời điểm, số đếm, checksum và thành phần của bản chia sẻ. Không có API key/token trong code hoặc archive được xuất.
