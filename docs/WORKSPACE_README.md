# LifeMate local pipeline

Sinh text ASR bằng API Google từ từng lượt user trong tập train: `./run_asr_datagen_api.sh run`.
Config và quota ở `asr_datagen_api_config.json`, prompt ở `prompts/asr_datagen_system.txt`;
token chỉ đọc từ `API_gg.txt`, không ghi token vào prompt/log. API key không làm tăng quota riêng:
worker dùng chung một bộ đếm bảo thủ cho mọi key và chỉ đổi key khi gặp lỗi xác thực.
Theo thứ tự Gemini 2.5 → 3.1 → 3.5 Flash Lite → Gemma 4 26B → 31B; tự chờ RPM/input-TPM,
đổi model khi hết quota ngày hoặc model không truy cập được; lỗi 503/429 tạm thời có backoff và thử nguồn khác.
Quota ngày reset theo giờ Pacific. Gemma 4 dùng thinkingLevel=minimal để giảm thời gian suy luận cho tác vụ diễn đạt lại.
Gemini 2.5 đã trả HTTP 404 cho tài khoản này nên được đánh dấu bỏ qua; response gốc được giữ.

Mỗi câu cuối hoặc câu user history là một prompt riêng (5.036 prompt), có đúng phần hội thoại trước nó,
reference, hai ASR và gợi ý lỗi từ khảo sát. Mỗi prompt sinh 10 câu: 5 hướng Gip tiếng Việt và 5 hướng Zip
tiếng Anh/tên/mã trong câu Việt. Không ép chèn tiếng Anh khi không phù hợp. Tên tiếng Anh khó sẵn có được giữ;
nếu nhóm Zip thay một slot thì ghi `slot_changes` và không sao chép nhãn tool chuẩn cho câu mới.
Đây là text để rà rồi dựng audio; chưa TTS, fine-tune hay chấm gọi tool. Toàn bộ corpus gốc đã được khảo sát,
vì vậy cả các dòng `source_original_split=test` trong đợt kỹ thuật trước cũng chỉ dùng làm dữ liệu train.

Output `outputs/asr_datagen_api/`: `generated.csv` để đọc, `generated.jsonl`, `gipformer.jsonl`,
`zipformer.jsonl`; `prompts/` và `requests/`/`responses/` giữ đầy đủ đầu vào/kết quả từng lần gọi,
`results/` lưu nguyên tử theo seed, `api_events.jsonl` ghi quota/lỗi/usage, `progress.json` và `summary.json`.
Các file gộp được cập nhật mỗi 10 seed và khi worker dừng. `qc_status` chỉ xác nhận cấu trúc;
ngữ nghĩa và độ tự nhiên vẫn cần rà. `semantic_warnings` gợi ý cụm quan trọng có thể bị mất.
Chạy lại cùng lệnh sẽ bỏ qua seed đã xong, không gọi lại. Nếu prompt/config/input thay đổi, phải dùng thư mục output mới.
Các lượt không đạt cấu trúc sau tối đa 5 lần được ghi `failed/` để rà, các seed khác tiếp tục chạy;
lần retry có thêm phản hồi lỗi cụ thể. `run --retry-failed` mở lại các seed lỗi, giữ mọi response cũ.
`./run_asr_datagen_api.sh status` đọc tiến độ; `./run_asr_datagen_api.sh export` xuất lại từ kết quả đã lưu
khi worker đã dừng. `--limit 1` hoặc `--seed-id train_1474_last` dùng để thử một số seed.
`./run_asr_datagen_api.sh recover` rà lại response đã có sau khi sửa bộ kiểm tra, không gọi API.
Các pilot được giữ ở `outputs/asr_datagen_api_pilot_v1/` và `outputs/asr_datagen_api_pilot_v2/`, không trộn vào corpus
sau khi sửa prompt; bộ đếm quota được giữ xuyên suốt. `--model gemma-4-26b-a4b-it --limit 1` thử riêng model dự phòng.
Lượt chạy toàn bộ hiện dùng user service `lifemate-asr-datagen.service`; log tại
`outputs/asr_datagen_api/run.log`. `systemctl --user status lifemate-asr-datagen.service` kiểm tra service;
`./run_asr_datagen_api.sh status` đọc số câu đã xong mà không can thiệp worker.

Ghép ASR theo timestamp/từng từ đã triển khai ở `asr/word_fusion.py`. Cả lượt cuối và audio user history dùng cùng bộ ghép.
Ví dụ đọc hoàn toàn từ cache đã lưu:

```sh
ASR_MODE=dual ASR_PRIMARY=zipformer ASR_SECONDARY=gipformer ASR_COMBINE=timestamp_word ASR_CACHE_ONLY=1 ./run.sh asr --id train_0001
```

`FUSION_SELECTOR_PATH=outputs/timestamp_word_fusion/selector.json` chọn bộ hiệu chỉnh đã đóng băng;
`ASR_TIMED_CACHE_DIR=outputs/asr_timed_v1` chứa BPE token/timestamp/log-probability của đủ 5.036 audio × 2 model.
`transcribe_detailed()` giữ dữ liệu gốc; `transcribe()` vẫn trả string như cũ. Mốc kết thúc từ là ước lượng, không phải forced alignment.
Mạng từ căn theo nội dung và thời gian, hỗ trợ thiếu/nối/tách từ; ràng buộc số/tên Latin/thuốc/phủ định để không trộn mảnh tùy ý.
Bộ chọn dùng điểm đã hiệu chỉnh và ngữ cảnh n-gram học từ phần fit; không dùng Qwen viết lại hai transcript ở bước ghép.

`bash run_timestamp_fusion.sh` tiếp tục cache còn thiếu rồi áp bộ chọn đã đóng băng và xuất so sánh từ khóa.
`FUSION_REFIT=1 bash run_timestamp_fusion.sh` mới hiệu chỉnh lại trên phần fit/validation đã cố định.
Để chỉ đổi cách xuất/kiểm tra từ cache: `/home/pc/venv/bin/python apply_word_fusion.py` rồi `/home/pc/venv/bin/python compare_word_fusion.py`.
Không cần nhận dạng hay chạy tool environment lại cho hai lệnh đó.

Kết quả tại `outputs/timestamp_word_fusion/comparison/`: `BAO_CAO_GHEP_TU.md`, `keyword_summary.csv`,
`keyword_evidence.csv`, `word_decisions.csv`, `medical_keywords.csv`, `language_review.csv`.
`needs_selector_fix.csv` tách ca có nguồn ASR đúng mà ghép chọn sai; `both_asr_data_candidates.csv` tách ca cả hai nguồn vẫn cần rà để gen data.
Toàn tập có 5.036 audio; 3.558 fit, 649 validation và 829 test theo hội thoại/template số của câu cuối.
Đây là phân vùng kiểm tra kỹ thuật trên train từng được khảo sát; số liệu cờ từ khóa không phải điểm calling chính thức.

So sánh ASR theo domain để chọn data: `/home/pc/venv/bin/python compare_asr_domains.py` chỉ đọc output đã lưu,
xuất `outputs/zipformer_150m/domain_comparison/BAO_CAO_SO_SANH_ASR.md`, CSV 60 trường được rà ngôn ngữ,
thống kê cụm thuốc/Anh/tên/mã/lệnh/giá trị và seed gen data. Cờ cụm tự động là ứng viên cần rà, không phải tỷ lệ lỗi ngữ nghĩa đã chấm tay.

Kết hợp hai ASR: `bash run_dual_asr_benchmark.sh` xác minh và dùng lại 10.072 transcript đã có cho 5.036 audio;
`ASR_CACHE_ONLY=1` chặn nhận dạng lại nếu thiếu cache. `ASR_MODE=dual ASR_COMBINE=llm_review ASR_REVIEW_HISTORY=1`
đưa cả hai transcript vào Qwen ở lượt cuối và từng audio user history, giữ riêng cache normalize của history.
Không đưa reference vào inference. Benchmark 1.545 mẫu lưu riêng tại `outputs/dual_asr_qwen/`.
Service nền: `lifemate-dual-asr-benchmark.service`; log `run.log`; tiến độ `benchmark_seed42_n1545/progress.json`.
Kết thúc tự xuất `comparison.csv` và `comparison_summary.json` so sánh GipFormer, ZipFormer và Qwen kết hợp.
Lượt kết hợp có thêm review history, nên cần đọc đây là thử nghiệm cả luồng kết hợp với cấu hình đã ghi trong run_metadata.json.

Bằng chứng ASR cho agent: `AGENT_ASR_EVIDENCE=1` đưa các bản ASR thô (không qua normalizer) vào payload agent: `asr_transcripts` cho lượt cuối,
`asr_alternatives` cho lượt user trong history (chỉ khi khác bản đã chọn ngoài chữ hoa/dấu câu), và nối khối quy tắc trong `run.sh`
(bản sao đọc: `prompts/tool_agent_asr_evidence.txt`). Ở dual mode, history luôn nhận dạng thêm ASR phụ khi cờ bật; `timestamp_word` đưa cả hai bản gốc.
`bash run_agent_asr_evidence_benchmark.sh` giữ nguyên cấu hình của `run_dual_asr_benchmark.sh`, chỉ bật cờ này, dùng lại cache ASR đã xác minh,
rồi chạy `compare_benchmark_runs.py` để xuất `outputs/agent_asr_evidence/comparison/` (đổi kết quả theo cặp, theo nhóm từ khoá ASR giữ/lệch, theo số tool call).
Cache `asr_timestamp_word` nay lưu cả text ghép và hai bản gốc; cache dạng chỉ-text cũ sẽ được tính lại từ timed cache, không chạy lại model.

Thử model hynt ZipFormer 150M: `ASR_PRIMARY=zipformer ./run.sh asr --id train_1474`.
Checkpoint FP32 được lưu tại `models/zipformer-150M-6000h`; revision, checksum và số tham số nằm trong `download_manifest.json`.
`config.json` của repo này chứa bảng 2.000 token; `tokens.txt` là bản sao nguyên nội dung đã kiểm tra ID.

`bash run_zipformer_benchmark.sh` chạy ASR toàn bộ 5.036 audio train trước, rồi Qwen normalization/tool calling trên 1.545 mẫu với cấu hình như baseline.
Output riêng ở `outputs/zipformer_150m/`: `asr_progress.json`, `asr_results.jsonl`, `asr_transcripts.csv`;
`benchmark_seed42_n1545/progress.json`, `results.jsonl`, `summary.json`, `run_metadata.json` và `samples/<id>/trace.jsonl`.
Cache ASR theo nội dung audio/model được ghi nguyên tử và đọc lại ở bước pipeline; restart không nhận dạng lại audio đã xong.
Mỗi mẫu có cache normalize, prompt Qwen, quyết định thô và tool/observation. Mẫu đã mở phiên simulator nhưng bị gián đoạn được giữ lỗi, không tạo lại phiên.
Kết thúc tự xuất `calling_comparison.csv` (ASR, normalize, gold/actual calls của hai model), `asr_comparison.csv` cho cả history và `comparison_summary.json`.
Chỉ đọc các file này để phân tích; không cần chạy inference lại. Tiến trình nền: `systemctl --user status lifemate-zipformer-benchmark.service`;
log: `outputs/zipformer_150m/run.log`. Có thể chạy lại script để tiếp tục các mục còn lại bằng cache.

Đánh giá ASR theo dạng lời nói: `/home/pc/venv/bin/python evaluate_spoken_asr.py` đọc transcript cache của benchmark full, chuẩn hóa cách viết số trên cả reference và ASR, rồi xuất `outputs/asr_spoken_eval/`. `summary.json` giữ riêng WER nguyên bản, WER sau chuẩn hóa số và WER có thêm quy tắc rút gọn theo ngữ cảnh. Xem cặp text trước/sau trong `normalization_changed.csv`; các báo cáo audit cũ tính trên dạng chữ viết ban đầu. Quy tắc này chỉ dùng cho đánh giá, không thay transcript hoặc model inference.

Chạy Python 3.10+; chỉnh cấu hình, model và hai system prompt trong `run.sh`. Có thể đặt `PYTHON_BIN` để chọn môi trường Python. `/home/pc/venv` ở máy hiện tại có PyTorch, sherpa-onnx, nhưng Transformers 4.57.1 quá cũ cho Qwen3.5; nên tạo môi trường riêng và cài `pip install -r requirements.txt`. Không tự thay đổi môi trường dùng chung của v2t. Chọn `LLM_DEVICE_MAP=auto` để Accelerate offload khi cần; `ASR_DEVICE=cpu` tránh tranh VRAM. Tải các checkpoint gốc từ Hugging Face hoặc cấu hình đường dẫn local. GipFormer cần ONNX encoder/decoder/joiner và tokens.txt tương thích sherpa-onnx; PhoWhisper cần Transformers.

```sh
./run.sh asr --audio sample.wav
./run.sh normalize --audio sample.wav
./run.sh normalize --text 'bật điều hòa hai mươi sáu độ'
./run.sh agent --case examples/sample_case.json --text 'Bật điều hòa phòng khách 26 độ'
./run.sh all --case examples/sample_case.json
python3 -m unittest discover -s tests -v
```

`--resume` sử dụng cache ASR/normalize theo nội dung audio, model revision/cấu hình và prompt. Cache không phục hồi execution: môi trường phải được khởi tạo lại, không tự chạy lại tác vụ đã thực hiện trên một môi trường đang tồn tại. `result.json` chỉ chứa kết quả chính; trace có prompt, model, thời gian, lỗi và observation, có thể chứa thông tin cá nhân. `SAVE_INTERMEDIATE=0` tắt result/cache, trace vẫn lưu. `ASR_MODE=dual ASR_COMBINE=llm_review` chạy ASR tuần tự và đưa cả hai transcript cho normalizer.

Mock là mô phỏng cục bộ và hash mock chỉ để debug. Bộ dữ liệu đã giải nén tại `data/train+public_test`, và official mode dùng `zaic_tools` tương ứng từng split, kiểm tra schema qua devkit và lấy state hash trực tiếp từ môi trường. Ví dụ kiểm thử từng mẫu:

```sh
ENV_MODE=official DATA_SPLIT=train ./run.sh all --id train_0002
ENV_MODE=official ./run.sh all --split public_test --id public_0002
```

Mỗi mẫu chỉ tạo một phiên môi trường khi chạy; theo hướng dẫn bộ dữ liệu, không reset một phiên để thử rồi làm lại. Audio trong history được nhận dạng bằng primary ASR; transcript chuẩn trong train không được đưa vào model. Batch submission chưa triển khai. Không dùng mock để nộp. Kết quả pilot chỉ là kiểm thử, chưa đủ cơ sở để nộp bài.

Tài liệu checkpoint: [GipFormer](https://huggingface.co/g-group-ai-lab/gipformer1.5-65M-rnnt), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large).

Checkpoint GipFormer 1.5 đã tải tại `models/gipformer1.5-68M-rnnt` (tên model 65M cũ được Hugging Face chuyển hướng về 68M). `run.sh` dùng đường dẫn local này mặc định. Đã thử nhận dạng với audio mẫu từ repository GipFormer.

Trên máy này, Qwen3.5-4B đã có tại `models/Qwen3.5-4B`; hai shard là symlink đến checkpoint ở `/home/pc/qwen`, đã đối chiếu SHA-256 chính thức. Với máy khác, có thể tải lại (khoảng 9,3 GB):

```sh
huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
# Sau khi tải đầy đủ, đặt LLM_MODEL_PATH=models/Qwen3.5-4B trong run.sh.
```

Mặc định `LLM_GPU_MAX_MEMORY=7GiB` để chừa VRAM cho tiến trình GPU hiện có; có thể chỉnh trong `run.sh`. `.deps521` và `.deps` là dependencies cài cục bộ để thử nghiệm, `run.sh` tự sử dụng nếu có. Lượt thử Transformers 5.21 với offload chạy thành công; đỉnh PyTorch khoảng 6,72 GiB, thời gian sinh 5,2 giây. Không đảm bảo mức VRAM này đủ với prompt dài hơn.

Khi load Qwen, hệ thống đếm tham số Qwen và tensor trong checkpoint ASR; đối với GipFormer số đếm gồm cả buffer nên là cận trên. `STRICT_COMPETITION=1` chặn nếu không xác minh được hoặc vượt giới hạn. Đã kiểm chứng số đếm trên checkpoint Qwen thực tế.

Đo trên mẫu train ngẫu nhiên với seed cố định: `ENV_MODE=official LLM_GPU_MAX_MEMORY='' LLM_DEVICE_MAP=cuda:0 DEBUG=0 ./run.sh all --benchmark 100 --seed 42`. Báo cáo state hash, tool calls khớp chính xác và các mẫu lỗi; đây không phải điểm leaderboard. `--resume` chỉ tiếp tục các mẫu chưa tạo phiên; mẫu dở có thể bị tính lỗi để tránh reset simulator.
