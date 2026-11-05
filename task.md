# CODEX TASK — Xây dựng LifeMate Voice Agent Pipeline cho Zalo AI Challenge 2026

Bạn là Senior AI Engineer chuyên về ASR, LLM inference và tool-calling agents.

Hãy triển khai một hệ thống inference Python chạy local cho bài thi **Zalo AI Challenge 2026 – LifeMate**.

**Mục tiêu ưu tiên:** hệ thống gọn, chạy được, dễ debug, thay model bằng config, kiểm tra được kết quả từng bước. Không xây framework phức tạp, UI, API server hoặc training pipeline.

Trước khi code, hãy kiểm tra cấu trúc repository hiện có, tận dụng model inference hoặc SDK đã có, không viết lại những thành phần đang hoạt động.

## 1. Luồng xử lý bắt buộc

Input: `audio + context + history + available tools + simulation environment`.

Pipeline:

1. ASR: Audio → `asr_text`
2. Normalizer: Qwen3.5-4B → `normalized_text`
3. Tool Agent: Qwen3.5-4B → quyết định hành động cần thực hiện
4. Tool Executor: gọi tool, nhận observation, tiếp tục gọi Qwen nếu cần
5. Export: `id + tool_calls + state_hash`

Chỉ có **3 output chính để kiểm tra**:

- `asr_text`: kết quả nhận dạng giọng nói.
- `normalized_text`: kết quả Qwen sửa và chuẩn hóa transcript.
- `tool_calls`: danh sách tool đã thực thi theo đúng thứ tự.

`state_hash` là dữ liệu bổ sung bắt buộc cho file submission.

Các bước phải độc lập để có thể chạy riêng hoặc resume từ output đã lưu.

## 2. Cấu trúc code

Tạo cấu trúc tối giản:

- `run.sh`: toàn bộ config chính, đường dẫn model, chọn ASR, run mode, debug mode, system prompts.
- `main.py`: CLI và điều phối pipeline.
- `asr/gipformer.py`: inference riêng GipFormer.
- `asr/phowhisper.py`: inference riêng PhoWhisper.
- `llm_qwen.py`: load Qwen và thực hiện generate, sử dụng chung cho normalize và tool agent.
- `agent.py`: normalization, tool planning và vòng lặp agent.
- `env_adapter.py`: kết nối môi trường/tool execution của Zalo development kit, cung cấp mock adapter khi test.
- `tests/test_pipeline.py`: smoke test cơ bản.
- `README.md`, `requirements.txt`: hướng dẫn và dependencies tối thiểu.

Chỉ thêm file khi thật sự cần. Không dùng LangChain, LangGraph hoặc framework agent khác nếu Python thông thường đã đủ.

## 3. run.sh — nơi điều khiển duy nhất

Tôi muốn chỉ sửa `run.sh` để đổi cấu hình hệ thống.

Các biến bắt buộc:

**ASR**

- `ASR_MODE=single` — cho phép `single` hoặc `dual`.
- `ASR_PRIMARY=gipformer`
- `ASR_SECONDARY=phowhisper_large`
- `ASR_COMBINE=primary` — cho phép `primary` hoặc `llm_review`.
- `GIPFORMER_MODEL_PATH=...`
- `PHOWHISPER_MODEL_PATH=...`

**LLM**

- `LLM_MODEL_PATH=Qwen/Qwen3.5-4B`
- `LLM_BACKEND=transformers`
- `LLM_DTYPE=bfloat16`
- `LLM_QUANTIZATION=none`
- `LLM_CONTEXT_TOKENS=16384`
- `LLM_TEMPERATURE=0`
- `NORMALIZE_MAX_NEW_TOKENS=512`
- `AGENT_MAX_NEW_TOKENS=1024`

Sử dụng checkpoint Qwen3.5-4B đầy đủ, không tự thay model hoặc tự bật quantization.

**Execution**

- `RUN_MODE=all` — nhận `asr`, `normalize`, `agent`, `all`.
- `DEBUG=1`
- `SAVE_INTERMEDIATE=1`
- `RESUME=0`
- `ENV_MODE=mock` — nhận `mock` hoặc `official`.
- `MAX_AGENT_STEPS=12`
- `OUTPUT_DIR=outputs`
- `STRICT_COMPETITION=1`

Toàn bộ system prompt phải được khai báo trực tiếp trong `run.sh` bằng multiline heredoc, truyền vào Python qua environment hoặc file tạm.

Có hai prompt riêng:

- `NORMALIZE_SYSTEM_PROMPT`
- `AGENT_SYSTEM_PROMPT`

Không hardcode nội dung system prompt trong các file Python. Python chỉ xây dựng input động từ context, tool schemas, observations và lịch sử.

Cho phép truyền tham số CLI để override run mode hoặc input mà không phải sửa `.sh` mỗi lần chạy.

## 4. Module ASR

Mỗi mô hình ASR có một file inference riêng, cung cấp giao diện thống nhất:

`transcribe(audio_path) -> str`

GipFormer:

- Ưu tiên dùng checkpoint hoặc code inference đã tồn tại trong repository.
- Nếu không có thì hỗ trợ `g-group-ai-lab/gipformer1.5-65M-rnnt`.
- Chỉ trả về transcript dạng string.
- Không tự thực hiện ITN phức tạp trong module ASR.

PhoWhisper:

- Hỗ trợ `vinai/PhoWhisper-large` và các checkpoint PhoWhisper khác qua config.
- Chỉ trả về transcript dạng string.
- Không trộn logic inference PhoWhisper vào GipFormer.

**Single mode:** chỉ chạy primary ASR.

**Dual mode:** chạy cả primary và secondary.

Ban đầu không cần xây thuật toán ensemble phức tạp.

- `primary`: dùng transcript của primary.
- `llm_review`: đưa cả hai transcript vào bước Qwen normalization để xem xét sự khác biệt.

Lưu riêng transcript mỗi ASR trong debug log, nhưng output chính vẫn là một `asr_text`.

Không mặc định transcript nào có WER tốt hơn. Không tự ghép các từ theo vị trí một cách thiếu căn cứ.

Các model phải hỗ trợ load/unload để tránh OOM. Cho phép chạy ASR tuần tự rồi giải phóng VRAM trước khi load Qwen.

Nếu thiếu checkpoint hoặc dependency, báo lỗi rõ ràng. Không được âm thầm trả về transcript giả.

## 5. Module Qwen — Normalization

Chạy Qwen3.5-4B với `NORMALIZE_SYSTEM_PROMPT`.

Nội dung prompt mặc định:

"Bạn là hệ thống hiệu chỉnh và chuẩn hóa transcript tiếng Việt dành cho Voice Agent.

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

Chỉ trả về JSON có khóa normalized_text. Không giải thích."

Ví dụ:

Input: `bật điều hòa phòng khách hai mươi sáu độ`

Output: `{"normalized_text": "Bật điều hòa phòng khách 26 độ."}`

Module phải parse và validate JSON trước khi trả kết quả.

Nếu parse lỗi, cho phép retry có kiểm soát tối đa một lần. Lưu raw output để debug.

Không được thay đổi transcript gốc của ASR sau khi chuẩn hóa; lưu riêng cả hai.

## 6. Module Qwen — Tool Agent

Sử dụng **cùng checkpoint Qwen đã load**, không load thêm một LLM khác.

Input của agent gồm:

- `normalized_text`
- `context`: thời gian, timezone, vị trí, user_profile
- `history`: lịch sử hội thoại
- `available_tools`: danh sách tool thực sự được cung cấp
- `tool_schemas`: JSON Schema chính thức của từng tool
- `observations`: kết quả các tool đã thực thi

Prompt mặc định:

"Bạn là LifeMate, một AI Agent thực hiện yêu cầu của người dùng bằng các công cụ có sẵn.

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
- Không sinh lời giải thích ngoài JSON."

Mỗi lượt Qwen chỉ được trả về một quyết định:

- `{"action":"call","name":"tool_name","arguments":{...}}`
- hoặc `{"action":"finish"}`

Đây là **giao thức nội bộ của hệ thống**, không phải định dạng submission chính thức.

Phải xử lý JSON parsing, schema validation và giới hạn số bước. Không thực thi tool khi model trả về JSON không hợp lệ.

## 7. Tool Executor và Agent Loop

Đây là phần quan trọng nhất.

Không để Qwen tự sinh toàn bộ tool_calls rồi bỏ qua phản hồi môi trường.

Luồng thực hiện:

1. Qwen chọn tool tiếp theo.
2. Python kiểm tra tool có tồn tại không.
3. Validate arguments theo JSON Schema.
4. Nếu hợp lệ, gọi tool thông qua environment adapter.
5. Nhận observation thật từ môi trường.
6. Lưu kết quả vào agent history.
7. Gửi observation trở lại Qwen.
8. Lặp lại đến khi `finish`, cần hỏi thêm thông tin hoặc đạt giới hạn số bước.

Danh sách `tool_calls` cuối cùng phải được xây dựng từ **những lời gọi đã thực sự thực thi**, không phải những lời gọi mới chỉ được model dự đoán.

Nếu tool lỗi, lưu lỗi và cho phép Qwen xử lý bước tiếp theo. Không tự động lặp lại các hành động có side effect như chuyển tiền hoặc đặt xe nếu chưa xác định được lần gọi trước có thành công hay không.

Không hardcode tool nghiệp vụ vào agent. Chỉ mock một số tool trong test adapter.

Môi trường chính thức phải được tích hợp bằng SDK/devkit của Zalo. Hãy kiểm tra code và tài liệu devkit hiện có trước khi gọi API. Không tự sáng tạo API của ban tổ chức.

Sau khi hoàn thành, lấy `state_hash` từ môi trường thông qua phương thức tương ứng `env.get_state_hash()`.

Tuyệt đối không hardcode hash mẫu, không tự tạo hash thay cho môi trường chính thức.

## 8. Mock và Official Environment

Hệ thống cần hai chế độ rõ ràng.

**Mock mode**

Cho phép kiểm tra pipeline ngay cả khi chưa cấu hình xong development kit.

Tạo ví dụ đơn giản có:

- `smart_home.get_status`
- `smart_home.climate_set`
- `ride.book`

Mock phải lưu và cập nhật trạng thái, trả observation cho agent.

Mock mode chỉ phục vụ debug, không được tạo file được coi là submission chính thức.

**Official mode**

Sử dụng đúng dữ liệu đầu vào, tool schemas, môi trường mô phỏng và bộ evaluator của Zalo.

Không được thực thi tác vụ trên ứng dụng ngân hàng hoặc đặt xe thật.

Nếu chưa có devkit trong repository, tạo adapter với thông báo thiếu tích hợp rõ ràng, không được giả vờ rằng official execution đã thành công.

## 9. Intermediate output và Debug

Mỗi sample lưu vào `outputs/<sample_id>/`.

Tạo `result.json` với các field chính:

`id`, `asr_text`, `normalized_text`, `tool_calls`, `state_hash`.

Trong đó `state_hash` có thể null ở những bước chưa chạy môi trường.

Debug chi tiết lưu riêng vào `trace.jsonl`, gồm:

- Tên stage.
- Tên model.
- Input/output của từng lần inference.
- Prompt thực sự gửi cho Qwen.
- Tool name, arguments và observation.
- Latency của từng bước.
- Parse/schema error nếu có.

Không nhồi metadata debug vào `result.json`.

Khi `DEBUG=1`, terminal in lần lượt:

`[ASR]`, `[NORMALIZED]`, `[TOOL]`, `[FINAL]`.

Khi `DEBUG=0`, chỉ in kết quả ngắn gọn và lỗi.

**Resume:** Có thể sử dụng kết quả stage trước đó mà không cần inference lại. Cache phải gắn với sample và cấu hình model/prompt để tránh dùng nhầm kết quả khi thay model.

## 10. CLI phải chạy được

Tạo một `run.sh` hỗ trợ các lệnh:

- `./run.sh asr --audio sample.wav`  
  Chỉ nhận dạng tiếng nói.

- `./run.sh normalize --audio sample.wav`  
  Chạy ASR rồi Qwen normalization, dừng trước tool agent.

- `./run.sh normalize --text "bật điều hòa hai mươi sáu độ"`  
  Bỏ qua ASR, test riêng Qwen normalization.

- `./run.sh agent --case examples/sample_case.json --text "Bật điều hòa phòng khách 26 độ"`  
  Bỏ qua ASR, test normalization và agent trên một case có context/environment.

- `./run.sh all --case examples/sample_case.json`  
  Chạy đầy đủ từ audio đến tool execution và final result.

- `./run.sh all --input data/public_test.jsonl --output submission.jsonl`  
  Chạy batch với định dạng input được điều chỉnh theo devkit.

Cho phép bổ sung `--resume` và `--debug` khi cần.

Phần chạy batch phải lưu một JSON object trên mỗi dòng, đúng format:

`{"id":"...","tool_calls":[...],"state_hash":"..."}`

Chỉ tạo submission hợp lệ trong official mode khi các bước execution và hash đã thành công. Không thay giá trị thiếu bằng hash hoặc tool_calls giả.

## 11. Kiểm tra giới hạn cuộc thi

Các ràng buộc:

- Tổng tham số của toàn hệ thống phải dưới 6B.
- LLM phải dưới 5B.
- Context tối đa 16.384 tokens.

Đếm tổng số tham số của các checkpoint được cấu hình trong hệ thống, chỉ tính Qwen một lần dù được dùng cho hai nhiệm vụ.

Không nhầm quantization với giảm số tham số.

Cảnh báo hoặc chặn các tổ hợp vượt giới hạn khi `STRICT_COMPETITION=1`.

Có thể cho phép cấu hình vượt giới hạn khi chạy nghiên cứu nội bộ, nhưng phải hiển thị rõ `OVER COMPETITION LIMIT` và không cho xuất submission chính thức.

Phải kiểm soát tổng số token prompt + history + observation + output reserve, không vượt 16.384.

## 12. Test bắt buộc

Tạo smoke tests tối thiểu:

**Test A — Multi-step**

Câu nói: "Tôi sắp về nhà, bật điều hòa phòng khách 26 độ và đặt xe đón tôi."

Context:
- Current location: 227 Nguyễn Văn Cừ, Quận 5, TP.HCM
- Home: 123 Trần Hưng Đạo, Quận 5, TP.HCM

Kỳ vọng:
- Agent đọc trạng thái smart home.
- Lấy đúng device_id từ observation.
- Chỉnh điều hòa về 26 độ.
- Đặt xe từ vị trí hiện tại về nhà.
- Lưu đúng chuỗi tool calls đã thực thi.

**Test B — Missing information**

Yêu cầu thiếu một thông tin bắt buộc không có trong context.

Kỳ vọng:
- Không tự bịa thông tin.
- Dùng `sys.ask_missing_info` theo schema của môi trường.

**Test C — Out of scope**

Yêu cầu không có ứng dụng nào xử lý được.

Kỳ vọng:
- `tool_calls=[]`.
- Không gọi tool sai phạm vi.

Các smoke test có thể dùng transcript được cung cấp sẵn, không cần load model để kiểm thử riêng orchestration. Đồng thời phải có đường chạy inference thật khi checkpoint được cấu hình.

## 13. Yêu cầu chất lượng code

- Python 3.10+.
- Có type hints ở những interface quan trọng.
- Không hardcode model path hoặc prompt trong Python.
- Không viết class hierarchy phức tạp.
- Không load Qwen hai lần.
- Không tự tạo fallback sang mô hình khác.
- Có exception rõ ràng.
- Giữ output thống nhất khi đổi ASR hoặc LLM.
- Đặt nhiệt độ sinh mặc định bằng 0 hoặc chế độ deterministic tương đương do backend hỗ trợ.
- Hỗ trợ xử lý OOM bằng cấu hình offload hoặc tuần tự, không âm thầm thay precision.
- Ưu tiên sửa code hiện tại thay vì tạo lại toàn bộ repository.

## 14. Definition of Done

Chỉ coi hoàn thành khi:

1. `run.sh` thật sự chạy được các mode đã mô tả.
2. Test ASR độc lập cho kết quả text.
3. Test normalization độc lập cho kết quả text.
4. Agent gọi được nhiều tool liên tiếp với observation từ môi trường.
5. Có thể bật/tắt ASR thứ hai bằng config.
6. Có thể thay checkpoint model chỉ bằng config.
7. Có debug trace cho từng lượt gọi model/tool.
8. Mock smoke tests chạy thành công.
9. Official adapter không sử dụng API giả và báo rõ nếu thiếu devkit.
10. Export submission chỉ dùng tool calls đã thực thi và state hash thật của official environment.

**Thứ tự thực hiện:**

Trước tiên kiểm tra repo, triển khai `run.sh` và pipeline single-ASR. Sau đó triển khai normalization, agent loop, mock environment và debug. Cuối cùng mới thêm dual-ASR và batch submission.

Không fine-tune model. Không xây UI. Không mở rộng kiến trúc vượt phạm vi này.

Sau khi code, hãy thực sự chạy các lệnh smoke test, sửa lỗi nếu có, rồi báo cáo ngắn gọn: file đã tạo/sửa, lệnh chạy được, kết quả test, các checkpoint hoặc thành phần official devkit còn thiếu. Không báo thành công nếu chưa chạy kiểm tra thực tế.