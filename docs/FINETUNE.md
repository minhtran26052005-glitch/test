# Fine-tune 3 model: GipFormer, ZipFormer-150M, Qwen3.5-4B

Lệnh gộp: [`run_finetune.sh`](../run_finetune.sh) (`data`, `tts`, `gipformer`, `zipformer`, `qwen`). Chạy trên máy có GPU, `data/` (devkit), `models/` và icefall.

## 1. Data ngoài: chưa đưa vào bộ fine-tune hiện tại

**Các bộ function-calling cần chuyển đổi và đánh giá trước khi dùng.** Task LifeMate cần:
- yêu cầu nói tiếng Việt, có lỗi ASR;
- history hội thoại;
- khoảng 100 tool của môi trường mô phỏng, với id lấy từ observation;
- nhiều bước nối nhau, có nhánh điều kiện;
- `sys.ask_missing_info` khi thiếu tham số;
- mỗi lượt trả về đúng một JSON `{"action": ...}`;
- chấm theo trạng thái cuối.

So với các bộ đã tìm thấy:

| Bộ | Thực chất | Lệch với task |
| --- | --- | --- |
| [5CD-AI/Vietnamese-Salesforce-xlam-function-calling-Gemini](https://huggingface.co/datasets/5CD-AI/Vietnamese-Salesforce-xlam-function-calling-Gemini) | Bản dịch xLAM; đã đọc mẫu thật, có `vi_conversations` với role `function-call` **và `function-response`** | Tool/schema/protocol khác devkit; chưa xác nhận độ phủ tình huống nhiều bước và hỏi lại. Có thể khảo sát làm data phụ sau khi chuyển định dạng |
| [wannaphong/vietnamese-Locutusque-function-calling-qwen](https://huggingface.co/datasets/wannaphong/vietnamese-Locutusque-function-calling-qwen) | Đã đọc schema và mẫu thật: `conversations`, `tools` | Cần map sang giao thức LifeMate. Mẫu đã xem không đủ để kết luận toàn bộ dataset chỉ trả lời bằng lời hoặc không hữu ích |
| [phamhai/Vietnamese-Function-Calling-Test](https://huggingface.co/datasets/phamhai/Vietnamese-Function-Calling-Test), [Multilingual-BFCL](https://huggingface.co/datasets/qualgo-technologies/Multilingual-BFCL) | Dataset đánh giá; BFCL yêu cầu quyền truy cập | Giữ riêng cho đánh giá, không trộn vào train |

Không nạp thẳng các dataset này vào train: phải chuyển tool schema, nhãn và protocol, rồi đo hiệu quả trên validation LifeMate. Khẳng định trước đây rằng xLAM không có kết quả tool là sai; lần review trên máy này đã truy cập được Hugging Face và đọc mẫu trực tiếp.

**Ưu tiên data đúng miền thi.** [Model card ZipFormer-150M](https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) liệt kê data train: VLSP2020/2021/2023, FPT, Bud500, VietSpeech, FLEURS, VietMed, một phần GigaSpeech2, viVoice và một phần PhoAudioBook. [GipFormer](https://github.com/ggroup-ai-lab/gipformer) công bố benchmark trên nhiều corpus, nhưng benchmark không chứng minh corpus đó đã dùng để train. Chưa xác nhận được danh sách train đầy đủ của Gip; cũng không thể kết luận data công khai không còn ích lợi chỉ vì đã xuất hiện trong pretrain. Bộ hiện tại ưu tiên:
- **audio train thật của cuộc thi**;
- **câu đã gen đọc bằng giọng của chính người nói trong train**.

## 2. Hai model ASR đều là icefall Zipformer2, khác nhau ở loss

| | GipFormer 1.5 (68M) | ZipFormer-150M (hynt) |
| --- | --- | --- |
| Encoder | Zipformer2, cấu hình mặc định của icefall `zipformer` | Zipformer2 ~153M, cấu hình riêng |
| Loss khi train gốc | Pruned RNN-T | CR-CTC + CTC + pruned RNN-T |
| Inference trong repo | Transducer, sherpa-onnx | Transducer, sherpa-onnx |
| Checkpoint phát hành | `model.pt` (state dict), `bpe.model`, ONNX | `pretrained.pt`, `jit_script.pt`, ONNX, `bpe.model`; code hiện nạp từ JIT đã có tại máy |
| BPE | 2.000 token | 2.000 token, **khác** BPE của Gip |

Objective fine-tune đang cấu hình trong code:
- **GipFormer**: pruned RNN-T + auxiliary CTC nếu checkpoint có head CTC; checkpoint tại máy có head này. SpecAugment áp trong dataset. Việc có head CTC không đủ xác nhận mọi hệ số loss của lần train gốc.
- **ZipFormer**: `AsrModel(use_cr_ctc=True)` của icefall. Model nhân đôi batch, mask hai bản độc lập, tính CTC trên cả hai bản và KL nhất quán giữa chúng, cộng thêm RNN-T. Hệ số theo icefall: simple 0,5, CTC 0,2, CR 0,2, `time_mask_ratio` 2,5.

Với ZipFormer, kiến trúc được dựng lại từ thuộc tính TorchScript (thiếu thì suy từ shape trọng số) và nạp trọng số đầy đủ. Sau đó **encoder và decoder được so với `jit_script.pt` trên input ngẫu nhiên**; lệch là dừng, và báo cần dùng `--model-config`.

**LoRA làm theo recipe `zipformer_lora` của icefall:**
- Gắn vào `in_proj` của attention, `in_proj` của self-attention, `in_proj`/`out_proj` của feed-forward.
- A khởi tạo kaiming, B bằng 0, nên lúc bắt đầu model trùng hệt checkpoint gốc.
- Chỉ train `lora_A/B`. Phần còn lại ở chế độ eval.
- Optimizer ScaledAdam với lr 0,0045, scheduler Eden.

Khác với recipe đó, LoRA được bọc lên bản `zipformer` hiện tại thay vì dùng `zipformer.py` cũ của recipe. Recipe có dòng `elif not self.training and ... const_attention_rate`, ngược với bản gốc.

`export_asr_lora.py` merge LoRA về đúng kiến trúc gốc (số tham số không đổi), rồi xuất:
- `encoder/decoder/joiner.onnx` bằng chính hàm của icefall `export-onnx.py`;
- `model.pt` cho Gip hoặc `jit_script.pt` cho Zip, vì `check_competition` đếm tham số từ hai file này;
- `tokens.txt` và `bpe.model`.

Script tự kiểm tra hai điều: merge khớp model còn LoRA, và ONNX khớp PyTorch.

## 3. Data đã tạo

| File | Nội dung | Số bản ghi |
| --- | --- | ---: |
| `outputs/asr_finetune/{train,validation,test}.jsonl` | Audio train thật, kể cả lượt history; đích viết như output ASR (chữ hoa, số đọc thành chữ, không dấu câu) | 3.558 / 649 / 829 |
| `outputs/asr_finetune/tts_requests.jsonl` | Câu đã gen đạt yêu cầu cho ASR, chỉ seed thuộc `fit`, kèm audio gốc để clone giọng | 7.496 |
| `outputs/qwen_sft/normalize/{train,validation,test}.jsonl` | Hai bản ASR (cả cặp lẫn từng bản riêng) → transcript reference, đúng payload của `agent.normalize` | 10.674 / 1.947 / 2.487 |
| `outputs/synthetic_agent/synthetic.jsonl` (viết tay trong chat, `build_synthetic_agent.py`) | 600 câu diễn đạt lại (300 seed `fit`, đủ 28 app chưa có data, nhiều câu 3–5 bước), 40 câu thiếu thông tin → `ask_missing_info`, 150 câu đổi ý → không gọi tool | 790 |
| `outputs/synthetic_agent/tts_requests.jsonl` | Chính 790 câu đó để TTS bằng giọng người nói của seed, nên dùng được cho cả ASR | 790 |
| `outputs/agent_sft/*.jsonl` (`./run_finetune.sh data`, cần devkit) | Gold call phát lại trong devkit, input sạch + 3 lượt ASR đã lưu + biến thể chấm tay đạt + synthetic | Xem số đo sau replay trong báo cáo review |

- **Data viết tay** (`outputs/synthetic_agent/authored/`):
  - Câu diễn đạt lại giữ nguyên mọi tên, số, điều kiện và từng việc con, nên dùng nhãn gold của seed. Khi phát lại trong devkit, script còn so được hash trạng thái cuối với nhãn.
  - Câu thiếu thông tin bỏ đúng một giá trị bắt buộc (giờ, số tiền, âm lượng…), nhãn hỏi lại giống quy ước của gold.
  - Câu đổi ý thêm một câu rút lại toàn bộ yêu cầu, nhãn rỗng.
  - Mọi câu đều qua kiểm tra tự động: từ khoá của tham số còn đủ, và giá trị số khớp câu gốc (hiểu cả "hai lăm ngàn rưỡi", "hai triệu tám", "năm chục"). Bộ kiểm tra bắt được 2 lỗi thật trong câu tôi viết, đã sửa.
- Mọi bản ghi tách theo split manifest: chỉ train trên `fit`, chọn epoch bằng `validation`, báo kết quả trên `test`.
- Data agent cần devkit vì observation phải lấy từ môi trường thật; không được tự bịa.
- TTS đọc `tts_text` nhưng học nhãn `text` lấy từ câu viết gốc: ví dụ đọc “ét em ét”, nhãn vẫn là `SMS`. Khi train, audio thật dùng chung; TTS được lọc theo `target_model` để Zip tập trung tiếng Anh, Gip tập trung tiếng Việt/domain.
- Mặc định chỉ tái sử dụng gold cho biến thể có `manual_decision=pass_text`. Khớp từ khoá tự động chưa chứng minh giữ nguyên ý; `--allow-unreviewed-generated` là lựa chọn riêng cho thử nghiệm.

## 4. Chạy

```bash
./run_finetune.sh data
./run_finetune.sh tts
./run_finetune.sh gipformer
./run_finetune.sh zipformer
./run_finetune.sh qwen
```

- **ASR**: script chấm WER validation trước khi train, rồi chấm lại sau mỗi epoch, và giữ `lora-best.pt`. Nếu không epoch nào tốt hơn model gốc thì không có `lora-best.pt`: giữ model gốc. Sau export, `eval_asr_onnx.py` so WER trên split `test` bằng đúng giải mã sherpa-onnx của pipeline.
  - Mặc định: r 16, alpha 32, dropout 0,05, 15 epoch, `--max-duration` 300 giây (200 cho Zip).
  - `--lora-targets extended` gắn thêm LoRA vào nonlin attention, conv và `out_proj` của attention.
- **Qwen**: prompt render đúng như `llm_qwen.generate`; loss chỉ tính trên JSON đích + `<|im_end|>`. LoRA gắn vào q/k/v/o, Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `out_proj`) và MLP; không đụng vision tower và MTP.
  - Mặc định: r 16, alpha 32, lr 1e-4, 2 epoch, batch 1 × gom gradient 16, bản ghi normalize lấy một nửa.
  - GPU dưới khoảng 24 GB: `QWEN_ARGS=--load-in-4bit` (QLoRA; khi merge vẫn ghi ra trọng số bf16).
  - Cài `flash-linear-attention` và `causal-conv1d` để có kernel nhanh.
  - Benchmark lại với **đúng các cờ `AGENT_*`** đã dùng khi dựng data SFT (mặc định trong `run_finetune.sh`: evidence, task rules, repeat guard; tắt few-shot).
- Chuẩn bị:
  - icefall: `git clone https://github.com/k2-fsa/icefall ~/.cache/icefall && git -C ~/.cache/icefall checkout 9c966762e058bd7ba79d9da71e42fc06b64b703e`, cài k2 và lhotse theo hướng dẫn của icefall.
  - Máy hiện tại đã có `model.pt`, `bpe.model` của Gip và `jit_script.pt`, `bpe.model` của Zip. Dùng đường dẫn hiện có; máy khác cần tải các file tương ứng.
  - Qwen cần `peft` và transformers hỗ trợ Qwen3.5. Môi trường review riêng dùng torch 2.10.0, transformers 5.21.0, peft 0.21.2; không thay môi trường dùng chung.

## 5. Mức độ đã kiểm tra

- **Qwen**: train → lưu adapter → merge đã chạy trọn trên một Qwen3.5 thu nhỏ khởi tạo ngẫu nhiên, cùng kiến trúc, với tokenizer thật, trên CPU. Đã xác nhận:
  - token id lúc train trùng với `processor(text=...)` lúc inference;
  - `<|im_end|>` là đúng một token;
  - LoRA gắn đúng 10 loại projection;
  - logits sau merge lệch 0,006 so với model còn adapter (sai số bf16), số tham số giữ nguyên.
- **ASR**:
  - Đã test trên các module mô phỏng đúng tên lớp và thuộc tính của icefall: gắn LoRA, tính SwooshL, merge chính xác, lưu/nạp adapter.
  - Đã test suy cấu hình trên state dict có cấu trúc key của icefall, và hệ số loss.
  - Đã test data loader bằng lhotse thật.
  - Review 05/11/2026 đã chạy **icefall thật, k2 thật và checkpoint thật trên CPU**: nạp cả hai model, loss/backward/update với audio thật, vòng train CLI nhỏ, merge, export ONNX và decode bằng sherpa-onnx.
- 95 unit test pass trong môi trường review. Đã sửa lời gọi VieNeu để truyền cả audio tham chiếu lẫn transcript của audio đó. Chưa train toàn bộ bằng GPU, chưa sinh audio TTS thật và chưa có kết quả tăng độ chính xác sau fine-tune. Chi tiết và số đo ở [CLAUDE_REVIEW_20261105.md](CLAUDE_REVIEW_20261105.md).
