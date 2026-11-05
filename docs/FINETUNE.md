# Fine-tune 3 model: GipFormer, ZipFormer-150M, Qwen3.5-4B

Lệnh gộp: [`run_finetune.sh`](../run_finetune.sh) (`data`, `tts`, `gipformer`, `zipformer`, `qwen`). Chạy trên máy có GPU, `data/` (devkit), `models/` và icefall.

## 1. Data ngoài: không dùng

**Các bộ function-calling tiếng Việt không đúng task.** Task LifeMate cần:
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
| 5CD-AI/Vietnamese-Salesforce-xlam-function-calling-Gemini | Bản dịch bằng Gemini của Salesforce xLAM (APIGen): một câu hỏi, danh sách API kiểu RapidAPI, đáp án là các lời gọi song song | Một lượt, không có observation, không có bước phụ thuộc kết quả trước, không hỏi lại, không có history; tool khác hẳn devkit; câu viết sạch |
| wannaphong/vietnamese-Locutusque-function-calling-qwen | Hội thoại kiểu Glaive: gọi hàm, nhận kết quả, rồi trả lời bằng lời | Đích cuối là câu trả lời tự nhiên, trái với giao thức JSON; tool chung chung (giờ hiện tại, BMI…) |
| phamhai/Vietnamese-Function-Calling-Test, qualgo-technologies/Multilingual-BFCL | Benchmark một lượt (BFCL bị khoá quyền truy cập) | Dùng để đánh giá, không để train |

Đưa các bộ này vào train có nguy cơ dạy model trả lời bằng lời, gọi hàm song song, hoặc bịa tham số. Đó đúng là các lỗi agent đang mắc. (Môi trường này chặn `huggingface.co` nên không mở trực tiếp được dataset; nhận định trên dựa vào mô tả công khai và bộ dữ liệu gốc mà chúng dẫn xuất.)

**Các bộ ASR công khai cũng không cần.** ZipFormer-150M đã được train trên VLSP2020/2021/2023, FPT, Bud500, VietSpeech, FLEURS, VietMed, một phần GigaSpeech2, viVoice và một phần PhoAudioBook. GipFormer được đánh giá trên VIVOS, Common Voice, VLSP, VietMed, MultiMED, LSVSC, Fleurs và ViMD. Data có ích là data đúng miền thi:
- **audio train thật của cuộc thi**;
- **câu đã gen đọc bằng giọng của chính người nói trong train**.

## 2. Hai model ASR đều là icefall Zipformer2, khác nhau ở loss

| | GipFormer 1.5 (68M) | ZipFormer-150M (hynt) |
| --- | --- | --- |
| Encoder | Zipformer2, cấu hình mặc định của icefall `zipformer` | Zipformer2 ~153M, cấu hình riêng |
| Loss khi train gốc | Pruned RNN-T | CR-CTC + CTC + pruned RNN-T |
| Inference trong repo | Transducer, sherpa-onnx | Transducer, sherpa-onnx |
| Checkpoint phát hành | `model.pt` (state dict), `bpe.model`, ONNX | **chỉ** `jit_script.pt` + ONNX + `bpe.model` |
| BPE | 2.000 token | 2.000 token, **khác** BPE của Gip |

Fine-tune giữ đúng loss gốc của từng model:
- **GipFormer**: pruned RNN-T, SpecAugment áp trong dataset.
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
| `outputs/agent_sft/*.jsonl` (`./run_finetune.sh data`, cần devkit) | Từng bước của agent: phát lại gold call trong devkit, payload giống inference, input sạch + input có lỗi ASR từ 3 lượt benchmark + câu đã gen | ~18k bước (ước tính) |

- Mọi bản ghi tách theo split manifest: chỉ train trên `fit`, chọn epoch bằng `validation`, báo kết quả trên `test`.
- Data agent cần devkit vì observation phải lấy từ môi trường thật; không được tự bịa.

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
  - `models/gipformer1.5-68M-rnnt` hiện chỉ có ONNX; cần tải thêm `model.pt` và `bpe.model` từ `g-group-ai-lab/gipformer1.5-68M-rnnt`.

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
  - **Chưa chạy với mã icefall thật**: ở đây không có k2 và không tải được checkpoint. Các kiểm tra lúc chạy trên máy bạn (so với `jit_script.pt`, merge, ONNX) là chốt chặn.
