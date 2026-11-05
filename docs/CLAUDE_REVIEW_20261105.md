# Review code Claude trên máy có sẵn Gip / Zip — 05/11/2026

Đã lấy tới commit Claude **`c6b0dc8b4841ebfde39710b58cb7d61283e68182`**, gồm phần LoRA và 790 câu synthetic mới. Code chạy được qua các kiểm tra CPU bên dưới sau khi sửa lỗi. Chưa có bằng chứng tăng độ chính xác sau fine-tune toàn bộ.

Nhánh chứa bản sửa: `review/claude-finetune-20261105`. Checkout riêng tại `/home/pc/zalo_ai_challenge_claude_review_20261105`; thư mục làm việc gốc và nhánh `main` không bị thay thế.

## Những lỗi đã sửa

| Vấn đề | Ảnh hưởng trước khi sửa | Bản sửa |
| --- | --- | --- |
| Nhãn ASR lấy từ cách đọc TTS | Ví dụ đọc “ét em ét” thì nhãn cũng thành `ÉT EM ÉT`, thay vì học cách viết `SMS` cần cho bước sau | TTS đọc `spoken_text`; ASR học cách viết câu gốc, số được chuyển sang lời đọc. 1.505/7.496 nhãn thay đổi; không phải cả 1.505 khác biệt đều là lỗi nghĩa |
| VieNeu thiếu transcript tham chiếu | API voice cloning cần cả audio và text tham chiếu; chỉ có `ref_audio` có thể làm mọi yêu cầu thất bại | Truyền `speaker_text` vào `ref_text`, từ chối transcript trống. Đối chiếu với [mã nguồn VieNeu](https://github.com/pnnbao97/VieNeu-TTS/blob/main/src/vieneu/base.py); chưa chạy engine TTS thật |
| TTS chưa chia theo model | Cả hai ASR cùng học mọi dữ liệu dù mục tiêu là Zip tiếng Anh, Gip tiếng Việt/domain | Manifest giữ `target_model`; loader lọc theo model. 3.748 câu Zip, 3.748 câu Gip từ API; 790 câu synthetic mới mặc định dành cho Gip |
| Synthetic TTS mới thiếu metadata | Không có `target_model`, không tương thích với bước synthesis/lọc data | Builder xuất đầy đủ metadata, giữ nhãn viết gốc; có cờ đổi model đích và hỗ trợ tạo thư mục output mới |
| Qwen có hai token kết thúc khác nhau | Target học `<\|im_end\|>` = 248046 nhưng config gốc dùng 248044; có thể tiếp tục sinh sau khi hết JSON | Lưu generation stop ids `[248046, 248044]` khi train và merge |
| Tái sử dụng gold chỉ dựa vào từ khoá | Đủ tên/số chưa chứng minh biến thể giữ nguyên yêu cầu | Nguồn `generated` mặc định yêu cầu `manual_decision=pass_text`; muốn dùng bản chưa chấm phải bật cờ riêng |
| Trace cũ thiếu bằng chứng ASR | Bật evidence khi tạo SFT nhưng input cũ vẫn thiếu transcript thô | Khôi phục từ cache ASR đã lưu, theo đúng quy tắc runtime; bỏ cache chuẩn hoá/ghép từ khỏi tập hypothesis |
| Replay chưa mô phỏng repeat guard | Gold có hai lần gọi giống nhau sẽ thiếu bước cảnh báo/xác nhận mà runtime tạo ra | Replay dùng `repeat_check` của runtime; bộ đầy đủ qua kiểm tra trạng thái cuối |
| WER không cải thiện hoặc chạy lại exp dir cũ | Có thể export file best không tồn tại, hoặc vô tình dùng best từ lần chạy trước | Không export khi chưa cải thiện; best cũ được lưu riêng trước vòng train mới |
| So sánh output chứa NaN | `NaN > tolerance` trả về false, có thể lọt kiểm tra parity | Chặn output không hữu hạn trước khi so sánh JIT/merge/ONNX/Qwen; bỏ batch có gradient không hữu hạn |

Có một phép stress bằng feature ngẫu nhiên gây gradient/output không hữu hạn ở Gip sau update; log này được giữ với trạng thái **failed**, không tính là một test thành công. Audio thật có loss và gradient hữu hạn ở cả hai model. Vì vậy cần theo dõi số batch bị bỏ khi train dài; smoke test không thay thế chạy train đầy đủ.

## Kết quả kiểm tra thật trên CPU

| Kiểm tra | GipFormer | ZipFormer-150M |
| --- | ---: | ---: |
| Tham số model nạp được | 69.651.511 | 153.438.074 |
| Nạp checkpoint / dựng lại JIT | Pass | Pass; encoder lệch tối đa 3,10e-6, decoder 0 |
| Loss trên audio thật | 7,090986 | 27,005833 |
| Số projection LoRA có gradient hữu hạn, khác 0 | 144 | 171 |
| Sai lệch encoder sau merge trong bài export | 1,27e-6 | 3,70e-6 |
| ONNX encoder so với PyTorch | 1,88e-6 | 3,81e-6 |
| Vòng CLI train: 2 audio train, 1 validation, 1 epoch | Pass | Pass |
| Nạp ONNX và decode bằng sherpa-onnx | Pass | Pass |

Export dùng adapter nhỏ có trọng số khác 0 được tạo riêng để kiểm tra phép merge. Đây không phải checkpoint đã fine-tune đạt chất lượng. Decode chỉ chạy 3 audio test cho mỗi bản gốc/export. Vòng train nhỏ không cải thiện WER trên 39 từ validation, và đã xác nhận không tạo `lora-best.pt`; file best từ lần trước được lưu riêng đúng cách.

**Qwen:** chạy đầy đủ train → lưu adapter → merge trên bản thu nhỏ cùng kiến trúc Qwen3.5 hybrid, dùng tokenizer thật; 16.131.228 tham số. Logits sau merge lệch tối đa **0,005859375** ở bf16, số tham số giữ nguyên. Không nạp Qwen 4B đầy đủ để train trên CPU. Chưa đo VRAM ở context 16K hoặc thử QLoRA trên GPU.

**Unit test:** 104/104 pass, không skip trong môi trường review. Profile CI nhẹ: 104 test, 13 skip do không có thư viện model, không có lỗi. Không gọi API sinh data trong các bài test này.

## Data đã kiểm tra và dựng xong

| Data | Train / validation / test hoặc số yêu cầu |
| --- | ---: |
| Audio thật: lượt cuối và user history | 3.558 / 649 / 829 |
| Qwen normalizer, từ hai ASR đã cache | 10.674 / 1.947 / 2.487 |
| Agent SFT, sau replay trong devkit thật | **15.892 / 2.985 / 3.243** |
| TTS request API, chỉ seed fit | 7.496 |
| Synthetic mới, chỉ seed fit | 790: 600 paraphrase, 40 thiếu thông tin, 150 rút lại yêu cầu |

- 7.021 quỹ đạo được giữ lại. Cả 790 synthetic replay được; không có quỹ đạo bị loại vì lỗi tool hoặc lệch state trong lần dựng đầy đủ. Hai trace từ benchmark cũ bị thiếu, được bỏ qua và ghi vào summary.
- Audio thật không trùng sample id hoặc đường dẫn audio giữa các split. TTS API chỉ lấy seed fit. Đây là kiểm tra theo id/path, chưa phải đối chiếu nội dung waveform hoặc tách người nói.
- Đã đọc toàn bộ JSONL agent để kiểm tra id duy nhất, assistant JSON hợp lệ, payload và tách sample giữa split; kiểm tra vocab BPE hai model trên nhãn TTS. Các số đo chi tiết ở `data_integrity_final.json`.
- 8.286 TTS request đều có audio/text tham chiếu và không sinh token unknown với BPE hai model. Probe 120 bản ghi agent dài nhất theo kích thước JSON cho độ dài lớn nhất 11.635 token, dưới 16K; đây chưa phải đo số token toàn bộ dataset.
- Đã đọc các câu thiếu thông tin và câu rút lại yêu cầu mới. Kiểm tra từ khoá/số và replay không chứng minh mọi paraphrase đều đúng nghĩa. Chưa xác nhận ngữ nghĩa thủ công toàn bộ 600 paraphrase; chưa có audio TTS để nghe/chấm chất lượng.
- Data agent SFT đầy đủ ở `outputs/agent_sft/` trong checkout review. Bản nén có trong [`artifacts/claude_review_20261105/agent_sft.tar.gz`](../artifacts/claude_review_20261105/agent_sft.tar.gz), gồm JSONL, summary và checksum.

## Những kết luận trong handoff Claude cần đính chính

1. **Zip có checkpoint PyTorch `pretrained.pt`**, ngoài JIT và ONNX: [file phát hành chính thức](https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h/tree/main). Không cần tải thêm ở lần review này vì JIT tại máy dựng lại được.
2. **Gip tại máy đã có `model.pt` và `bpe.model`**, nạp được bằng icefall thật. Checkpoint có head CTC; preset fine-tune hiện tính cả auxiliary CTC khi có head, không chỉ RNN-T.
3. **Không xác nhận được data train của Gip từ bảng benchmark.** [GipFormer](https://github.com/ggroup-ai-lab/gipformer) công bố kết quả đánh giá; đó không phải bằng chứng đã train trên từng corpus. Zip có liệt kê nguồn train trong model card.
4. **xLAM tiếng Việt có tool response.** Mẫu thật của [5CD-AI](https://huggingface.co/datasets/5CD-AI/Vietnamese-Salesforce-xlam-function-calling-Gemini) có role `function-call`, `function-response`. Tool/protocol khác LifeMate nên cần chuyển đổi, nhưng không nên loại bằng kết luận “không có observation”.
5. Đã đọc schema/mẫu [wannaphong](https://huggingface.co/datasets/wannaphong/vietnamese-Locutusque-function-calling-qwen); mẫu đã xem không đủ kết luận cả bộ chỉ trả lời bằng lời. Phần nguồn ngoài chưa được đưa vào train hiện tại.

## Môi trường và cách chạy lại

Đã kiểm tra `/home/pc/v2t`, `/home/pc/venv`, `/home/pc/venv_tsasr` và checkpoint trong workspace gốc. Tái sử dụng package/checkpoint hiện có bằng đường dẫn; không cài đè môi trường v2t. Không có lệnh Docker trên máy nên không cài/chạy Docker.

Môi trường review riêng `.venv-review`: torch 2.10.0+cu128, k2 1.24.4.dev20260306, lhotse 1.33.0, transformers 5.21.0, peft 0.21.2. Chỉ peft được cài thêm vào venv riêng; các package nặng được đọc từ môi trường có sẵn. icefall được checkout riêng đúng commit `9c966762e058bd7ba79d9da71e42fc06b64b703e`.

Khi chạy fine-tune sau này trên máy này, chỉ định `PYTHON_BIN=/home/pc/zalo_ai_challenge_claude_review_20261105/.venv-review/bin/python` và `ICEFALL_DIR=/home/pc/zalo_ai_challenge_claude_review_20261105/.review-deps/icefall`. Python mặc định `/home/pc/venv` có transformers 4.57.1 và chưa có peft; không dùng trực tiếp môi trường đó cho Qwen3.5 LoRA. Các đường dẫn môi trường riêng không được đóng gói lên Git.

Mọi bài chạy model đặt `CUDA_VISIBLE_DEVICES=''`, giới hạn OMP/BLAS/MKL ở 2 thread và chạy ưu tiên thấp. `torch.cuda.is_available()` trả về false trong review. Các lần kiểm tra GPU trước/sau vẫn chỉ có PID **522957**, Python của `/home/pc/venv_tsasr`, dùng **24.956 MiB**; không có process GPU mới từ review.

Chạy lại unit test trên máy hiện tại:

```bash
cd /home/pc/zalo_ai_challenge_claude_review_20261105
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv-review/bin/python -m unittest discover -s tests -v
```

Khôi phục data SFT đã dựng, chạy từ thư mục repo:

```bash
tar -xzf artifacts/claude_review_20261105/agent_sft.tar.gz
sha256sum -c outputs/agent_sft/SHA256SUMS
```

Log/JSON nhẹ để đọc trực tiếp nằm ở [`reports/claude_review_20261105/`](../reports/claude_review_20261105/). Toàn bộ log và artifact smoke cục bộ ở `/home/pc/zalo_ai_challenge/outputs/claude_review_20261105/`; không đưa trọng số smoke lên Git. Khi có lịch train GPU, dùng [FINETUNE.md](FINETUNE.md), theo dõi gradient/validation rồi mới so sánh ASR và agent trên split test.
