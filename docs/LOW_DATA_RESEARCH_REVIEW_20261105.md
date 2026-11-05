# Kiểm chứng nghiên cứu và code: ít audio gốc, nhiều TTS clone giọng

Ngày 05/11/2026. Đã pull tới **`0d0a2a4`** của `claude/loving-goldberg-fnkqij`, rồi kiểm tra trên checkout review riêng. Phần nghiên cứu dùng nguồn gốc, không chỉ snippet tìm kiếm; phần thực thi dùng CPU, CUDA bị tắt. Không có kết quả train full hoặc cải thiện calling sau fine-tune trong báo cáo này.

## Kết luận để quyết định

**Hướng làm có cơ sở, nhưng các ngưỡng số trong handoff chưa đủ để chốt cấu hình.** Trộn TTS theo epoch, giữ audio gốc, tăng biến thiên âm thanh và thử thích nghi phần ngôn ngữ của RNN-T đều đáng thử. Chưa có bằng chứng rằng với Gip/Zip tiếng Việt, TTS phải dưới 2 lần audio gốc, 40 giọng luôn đủ, hoặc joint-only là phương án tốt nhất.

Ưu tiên hiện tại: **QC đúng các từ cần sửa → thử hotwords/beam trên validation → mốc chỉ audio gốc → ablation TTS và joint-only**. Chọn theo khả năng giữ đúng thông tin cần cho calling, không chỉ WER. Zip tiếp tục nhắm tiếng Anh/mã/tên ngoại; Gip nhắm tiếng Việt và domain yếu.

## 1. Đã đo được gì ở dữ liệu của repo

| Split | Số audio | Thời lượng đo từ WAV |
| --- | ---: | ---: |
| Fit | 3.558 | **4,805 giờ** |
| Validation | 649 | 0,918 giờ |
| Test | 829 | 1,106 giờ |

“Audio gốc/thật” ở đây là audio có nhãn từ tập train, phân biệt với audio sắp sinh thêm. Chưa xác nhận toàn bộ audio gốc là người thật thu âm, cũng chưa có nhãn speaker identity để đếm người nói.

Kho trong nhánh Git có **8.286 yêu cầu TTS dạng text**: 3.748 cho Zip, 4.538 cho Gip (đã tính 790 câu synthetic mới). Chưa sinh/chấm audio TTS thật trong lần review này. Chưa thể chuyển số yêu cầu thành số giờ TTS hay kết luận TTS đang chiếm bao nhiêu phần trăm thời lượng.

`real-repeat=2` không tạo thêm giờ dữ liệu độc lập: vẫn chỉ có 4,805 giờ gốc. Bản speed 0,9× làm tăng thời lượng được phát lại trong epoch. Các con số “61% / 65% thật” trước đây là **tỉ lệ lượt câu sau lặp**, không phải tỉ lệ giờ hay tỉ lệ thông tin mới.

## 2. Đối chiếu các kết luận chính với nguồn gốc

Đây là kiểm chứng có trọng tâm, không phải systematic review hoặc meta-analysis. Không gộp phần trăm giảm WER của ngôn ngữ/kiến trúc khác nhau thành dự đoán cho repo. CSV [đối chiếu claim/nguồn](../reports/low_data_review_20261105/claim_source_review.csv) phân biệt đã đọc thực nghiệm, chỉ kiểm tra abstract và giả thuyết chuyển giao.

| Kết luận | Bằng chứng đã đọc | Kết luận áp vào repo |
| --- | --- | --- |
| TTS giúp cả khi cực ít audio thật | [Bartelds et al., ACL 2023](https://aclanthology.org/2023.acl-long.42/): TTS trên Gronings giảm WER tương đối tới 25,5% so với mốc 24 phút | Có cơ sở thử; không dự đoán được mức tăng với 4,805 giờ tiếng Việt và ASR đã pretrain lớn |
| TTS quá 2 lần thật thì ít lợi | [Hao et al., LREC 2026](https://aclanthology.org/2026.lrec-1.442/), §3.3/Fig. 3: với 1 giờ Frisian, tập TTS cố định tốt nhất ở **7:1**; chế độ dùng câu mới tốt nhất ở **2:1**. Cùng XLS-R/XTTS và 2.000 bước | 2:1 là điểm thực nghiệm trong một chế độ, không phải trần chung. Bài tự nêu hạn chế chỉ một ASR backbone và một TTS architecture |
| 1:1 tốt cả trong/ngoài miền | Abstract chính thức của [SpeD 2025](https://xplorestaging.ieee.org/document/11252042), DOI `10.1109/SpeD67700.2025.11252042`, có kết luận này | Xác nhận được abstract; chưa kiểm tra đầy đủ bảng/phương pháp. Dùng 1:1 làm ứng viên, không làm quy luật |
| Khoảng 40 giọng là đủ | [Goodarzi et al., 2026](https://arxiv.org/abs/2611.03050): Whisper-small, OmniVoice, trẻ em nói tiếng Anh; plateau khoảng 40 cho chuyển miền | Không suy ra Gip/Zip đủ 40 giọng; 753 clip tham chiếu không chứng minh có 753 người nói. Đây là nguồn rất mới, chưa kiểm tra tái lập độc lập |
| Text quan trọng hơn speaker | [Yang et al., 2024](https://arxiv.org/abs/2410.16726), Tables I–IV, có kết quả này với CosyVoice và Conformer CTC/attention; các tập 200–3.700 giờ. Bài trẻ em ở trên lại nhấn mạnh đa dạng âm học trên cùng transcript | Cả text lẫn âm học cần được đo. Không nên bỏ đa dạng giọng/điều kiện thu âm dựa trên một ngưỡng duy nhất |
| TTS nên chỉ chỉnh phần prediction/joint | [Li et al., Interspeech 2020](https://arxiv.org/abs/2007.15188), §4.3/Table 4: chỉnh toàn model bằng TTS làm xấu; prediction + joint cải thiện. Họ dùng 1,5 triệu câu, khoảng 3.000 giờ TTS sau augmentation, RNN-T gốc train 65.000 giờ | Hỗ trợ hướng đóng băng encoder. **Code repo chỉ LoRA joint**, không cập nhật prediction network, nên không tái lập đúng phương pháp của bài |
| Chọn hard seed rồi LLM + clone giọng đúng hướng | [Hard-Synth](https://arxiv.org/abs/2411.13159) có chọn hard prompt, sinh text/audio và lọc | Hướng phù hợp; chọn câu ASR sai trong repo chưa tương đương toàn bộ thuật toán chọn prompt của bài |
| Lọc TTS giúp giảm lệch | [Synt++ / Hu et al.](https://machinelearning.apple.com/research/utilizing-imperfect-synthetic) báo lợi ích từ rejection sampling và thống kê batch norm tách riêng | QC bằng WER hai ASR là biện pháp khác. Không được coi hai ASR đồng thuận là ground truth hoặc gọi QC hiện tại là tái lập Synt++ |
| TTS còn khoảng cách với speech gốc | [Rosenberg et al.](https://arxiv.org/abs/1909.11699), [Hilmes et al.](https://arxiv.org/abs/2407.17997) cho thấy khoảng cách và độ nhạy theo architecture | Cần validation audio gốc; không suy ra mọi TTS mới đều kém như nhau hoặc có trần tỉ lệ cố định |

Một nguồn bổ sung về nhiều ngôn ngữ là [Synthetic Voice Data for ASR in African Languages](https://aclanthology.org/2025.lowresnlp-1.16/): Chichewa hưởng lợi ở real:TTS 1:2, trong khi Dholuo không nhất quán giữa tập chấm. Đây là bằng chứng thêm cho việc chọn tỉ lệ theo dữ liệu thực tế, không phải công thức riêng cho tiếng Việt.

### Learning rate 0,0045 có chắc là “bug” không?

**Chưa chứng minh được trong ASR.** Trong [recipe icefall đúng commit đang dùng](https://github.com/k2-fsa/icefall/blob/9c966762e058bd7ba79d9da71e42fc06b64b703e/egs/librispeech/ASR/zipformer_lora/finetune.py), parser để 0,045 nhưng hai usage example ở đầu file vẫn ghi 0,0045. Vì vậy không thể chỉ nhìn parser rồi gọi mọi cấu hình 0,0045 là sai.

Đã lặp lại bài fit ma trận bằng **ScaledAdam thật của icefall**, 720 bước, cùng target/khởi tạo ở hai LR:

| LR | RMS của B sau train | Sai số cập nhật còn lại |
| --- | ---: | ---: |
| 0,0045 | 4,87e-5 | 97,97% |
| 0,045 | 2,93e-4 | 4,92% |

Thí nghiệm này ủng hộ giả thuyết LR thấp làm adapter khởi động chậm. Nó chưa chứng minh LR tối ưu cho ASR, chưa dùng loss RNN-T, và không loại trừ nhãn sai, thiếu đa dạng âm học, scheduler hoặc lượng data là nguyên nhân không cải thiện. Giữ 0,045 làm mặc định hiện tại, nhưng chọn bằng validation. Các LR thấp hơn vẫn là ứng viên ablation.

### Phần Qwen trong báo cáo cũ

- [FireAct](https://arxiv.org/abs/2310.05915) có 500 quỹ đạo và tăng 77% **tương đối**: Llama-2-7B trên HotpotQA tăng EM từ 14,8 lên 26,2, tức +11,4 điểm. Đây không phải +77 điểm và không chứng minh 7.000 quỹ đạo LifeMate là đủ; nhiều quỹ đạo của repo là các input ASR khác nhau của cùng task.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) ủng hộ LoRA có MLP và việc tìm LR; thí nghiệm batch 32 tốt hơn batch lớn trong các thiết lập đã xét. Không có bảo đảm batch phải dưới 32 cho mọi Qwen3.5 hybrid hoặc payload 16K.
- [AgentTuning](https://arxiv.org/abs/2310.12823), [Agent-FLAN](https://arxiv.org/abs/2403.12881), [ReST-EM](https://arxiv.org/abs/2312.06585), [SDFT](https://arxiv.org/abs/2402.13669), [ETO](https://arxiv.org/abs/2403.02502), [ToolRL](https://arxiv.org/abs/2504.13958) đều có nguồn gốc xác nhận được. Chúng ủng hộ hướng data chất lượng, verifier và self-training; không xác lập hiệu quả cho task này nếu chưa ablation.
- `self:` cần tách **state match** và **chuỗi calling đúng**. Một đường gọi khác đạt cùng state chưa chắc phù hợp cách chấm chính thức. Giữ bộ đếm khác gold và chưa mặc định đưa quỹ đạo thay thế vào tập chính.
- NEFTune, rsLoRA, early stopping và scale merge đã qua test cơ chế trên Qwen thu nhỏ, chưa chứng minh tăng điểm Qwen 4B. Loss validation không thay thế benchmark calling.

## 3. Vấn đề QC quan trọng cho data từ khó

Chạy chính quy tắc QC mới trên **5.036 audio gốc đã có transcript ASR**, không gọi lại model:

| Kết quả | Tỉ lệ |
| --- | ---: |
| Hai model cùng sai và output giống hệt nhau | **11,72%** |
| Bị quy tắc gắn `tts_misread` | **6,18%** |
| Bị gắn `tts_failed` | 1,85% |

Đây không phải đo tỉ lệ TTS sai. Nó cho thấy hai ASR có lỗi tương quan và bộ lọc tự động cũng loại audio gốc có nhãn. Với tên thuốc, mã và từ ngoại đang khó cho cả hai model, lọc đồng thuận có thể loại đúng phần data cần học.

Đã bổ sung `--review-csv`: nghe các ca khó trong `listen_sample.csv`, điền `nghe_dung_nhan` bằng `đúng`/`sai` hoặc `1`/`0`, rồi chạy QC lại với CSV đó. Audio đúng nhãn được giữ dù hai ASR sai. Audio hash và nhãn phải khớp lần nghe; quyết định cũ không áp vào file audio/nhãn đã thay đổi. File quyết định được giữ nguyên; danh sách chờ nghe tiếp nằm ở `listen_pending.csv`.

```bash
# Chạy trong môi trường ASR đã có, trên audio đã sinh:
CUDA_VISIBLE_DEVICES='' /path/to/python qc_tts_asr.py \
  --manifest outputs/asr_finetune/tts/train.jsonl \
  --model gipformer=models/gipformer1.5-68M-rnnt \
  --model zipformer=models/zipformer-150M-6000h \
  --review-csv outputs/asr_finetune/tts/train_qc/listen_sample.csv
```

Cache đã có fingerprint audio/model được dùng lại; không cần decode lại chỉ để đổi threshold hoặc áp quyết định nghe. Test của chức năng manual dùng fixture, chưa có người chấm TTS thật trong phiên này.

## 4. Code đã sửa sau khi pull

| Chỗ | Thay đổi và lý do |
| --- | --- |
| `asr_lora/data.py` | Thay lấy mẫu TTS độc lập mỗi epoch bằng xoay vòng permutation có seed. Ở chế độ đếm, pool được phủ trong các lượt kế tiếp; ở chế độ thời lượng, xoay vòng các nhóm vừa budget và báo câu quá dài |
| `--tts-ratio-unit seconds` | Giới hạn bằng số giây sau speed perturb; budget bảo thủ theo tốc độ chậm nhất. Log cả số câu, giây và real share theo thời lượng. Train CLI trực tiếp giữ `utterances` để tương thích; `asr2` mặc định `seconds` |
| Cổng dữ liệu TTS | Kiểm tra tất cả dòng manifest. Trước đây dòng đầu là audio thật có thể che TTS chưa QC ở các dòng sau |
| Nhãn TTS | Bỏ fuzzy character similarity để suy cách viết Latin: phản ví dụ “Linh” bị gán thành “Lan”, “Panadol” thành “Paracetamol”. Giữ spell-out và lexicon đọc→viết tường minh, giữ từ phụ như CỘNG/TRÊN; tên/thuốc lạ chưa có ánh xạ không được tự đoán bằng độ giống chữ |
| `run_finetune.sh asr2` | Yêu cầu manifest TTS đã QC không rỗng trước khi train. Nếu encoder stage 1 không hơn base, vẫn thử joint stage 2 từ base thay vì kết thúc sớm |
| `qc_tts_asr.py` | Cache theo hash audio và fingerprint model, không chỉ utterance id. Audio sinh lại hoặc model đổi sẽ được nghe lại; thêm quyết định nghe có kiểm tra audio/nhãn |
| `export_asr_lora.py` | So parity đủ encoder projection, decoder projection và logits joint, cả trước/sau merge và qua ONNX. Kiểm tra encoder đơn lẻ trước đây chưa đủ cho adapter có sửa joint |
| Validation | Từ chối loss/output NaN/Inf và validation rỗng, không báo WER từ output không hợp lệ |
| Tài liệu | Sửa các kết luận tuyệt đối về hotwords, LR, tỉ lệ 2:1 và số speaker; tách bằng chứng, giả thuyết và kết quả chạy tại máy |

Tạo lại 7.496 TTS request API: **91 nhãn khác snapshot sau pull**, vẫn không có token unknown với BPE cả hai model. Không phải cả 91 đều là lỗi thuốc/tên: có ca giữ lại từ CỘNG/TRÊN hoặc cách đọc chưa xác nhận. Lexicon tường minh ở `asr_lora/tts_pronunciations.json` giữ các cách viết Latin đã đối chiếu trên text; nó không thay thế nghe audio TTS. Cách đọc chưa đúng nên sửa trước khi tổng hợp, không ép nhãn thành từ gần giống.

Giới hạn số giây kiểm soát **mức phơi audio**, không bảo đảm TTS đóng góp đúng cùng tỉ lệ gradient. Xoay vòng chỉ phủ các câu vừa budget và cần đủ số epoch; `--fixed-tts-subset` chủ động giữ một nhóm cố định.

## 5. Các phép thử tại máy

### Gip/Zip: hai stage với model và icefall thật

Đã chạy cả hai model qua train encoder nhỏ → export → nạp lại → train joint nhỏ với real repeat, speed và reverb → export → decode bằng sherpa-onnx. Stage 2 có 3 projection trainable; so state dict trước/sau cho thấy chỉ trọng số `joiner.*` thay đổi, encoder và decoder giữ nguyên.

**Fixture TTS dùng audio gốc được gắn source TTS để thử nhánh code**, không phải audio mới tổng hợp. Mỗi stage chỉ 2 audio train và 1 validation. Script cố ý export checkpoint epoch để kiểm tra reload ngay cả khi không cải thiện. Không dùng các model smoke làm model production hay kết luận về độ chính xác.

So parity logits joint sau export ONNX lệch tối đa khoảng **3,82e-5 (Gip)** và **2,29e-5 (Zip)** trong probe, nằm trong tolerance. Tham số model sau merge giữ nguyên 69.651.511 và 153.438.074. Torchaudio 2.10.0 có sẵn; speed perturb và vang phòng đều nạp được audio thật.

### Hotwords: pilot 64 audio validation

Chọn seed 42, 32 câu có từ ngoại/mã theo heuristic và 32 câu không có; đây là mẫu phân tầng, **không đại diện tỉ lệ xuất hiện trong toàn validation**. Có 41 lượt từ ngoại/mã theo metric, giải mã beam=4, hotwords score=1,5. Không tune score theo mẫu này.

| Model / decode | Recall từ ngoại/mã | False alarm / 100 câu | Câu đúng hoàn toàn được sửa / bị làm hỏng so greedy |
| --- | ---: | ---: | ---: |
| Gip greedy | 56,10% | 25,00 | — |
| Gip beam | 56,10% | 25,00 | 0 / 1 |
| Gip beam + hotwords | **60,98%** | 23,44 | 1 / 1 |
| Zip greedy | 53,66% | 23,44 | — |
| Zip beam | 58,54% | 21,88 | 2 / 0 |
| Zip beam + hotwords | **68,29%** | 20,31 | 3 / 0 |

Zip có tín hiệu tốt để đo đầy đủ trên validation; Gip có lỗi mới nên không thể gọi hotwords là không có rủi ro. Các chỉ số trên không phải accuracy calling; recall heuristic chưa bao phủ hết thuốc, phủ định, số lượng hoặc điều kiện. CSV đối chiếu từng câu và JSONL dự đoán được lưu cùng log.

### Qwen và data SFT

Qwen3.5 hybrid thu nhỏ, tokenizer thật: train với NEFTune=5, rsLoRA, validation/early-stopping callback rồi merge scale 0,5 đều chạy được trên CPU. Logits lệch tối đa 0,0078125 ở bf16. Không đo hiệu quả Qwen 4B hoặc VRAM.

Dựng SFT đầy đủ sau dedup: **15.767 train / 2.937 validation / 3.176 test**, từ 6.923 quỹ đạo; bỏ 98 quỹ đạo trùng, thiếu 2 trace cũ. Đây là 1.545 task gốc được mở rộng bằng nhiều loại input, không phải 6.923 nhiệm vụ độc lập. Data mới ở `outputs/low_data_review_20261105/agent_sft/`; không ghi đè snapshot SFT trước đây.

Unit test: **145 pass** trong môi trường review; profile CI nhẹ có **27 skip** do thiếu thư viện model, không lỗi. Chưa chạy GPU, TTS engine thật, train full, hoặc benchmark agent sau fine-tune.

## 6. Thí nghiệm nên chạy tiếp khi có lịch GPU

| Bước | Cố định gì | So sánh gì | Quyết định dựa vào |
| --- | --- | --- | --- |
| 1. Decode | Model gốc, split hiện tại | Greedy / beam / vài score hotword, ưu tiên Zip | Từ cần cho calling sửa được, chèn sai và câu mới bị hỏng |
| 2. Mốc real-only | Cùng data/audio gốc, rank và seed | Một sweep LR nhỏ, rồi giữ LR đã chọn | Validation, update norm/gradient, không suy từ toy loss |
| 3. TTS | Cùng LR/rank/steps hoặc báo rõ compute, TTS đã QC | Real-only; encoder + mix; joint-only từ base/encoder real; duration ratio 0,5 / 1 / 2 | Audio gốc validation và giữ đúng entity/slot; không coi 2 là trần |
| 4. Pipeline | Cùng Qwen/prompt/flags, cache mới đúng model | ASR → normalizer → calling; chọn xong mới chấm test | Tách lỗi ASR, normalize và calling; state và exact calls nếu có |

Cho **Zip**: text có tiếng Anh, thương hiệu, chữ viết tắt và tên ngoại thật sự cần trong task; có cách đọc kiểm tra được. Cho **Gip**: từ tiếng Việt hay lẫn làm đổi việc/điều kiện, tên thuốc/địa danh/domain yếu. Với từ khó, thử nhiều ngữ cảnh và một số giọng/điều kiện âm học khác nhau; không nhân 10 bản cùng một nội dung chỉ để tăng số giờ.

Ngoài WER, lập nhóm metric cho: tên thuốc/người/mã; số tiền/giờ/đơn vị; động từ lệnh; phủ định; điều kiện; thứ tự tác vụ. Đếm **sửa được thông tin nào, làm hỏng thông tin nào**, rồi đo calling với cùng Qwen. Nếu TTS đúng lời đọc nhưng đã đổi nghĩa so với câu viết gốc, không tái sử dụng gold calling của câu cũ.

`timestamp_word` hiện được hiệu chỉnh trên greedy và từ chối beam/hotwords. Kết quả pilot ASR đơn chưa cắm được trực tiếp vào bộ ghép từng từ này. Khi đổi decoder hoặc model ASR, cần đo/hiệu chỉnh lại selector và không dùng cache cũ.

Với thống kê, McNemar áp dụng cho đúng/sai **theo task** khi so agent. Không áp ngưỡng “cần tăng 5 điểm” cho mọi tình huống hay cho keyword recall; mức ý nghĩa phụ thuộc số cặp sửa được/làm hỏng. Audio history và lượt cuối cùng task có tương quan, nên các khoảng tin cậy ASR cần tính theo nhóm task.

## 7. Log, nguồn và cách chạy lại

- Log/JSON/CSV nhẹ được chia sẻ tại [`reports/low_data_review_20261105/`](../reports/low_data_review_20261105/).
- Artifact cục bộ, gồm PDF gốc và model smoke: `/home/pc/zalo_ai_challenge/outputs/low_data_review_20261105/`. PDF có hash tải xuống; không đưa PDF, trọng số smoke hoặc audio lên Git.
- Môi trường riêng: `.venv-review`; icefall riêng: `.review-deps/icefall`. Không cài đè môi trường `/home/pc/v2t`, `/home/pc/venv` hoặc `/home/pc/venv_tsasr`.
- GPU trước/sau vẫn có tiến trình venv_tsasr PID 522957, 24.956 MiB, không có process GPU mới từ review.

```bash
cd /home/pc/zalo_ai_challenge_claude_review_20261105
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv-review/bin/python -m unittest discover -s tests -v

# Smoke hai stage; tạo model riêng trong outputs, không train full:
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv-review/bin/python scripts/smoke_low_data_cpu.py
```

Chưa có phép thử nào ở đây đủ để khẳng định “hai stage tốt nhất”, “TTS cải thiện bao nhiêu điểm”, hoặc “train đủ VRAM”. Phần đã xác nhận là bằng chứng đúng phạm vi, cơ chế code chạy được và các lỗi cụ thể đã được sửa.
