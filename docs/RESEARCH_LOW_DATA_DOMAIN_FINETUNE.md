# Train ASR và Qwen với ít dữ liệu, một domain: nghiên cứu và công thức cho LifeMate

> **Đã kiểm chứng lại trên máy, 05/11/2026:** xem [báo cáo đối chiếu nguồn và test thật](LOW_DATA_RESEARCH_REVIEW_20261105.md). Đã đọc PDF gốc cho các kết luận chính. Tỉ lệ 2:1, ngưỡng 40 giọng và LR 0.045 đều phụ thuộc thiết lập; không phải quy luật chung. Hotwords có thể làm hỏng câu dù không đổi trọng số. Code bổ sung giới hạn TTS theo số giây, xoay vòng pool, cache QC theo nội dung và kiểm tra export đủ ba nhánh transducer.

Câu hỏi: hai ASR (GipFormer 68M RNN-T, ZipFormer-150M CR-CTC) và Qwen3.5-4B đều đã học rất nhiều dữ liệu. Ta chỉ có
vài nghìn câu trong miền thi. Làm sao để model học được miền thi mà không lệch, không quên và không học vẹt?

Tài liệu gồm ba phần: bằng chứng từ các công trình đã công bố, cách áp vào số liệu của repo, và các tuỳ chọn đã cài.
Các tuỳ chọn mới mặc định tắt. Có ba mặc định thay đổi:

- lr LoRA ASR, sửa lỗi ở mục 3.2;
- tập TTS được lấy mới mỗi epoch, mục 3.6;
- bỏ quỹ đạo trùng khi build SFT, mục 4.1.

Hạn chế về nguồn được ghi ở mục 8.

## 0. Tóm tắt

1. **Chưa train ASR vội; thử giải mã có thiên vị (contextual biasing) trước.** Lỗi khó học nhất bằng ít data là từ
   hiếm: mã, thương hiệu, từ mượn (SJC, ZALO, SMS, MEETING…). Cách sửa là `modified_beam_search` + hotwords của
   sherpa-onnx: không đổi trọng số, nhưng vẫn có thể làm sai các câu khác do đổi điểm giải mã.
   - Danh sách lấy từ split fit phủ được 73/136 (Gip) và 66/126 (Zip) lần nghe sai loại từ này trên validation.
     Đây là độ phủ của danh sách, chưa phải tỉ lệ sửa được hoặc một chặn trên cho hiệu quả decode.
   - Cách chạy: `./run_finetune.sh hotwords`.
2. **Nếu train ASR thì dùng LoRA nhỏ, chọn checkpoint trên validation (chỉ audio thật).** Sau đó nội suy trọng số giữa
   model gốc và model đã train (`--scale`, kiểu WiSE-FT), chọn hệ số trên validation, chỉ chấm test một lần.
   - Khi audio thật chỉ vài giờ và phần lớn là TTS (mục 3.6): mỗi epoch cho TTS khoảng 1–2 lần số câu thật, lấy tập TTS
     mới mỗi epoch để dùng hết kho. Tăng biến thiên âm học cho TTS.
   - Một nhánh cần so sánh: `./run_finetune.sh asr2`. Encoder chỉ cập nhật từ audio gốc; TTS dạy joint network. Li et al. cập nhật cả prediction + joint, nên chưa phải cùng phương pháp.
   - LR mặc định nay là 0.045 theo parser recipe `zipformer_lora`; usage của chính recipe cũng có 0.0045. Mô phỏng ma trận ủng hộ thử LR cao hơn, chưa chứng minh nguyên nhân model ASR không cải thiện.
3. **Đo đúng chỗ ít data tác động.** Ngoài WER, cần xem:
   - recall và false alarm của từ ngoại/mã;
   - số câu "sửa được" và số câu "làm hỏng";
   - độ trôi trên public_test không nhãn.
4. **Qwen: dữ liệu quan trọng hơn thủ thuật.** Vài trăm đến vài nghìn quỹ đạo đúng là đủ để tăng mạnh
   (FireAct, LIMA). Cấu hình nên dùng:
   - LoRA trên mọi lớp tuyến tính, lr khoảng 1e-4, batch hiệu dụng dưới 32;
   - dừng sớm theo validation.
5. **Giảm trôi phân phối cho Qwen bằng dữ liệu "của chính nó".** Lấy mẫu nhiều lượt trên train, giữ quỹ đạo devkit xác
   nhận đúng (rejection sampling / RFT / ReST-EM, cùng tinh thần SDFT). Nguồn `self:DIR` mới trong
   `build_agent_sft.py` làm việc này.
6. **Kiểm định trước khi tin.** Test chỉ có 248 mẫu. Với 10–20 mẫu bị làm hỏng, mức tăng ròng phải từ khoảng 5–6,5
   điểm mới có ý nghĩa thống kê (McNemar, p < 0,05). `compare_benchmark_runs.py` nay in `mcnemar_exact_p`.

## 1. "Ít data" ở repo này cụ thể là bao nhiêu

| Phần | Số liệu (đã có trong repo) |
| --- | --- |
| ASR, audio gốc | fit / validation / test: 3.558 / 649 / 829 câu, gồm cả history; đo bằng header WAV: **4,805 / 0,918 / 1,106 giờ** |
| ASR, TTS | 7.496 yêu cầu (3.748 mỗi model) và 790 câu agent viết tay; chỉ dùng sau QC (`*_qc.jsonl`) |
| WER gốc trên 5.036 audio train | Gip 15,24%, Zip 14,0% |
| Agent | 1.545 mẫu: fit / validation / test là 1.066 / 231 / 248. State match test hiện là 26,6–29,0% tuỳ cấu hình ASR |
| SFT agent (bản user build) | 15.892 bản ghi train / 7.021 quỹ đạo; prompt khoảng 11,6k token, JSON đích ngắn |

Lỗi ASR, theo `outputs/asr_error_audit` và phép đo mới trong `build_hotwords.py`:

- **Từ không thể là âm tiết tiếng Việt** (mã, thương hiệu, từ mượn, tên nước ngoài) có mặt trong 26% câu fit (929/3.558).
  - Khoảng 15% câu có ít nhất một từ như vậy bị nghe sai: Gip 547 câu, Zip 517 câu.
  - Ví dụ: SJC sai 55/74 lần (Gip) và 63/74 lần (Zip); ZALO 22 và 27/63; CHECK 21 và 16/31; MEETING 13 và 5/15.
- **Nhầm động từ lệnh:** `nhắn → nhấn`, `họp → học`, `dời → giờ`, `đừng → được`, `đặt → đọc`. Đây là từ thông dụng, nên
  hotword không hợp. LoRA trên audio thật và bước chuẩn hoá theo ngữ cảnh phù hợp hơn.
- **Khác chính tả hoặc phương ngữ:** `hông/không`, `hoà/hòa`, `dùm/giùm`. Loại này làm tăng WER nhưng ít đổi hành động.
  Đừng để LoRA "học" chính tả của nhãn mà quên nghe.

Hai rủi ro cần tách riêng:

- **Lệch hoặc quên:** model tệ đi ở phần không có trong data mới, như giọng khác, câu ngoài mẫu hay từ thông dụng.
- **Học vẹt:** model nhớ câu và giá trị tham số của train, không tổng quát sang test.

Với ASR, cả hai đều nguy hiểm. Với Qwen, gần như chỉ dùng cho tác vụ này, nên học vẹt nguy hiểm hơn quên kiến thức
chung. Tuy vậy, Qwen vẫn cần giữ năng lực đọc tiếng Việt, số, giờ và điều kiện.

## 2. Bằng chứng chung: fine-tune ít data có lệch thật không

- **LoRA vẫn quên, chỉ quên ít hơn.**
  - Lượng quên tăng theo luật luỹ thừa của số tham số được train và số bước cập nhật. Dừng sớm hay train ít tham số
    hơn đều không tránh được quên ([Kalajdzievski 2024](https://arxiv.org/abs/2401.05605v1)).
  - "LoRA Learns Less and Forgets Less": LoRA học miền mới kém hơn full fine-tune nhưng giữ năng lực ngoài miền tốt
    hơn. Kết quả nhạy nhất với learning rate, rồi tới chọn module, sau cùng là rank; nên gắn vào mọi module, nhất là
    MLP ([Biderman et al. 2024](https://arxiv.org/html/2405.09673v2)).
- **ASR transducer quên mạnh nếu fine-tune thẳng.** Báo cáo "Damage control" của NVIDIA: fine-tune thẳng một
  Conformer-transducer 120M làm WER miền gốc tăng từ khoảng 5% lên 30–50%. Họ đề xuất adapter có điều chuẩn ở encoder,
  prediction network và joint network, cộng chiến lược train giới hạn
  ([Majumdar et al. 2022](https://export.arxiv.org/pdf/2210.03255),
  [trang NVIDIA](https://research.nvidia.com/publication/2022-10_damage-control-during-domain-adaptation-transducer-based-automatic-speech)).
  Model E2E quên cả khi thích nghi trong cùng một ngôn ngữ
  ([Vander Eeckt & Van hamme](https://lirias.kuleuven.be/retrieve/666478)).
- **Trộn một ít dữ liệu cũ giữ được năng lực cũ.** Một nghiên cứu ASR tiếng Đức thêm một phần nhỏ dữ liệu miền gốc: WER
  miền mới vẫn dưới 5% mà năng lực chung ổn định. Nghiên cứu này cũng dẫn kết quả trước đó rằng với RNN-T, phần lớn lợi
  ích đến từ việc chỉnh joint layer và lớp encoder đầu
  ([Uni Hamburg](https://www2.informatik.uni-hamburg.de/wtm/publications/2023/PW23/978-3-031-44195-0_40.pdf)).
  CL-MASR dùng 10% dữ liệu cũ trộn ở mức dataset ([CL-MASR](https://arxiv.org/pdf/2310.16931)).
- **Nội suy trọng số lấy lại phần lớn độ bền mà vẫn giữ lợi ích.** WiSE-FT trộn tuyến tính trọng số gốc và trọng số đã
  fine-tune: θ = (1−α)θ₀ + αθ₁. Kết quả là tăng vài điểm (khoảng 2–10 tuỳ tập) ngoài phân phối mà vẫn giữ độ chính xác
  trong phân phối, và không tốn thêm chi phí train hay suy luận ([Wortsman et al.](https://arxiv.org/pdf/2109.01903)). Với LoRA, cách này đúng bằng
  nhân phần cập nhật BA với α. Có các nghiên cứu merge hoặc nội suy model cho ASR
  ([2603.05354](https://arxiv.org/abs/2603.05354), [2505.20477](https://arxiv.org/html/2505.20477v1),
  [2502.17380](https://arxiv.org/pdf/2502.17380)).

## 3. ASR: công thức theo thứ tự ưu tiên

### 3.1 Bậc 0, không train: beam search + hotwords

**Vì sao làm trước.** Sửa lúc decode không đụng trọng số, nên không có rủi ro quên. Loại lỗi mà nó nhắm tới (từ hiếm
trong miền) cũng là loại ít data dạy kém nhất. Nó cũng không chặn LoRA: sau khi train vẫn dùng chung hotword được.

**Sherpa-onnx làm gì.** Hotword chỉ có tác dụng với `modified_beam_search`, cần `bpe.vocab` để tách token, và điểm
thưởng mặc định là 1,5 ([tài liệu hotwords](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/hotwords/index.rst)).

Tôi đã đọc `sherpa-onnx/csrc/utils.cc` và `scripts/export_bpe_vocab.py` của sherpa-onnx:

- mỗi dòng là một cụm từ, tách token theo từng từ bằng BPE;
- có thể thêm `:điểm` riêng ở cuối dòng;
- sherpa-onnx không tự đổi hoa thường, nên hotword phải viết hoa giống output của model.

Bản 2.0.1 còn có tham số `lm`, `lm_scale`, `lodr_fst`, `lodr_scale` để kết hợp LM ngoài.

**Đã cài:**

| Thành phần | Việc |
| --- | --- |
| `build_hotwords.py` | Lấy từ ứng viên trong nhãn split **fit**: từ có ít nhất 2 chữ cái và không khớp cấu trúc âm tiết tiếng Việt (phụ âm đầu + vần + âm cuối, sau khi bỏ dấu thanh). Mỗi model giữ từ xuất hiện ≥ 3 câu và bị chính model đó nghe sai ≥ 30% (theo transcript greedy đã lưu). Không đọc validation/test |
| `outputs/asr_hotwords/` | Danh sách đã tạo: Gip 81 từ, Zip 63 từ. Trên fit, phủ 456/738 (Gip) và 400/644 (Zip) lần nghe sai |
| `GIPFORMER_DECODING`, `GIPFORMER_HOTWORDS`, `GIPFORMER_HOTWORDS_SCORE`, `GIPFORMER_BEAM` | Biến môi trường cho pipeline; tương tự với `ZIPFORMER_*`. Không đặt thì vẫn là greedy như cũ |
| Khoá cache và identity benchmark | Có thêm cấu hình decode và **hash nội dung** danh sách hotword, nên không dùng nhầm transcript greedy cũ. Cấu hình mặc định giữ nguyên khoá cache cũ. `timestamp_word` từ chối chạy nếu bật beam hoặc hotword, vì bộ chọn của nó hiệu chỉnh trên greedy |
| `eval_asr_onnx.py` | Mỗi `--model` có thể kèm `@decoding=…,hotwords=…,hotwords_score=…`. Báo cáo thêm `foreign_words` (recall và false alarm trên 100 câu) |
| `./run_finetune.sh hotwords` | Tạo danh sách, rồi so greedy, beam và beam + hotword trên validation cho cả hai model |

**Lưu ý:**

- **Chọn điểm thưởng trên validation.** Thử khoảng 1,0–2,5 và xem cả false alarm lẫn `broken_utterances`, không chỉ WER.
- **Danh sách có tên người và thương hiệu hư cấu** (SARAH, NAKAMURA, CINEVIỆT, MINHPOINT…). Chúng lặp lại vì đến từ
  cùng môi trường devkit. Trên test có SARAH 3 lần, TANAKA 2 lần, còn GRACE 0 lần. Nếu false alarm cao, bỏ nhóm tên
  hoặc tăng `--min-count`.
- **Không thêm từ thông dụng** như NHẮN, HỌP: boost những từ này sẽ chèn sai khắp nơi.
- Hướng mở rộng: hotword động theo từng mẫu (tên trong history hoặc context). Sherpa-onnx cho truyền hotword khi tạo
  stream. Hướng này chưa cài.

**LM miền thi kết hợp lúc decode.** Ta có nhiều text trong miền: nhãn fit, câu diễn đạt lại và câu viết tay. Text-only
adaptation là con đường có bằng chứng:

- ILME tốt hơn density ratio và shallow fusion khi chỉ có text miền đích
  ([Meng et al.](https://ar5iv.labs.arxiv.org/html/2011.01991)).
- LODR cho rằng RNN-T chỉ học LM bậc thấp, nên dùng LM bậc thấp làm mẫu số. Cách này rẻ; theo tóm tắt bài báo, nó tốt
  hơn shallow fusion ở mọi tác vụ, thường ngang ILME và tốt hơn DR
  ([Zheng et al. 2022](https://www.isca-archive.org/interspeech_2022/zheng22c_interspeech.pdf),
  [icefall LODR](https://icefall.readthedocs.io/en/latest/decoding-with-langugage-models/LODR.html)).
- Chỉnh prediction network bằng text giảm lỗi 10–45% ([Pylkkönen et al.](https://www.isca-archive.org/interspeech_2021/pylkkonen21_interspeech.html)).
  Tuy nhiên, decoder của icefall là stateless (context 2), nên ở đây lợi ích có thể thấp hơn.

Muốn kết hợp LM cần train RNN-LM bằng icefall hoặc n-gram bằng kaldilm. Phần này chưa cài.

### 3.2 Bậc 1: LoRA nhỏ, audio thật làm chủ

**Bằng chứng trên chính Zipformer** (tutorial icefall, đọc từ checkout `9c96676`). Hai thí nghiệm đều thích nghi
Zipformer LibriSpeech 960 giờ sang tập con nhỏ của GigaSpeech, nhiều data hơn ta khá nhiều:

| Cách | Cấu hình | WER test (greedy) | Miền gốc |
| --- | --- | --- | --- |
| Adapter | 4 adapter mỗi lớp, dim 8, `--base-lr 0.045`, 761.344 tham số (1,148%) | 19,27 → 15,42 | Tắt adapter thì giữ nguyên 2,23 / 4,96 |
| Full fine-tune | `--base-lr 0.0045` (khoảng 1/10 lr gốc), `--use-mux` trộn dữ liệu gốc | 13,66 | Cần `--use-mux` để giữ |

Vậy là:

- full fine-tune (có trộn dữ liệu cũ bằng `--use-mux`) đạt WER miền mới tốt hơn adapter;
- adapter giữ miền gốc tuyệt đối khi tắt;
- adapter tồn dư đạt gần bằng full fine-tune với dưới 0,5% tham số ([2109.06952](https://arxiv.org/abs/2109.06952)).

**Cấu hình đề xuất cho khoảng vài giờ audio thật:**

| Tham số | Giá trị | Lý do |
| --- | --- | --- |
| Kiểu train | LoRA encoder theo icefall `zipformer_lora` (mặc định), r 16, alpha 32, dropout 0,05 | Khoảng 1% tham số, khởi tạo B = 0 nên bắt đầu đúng bằng model gốc |
| `--lora-joiner` (mới, tuỳ chọn) | Chỉ bật như một thí nghiệm riêng | Mục 2: với RNN-T, phần lớn lợi ích đến từ joint layer, nơi chấm điểm từng token đầu ra. Nhưng nó đổi mọi output, nên kiểm tra drift |
| lr | ScaledAdam **0.045** (mặc định của icefall `zipformer_lora/finetune.py` và tutorial adapter). Commit này sửa mặc định cũ 0.0045; xem phần ngay dưới bảng | Nếu `broken_utterances` tăng nhanh qua các epoch, thử 0.02 |
| Số epoch | Tối đa 15, giữ epoch tốt nhất theo WER validation (đã có) | Dừng theo audio thật, không theo TTS |
| `--speed-perturb` | Bật | Tăng biến thể với giá rẻ (0,9 / 1,1); các recipe icefall đều dùng |
| TTS | `--max-tts-ratio 1` (mặc định, theo từng epoch). Chạy mốc `--max-tts-ratio 0` trước | Mục 3.4 và 3.6 |

**Điều chỉnh LR cần xác nhận bằng ASR.** Parser LoRA recipe mặc định 0.045, nhưng usage ở đầu chính file đó vẫn ghi 0.0045. Với LoRA khởi tạo B = 0, ScaledAdam có thể cập nhật chậm ở LR thấp:

- ScaledAdam nhân mỗi bước với max(RMS(tham số), 1e-5).
- `lora_B` bắt đầu bằng 0, nên mỗi bước nó lớn lên nhiều nhất khoảng (1 + lr) lần.

`scripts/scaled_adam_lora_lr_check.py` có bản quy tắc đơn giản và tuỳ chọn optimizer icefall thật, cho LoRA học một cập nhật hạng thấp trong 720 bước minh hoạ. Không quy đổi 720 bước này thành số epoch ASR:

| lr | rms(B) | Phần cập nhật cần học còn thiếu |
| --- | --- | --- |
| 0.0045 | ~4e-5 | ~99% |
| 0.045 | ~6e-4 | ~5% |

Đã chạy lại bài fit ma trận với ScaledAdam thật: sai số cập nhật còn khoảng 97,97% ở 0.0045 và 4,92% ở 0.045. Đây không phải đo độ chính xác ASR. Không thể từ đó kết luận mọi lần không cải thiện đều do LR hoặc loại trừ lỗi data. Mặc định 0.045 là một ứng viên cần chọn trên validation; AdamW mặc định 5e-4 có quy tắc cập nhật khác.

**Chạy hai lần rồi so.** Lần đầu chỉ audio thật; lần sau thêm TTS đã qua QC. Chỉ giữ TTS nếu validation tốt hơn *và*
số câu bị làm hỏng không tăng.

**Rehearsal (trộn dữ liệu cũ).** Bạn không muốn dùng các tập public phổ biến để *dạy thêm*; điều đó hợp lý. Nhưng bằng
chứng ở mục 2 cho thấy chừng 10% dữ liệu chung chỉ để *giữ* là cách chống quên rẻ nhất.

- Nếu thấy drift lớn trên public_test, cân nhắc thêm một phần nhỏ audio chung, khoảng 10% số câu.
- Thay thế không cần nhãn: lấy transcript của model gốc trên audio trong miền chưa nhãn làm nhãn giả để giữ hành vi cũ
  (kiểu learning-without-forgetting, noisy student;
  [2110.04056](https://ar5iv.labs.arxiv.org/html/2110.04056), [2110.00165](https://arxiv.org/pdf/2110.00165v3)).
- **Chỉ dùng public_test theo cách này nếu luật thi cho phép.** Hiện repo chỉ dùng nó để đo drift, không train.

### 3.3 Bậc 2: nội suy sau khi train

`export_asr_lora.py --scale s` merge `s × BA` vào trọng số. Với s = 0 là model gốc, s = 1 là model đã train.

`./run_finetune.sh scales gipformer` làm ba việc:

1. export các hệ số `LORA_SCALES` (mặc định 1.0, 0.7, 0.5);
2. chấm tất cả trên **validation**;
3. so mỗi bản với model gốc: số câu sửa được, số câu làm hỏng, recall từ ngoại.

Chọn hệ số rồi mới chấm test một lần.

Lý do: với ít data, phần cập nhật thường "vượt" ở chỗ data không phủ. Thu nhỏ nó giữ phần lớn lợi ích trong miền mà
giảm độ trôi (WiSE-FT, mục 2).

### 3.4 TTS: giúp gì và không giúp gì

- **Giúp từ mới khi trộn với audio thật.**
  - Amazon dùng TTS cho từ ngoài từ vựng, kèm EWC trên encoder: cải thiện 57% tương đối trên câu chứa từ đó, không
    giảm ở phần còn lại ([Zheng et al. 2021](https://cdn.amazon.science/aa/d5/427d9f744eb9afb2935201d6f80b/using-synthetic-audio-to-improve-the-recognition-of-out-of-vocabulary-words-in-end-to-end-asr-systems.pdf)).
    Ở mức 10 giờ, audio tổng hợp giúp hội tụ nhưng không thay được audio thật.
  - SynthASR báo hơn 65% tương đối trên ứng dụng mới mà không giảm ở ứng dụng cũ
    ([SynthASR](https://fugumt.com/fugumt/paper_check/2106.07803v1_enmode)).
  - Hướng gần đây: LLM sinh text, rồi TTS, rồi lọc, rồi LoRA, tránh tụt ngoài miền
    ([DAS](https://arxiv.org/html/2501.12501v1), [Hitachi 2026](https://arxiv.org/html/2603.16920)).
  - Có thể tăng trọng số loss cho từ mới, kèm L2/EWC để không quên
    ([Qu et al. 2023](https://api.deepai.org/publication/emphasizing-unseen-words-new-vocabulary-acquisition-for-end-to-end-speech-recognition)).
- **Không thay được audio thật.**
  - Chỉ dùng dữ liệu tổng hợp thì vẫn còn khoảng cách lớn ([Hilmes et al. 2024](https://arxiv.org/abs/2407.17997)).
  - Lợi ích của TTS phụ thuộc kiến trúc và rõ nhất với AED ([Rossenbach et al. 2021](https://arxiv.org/abs/2104.05379?context=cs)).
    Model của ta là transducer, nên kỳ vọng vừa phải.
- **Text và âm học đều cần kiểm tra.** Có nghiên cứu thấy tăng giọng bằng voice conversion không hiệu quả
  ([2204.00618](https://arxiv.org/pdf/2204.00618), [2406.07090](https://arxiv.org/html/2406.07090v2),
  [2410.16726](https://export.arxiv.org/pdf/2410.16726)); kết quả không đồng nhất giữa corpus/architecture. Không kết luận luôn ưu tiên text hơn mọi biến thiên giọng.
- **Repo đã làm:**
  - nhãn TTS theo đúng chữ được đọc;
  - hai ASR nghe lại (QC);
  - giới hạn tỉ lệ TTS;
  - kiểm tra drift.

### 3.5 Đo lường cho ASR

`eval_asr_onnx.py` nay báo các chỉ số sau:

| Chỉ số | Ý nghĩa |
| --- | --- |
| `wer_percent` | WER như cũ |
| `foreign_words.recall_percent` | Tỉ lệ từ ngoại/mã trong nhãn được nghe đúng |
| `foreign_words.false_alarms_per_100_utterances` | Từ ngoại/mã xuất hiện trong output nhưng không có trong nhãn. Nghe SJC thành SC tính là một lần bỏ sót cộng một false alarm, như keyword spotting |
| `fixed_utterances`, `broken_utterances` | So với model gốc: câu sửa được và câu làm hỏng |
| `changed_words_per_100` | Độ trôi trên public_test không nhãn |

Quy tắc: chọn mọi thứ (điểm hotword, epoch, scale, TTS ratio) trên **validation**; test chỉ để báo cáo lần cuối.

### 3.6 Khi audio thật chỉ vài giờ, phần còn lại là TTS clone giọng

Đây là tình huống của repo:

- **Audio thật:** 3.558 câu fit, cỡ vài giờ.
- **TTS:** 7.496 câu từ 229 seed, clone từ 753 đoạn giọng thật của train; thêm 790 câu agent từ 382 đoạn giọng.
- **Cách chọn seed:** theo nội dung ASR làm hỏng ([khảo sát](../outputs/asr_datagen_targets/KHAO_SAT_DE_GEN_DATA.md)),
  và giọng clone là giọng của chính seed. Đây gần đúng công thức Hard-Synth: LLM viết lại câu, rồi clone giọng của các
  câu model nghe sai, để bù phần âm học hiếm trong data ([Hard-Synth](https://arxiv.org/abs/2411.13159)). Hard-Synth
  báo giảm WER tương đối 6,5% / 4,4% trên LibriSpeech dev/test-other và hiệu quả về lượng data.

**Bằng chứng cho trường hợp "ít thật, nhiều TTS":**

| Câu hỏi | Kết quả đã công bố |
| --- | --- |
| Ít data thật + TTS có giúp không | Có, kể cả rất ít. Với 24 phút audio có nhãn, TTS giảm WER tương đối tới 25,5%; self-training trên audio chưa nhãn giảm tới 20,5% ([Bartelds et al. 2023](https://aclanthology.org/2023.acl-long.42)). TTS train từ chính data ASR rồi đem tăng cường cũng cải thiện ([Laptev et al. 2020](https://arxiv.com/abs/2005.07157)) |
| Có trần không | Có. TTS mở rộng cả đa dạng từ vựng lẫn âm học nhưng lợi ích có giới hạn, và model train bằng giọng tổng hợp vẫn kém xa model train bằng giọng người ([Rosenberg et al. 2019](https://arxiv.org/pdf/1909.11699); [Hilmes et al. 2024](https://arxiv.org/abs/2407.17997)) |
| Tỉ lệ thật : TTS | 1:1 được ủng hộ trong abstract [IEEE](https://xplorestaging.ieee.org/document/11252042), chưa kiểm tra đủ bảng thực nghiệm. LREC 2026 §3.3 có **hai** chế độ: cố định tập TTS tốt nhất ở 7:1, đổi câu TTS tốt nhất ở 2:1; đều dùng 1 giờ tiếng Frisian, XLS-R + XTTS, 2.000 bước. Không suy ra giới hạn 2:1 cho mọi model. Bài 2410.16726 thử các tập 200–3.700 giờ, khác quy mô của repo |
| Bao nhiêu giọng | Ngưỡng khoảng 40 là kết quả Whisper-small/OmniVoice, tiếng trẻ em Anh; không áp trực tiếp cho Gip/Zip. Bài đó nhấn mạnh đa dạng âm học trên cùng transcript, còn 2410.16726 nhấn mạnh text trong thí nghiệm CosyVoice khác. 753 đoạn tham chiếu không xác nhận có 753 người nói |
| Khoảng cách phân phối | Synt++ lọc mẫu TTS (rejection sampling) và tách thống kê batch-norm cho thật/TTS, giảm tới 13% lỗi ASR tương đối so với dùng TTS thô ([Synt++](https://machinelearning.apple.com/research/utilizing-imperfect-synthetic)). Zipformer dùng BiasNorm, không có thống kê batch, nên chỉ phần lọc áp dụng được; repo đã có QC hai ASR |
| Cho TTS dạy phần nào | Với RNN-T, chỉnh prediction + joint network bằng TTS sinh từ text miền đích thắng cả sửa chính tả và LM rescoring, không thêm chi phí khi chạy ([Li et al. 2020](https://arxiv.org/pdf/2007.15188)) |

**Vì sao không để TTS dạy encoder.** TTS là nguồn tốt cho *từ* (SJC, ZALO, tên thuốc, cách nói trong miền) nhưng là
nguồn âm học có thể lệch: giọng clone có thể mang dấu vết vocoder/kênh sạch. Nếu TTS chiếm phần lớn gradient
của encoder, model có thể học "kênh TTS" thay vì học nghe. Encoder là nơi học âm học; joint network là nơi chấm điểm
từng token. Đóng băng encoder là một thí nghiệm hạn chế đường cập nhật, không bảo đảm toàn model không lệch và cũng có thể hạn chế sửa lỗi âm học.

**Đã cài cho tình huống này:**

| Tuỳ chọn | Tác dụng |
| --- | --- |
| `--max-tts-ratio R` | Nay áp **theo từng epoch**: tối đa R câu TTS cho mỗi câu thật của epoch đó (kể cả bản lặp). `-1` dùng hết |
| Xoay vòng một hoán vị TTS theo seed (`--fixed-tts-subset` để giữ cố định) | Tránh lấy mẫu độc lập làm một số câu không bao giờ được chọn; chế độ số giây báo các câu quá dài không vừa budget |
| `--tts-ratio-unit seconds` | Giới hạn theo thời lượng sau speed perturb, log cả số câu và số giây; `asr2` mặc định dùng cách này. Lệnh train trực tiếp giữ mặc định `utterances` để tương thích |
| `--real-repeat K` | Mỗi câu thật xuất hiện K lần mỗi epoch; bản lặp được đổi tốc độ 0,9 / 1,1 / 0,95 / 1,05 nên không trùng y hệt. Dùng khi muốn nhiều TTS hơn mà vẫn giữ tỉ lệ |
| `--tts-speed-perturb`, `--tts-reverb-prob P` | Chỉ cho TTS: tốc độ ngẫu nhiên 0,9 / 1,0 / 1,1, và vang phòng mô phỏng của lhotse (không cần bộ RIR). Vang cần torchaudio; script thử trước khi train và báo rõ nếu thiếu |
| `--lora-targets joiner` | LoRA chỉ ở joint network; encoder giữ nguyên. Với Zip, CR-CTC tự tắt vì không còn tham số nào trên đường đó |
| `./run_finetune.sh asr2 KIND` | Giai đoạn 1: LoRA encoder chỉ trên audio thật, có speed perturb. Giai đoạn 2: từ model giai đoạn 1, LoRA joiner trên thật (`REAL_REPEAT`, mặc định 2) + TTS (`MAX_TTS_RATIO`, mặc định 1; speed; `TTS_REVERB_PROB`, mặc định 0,3). Cuối cùng chấm model gốc / giai đoạn 1 / giai đoạn 2 trên validation |

`--real-repeat 2 --max-tts-ratio 1` cho mỗi epoch:

- ít nhất 50% là audio thật (bản gốc + bản 0,9×);
- TTS tối đa bằng hai lần số câu thật khác nhau.

Với kho hiện tại (trước QC: Gip 4.538 câu, Zip 3.748 câu), như vậy là dùng toàn bộ kho mỗi epoch, và audio thật chiếm
khoảng 61% (Gip) và 65% (Zip) **theo số lượt câu**, không phải số giờ hoặc lượng thông tin độc lập. Đây không phải bằng chứng đặt trần TTS ở 2 lần. `asr2` nay dùng duration budget; khi stage 1 không cải thiện, stage 2 vẫn có thể thử từ model gốc. Cần manifest TTS đã QC, không rỗng, trước khi chạy.

**Gợi ý thêm (chưa cài):**

- **Ưu tiên TTS "khó".** Nhóm `asr_learnable` dạy được nhiều. Nhóm `asr_already_right` chủ yếu thêm âm học TTS mà ít thêm
  từ. Khi phải cắt bớt TTS, nên ưu tiên nhóm đầu.
- **Giọng clone:** bắt đầu với một tập giọng đa dạng và đo saturation; chưa có ngưỡng chung cho tiếng Việt. Với câu chứa từ khó, có thể thử đọc
  bằng 2 giọng khác nhau. Clone ưu tiên giọng của các câu model nghe sai.
- **Kiểm tra model có học "kênh TTS" không.** Giữ lại khoảng 5% câu TTS (theo seed) làm tập kiểm tra. Nếu WER trên TTS giữ
  lại giảm mạnh mà validation gốc không giảm, đó là dấu hiệu lệch miền cần điều tra, chưa đủ xác định nguyên nhân.
- **Mỗi giờ audio thật đáng giá hơn nhiều giờ TTS.** Nếu có thể, nhờ vài người đọc câu miền thi bằng điện thoại.
  Self-training trên audio thật chưa nhãn cũng có bằng chứng tốt (Bartelds et al.), nhưng public_test chỉ dùng được nếu
  luật thi cho phép.

## 4. Qwen3.5-4B: công thức

### 4.1 Dữ liệu là đòn bẩy chính

- **Ít mà đúng là đủ.**
  - FireAct: fine-tune Llama-2-7B với 500 quỹ đạo agent tăng 77% trên HotpotQA ([FireAct](https://arxiv.org/pdf/2310.05915)).
  - LIMA: 1.000 ví dụ chọn kỹ đã cạnh tranh được với model dùng nhiều data hơn. Tăng đa dạng đầu vào và chất lượng đầu
    ra có tác dụng; chỉ tăng số lượng thì không chắc ([LIMA](https://arxiv.org/pdf/2305.11206)).
  - Ta có 7.021 quỹ đạo đã devkit xác nhận, đủ cho SFT.
- **Đa dạng đầu vào hơn là nhân bản.** Các nguồn đã có:
  - `run:DIR`: đầu vào ASR nhiễu thật;
  - câu diễn đạt lại `generated` (`pass_text`);
  - câu viết tay: thiếu thông tin, đổi ý.
  `build_agent_sft.py` nay **bỏ quỹ đạo trùng y hệt** (`skipped_duplicate`). Lý do: khi ba run nghe giống nhau, câu dễ
  không bị nhân ba trọng số. `--keep-duplicates` giữ hành vi cũ.
- **Mẫu âm và mẫu hỏi lại giảm gọi tool bừa** ([Agent-FLAN](https://arxiv.org/pdf/2403.12881)). Repo đã có mẫu
  `missing_info` và `cancel`.
- **Chỉ train dữ liệu agent có thể giảm tổng quát hoá.** AgentTuning trộn dữ liệu agent với dữ liệu tổng quát, và báo
  rằng chỉ train dữ liệu agent làm giảm năng lực trên tác vụ ngoài tập ([AgentTuning](https://arxiv.org/pdf/2310.12823)).
  Ở đây Qwen chỉ làm tác vụ này, nên không bắt buộc. Nếu thấy model mất khả năng hiểu câu lạ, thêm khoảng 10–20% hội
  thoại tiếng Việt tổng quát.

### 4.2 Cấu hình LoRA

| Tham số | Đề xuất | Bằng chứng |
| --- | --- | --- |
| Module | Mọi lớp tuyến tính của language model, gồm MLP và Gated DeltaNet (đang làm) | [Biderman et al.](https://arxiv.org/html/2405.09673v2); [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): chỉ gắn vào attention là kém |
| lr | 1e-4 (đang dùng). Thử 5e-5 và 2e-4 | LoRA cần lr khoảng 10 lần full fine-tune, gần như không phụ thuộc rank (LoRA Without Regret, qua [tài liệu TRL](https://huggingface.co/docs/trl/main/en/lora_without_regret)) |
| Batch hiệu dụng | 16 (1 × gom 16, đang dùng) | LoRA kém chịu batch lớn; khuyến nghị dưới 32 (LoRA Without Regret) |
| Rank | 16–32; `--use-rslora` nếu rank cao | Ở chế độ ít data, LoRA ngang full fine-tune. rsLoRA, DoRA, LoRA+ cho kết quả lẫn lộn ([2505.19819](https://arxiv.org/pdf/2505.19819), [2502.10497](https://arxiv.org/pdf/2502.10497)), nên coi là ablation |
| Dừng sớm | `--early-stopping-patience 3` (mới), giữ checkpoint có eval loss thấp nhất (đang làm) | Quên tăng theo số bước ([Kalajdzievski](https://arxiv.org/abs/2401.05605v1)) |
| `--neftune-alpha 5` (mới) | Ablation | NEFTune tăng AlpacaEval từ 29,79% lên 64,69% với Llama-2-7B ([NEFTune](https://arxiv.org/pdf/2310.05914)). Đó là văn bản tự do, chưa chắc có lợi cho JSON ngắn |
| Loss trên prompt | Hiện chỉ tính trên JSON đích. Thử trọng số nhỏ cho prompt (chưa cài) | Có lợi khi prompt dài, output ngắn và ít data ([Shi et al. 2024](https://arxiv.org/pdf/2405.14394)), đúng tình huống này. Nhưng prompt chứa schema lặp lại, nên trọng số phải nhỏ |
| `merge_qwen_lora.py --scale` (mới) | Merge bản 1,0 và 0,7, so trên validation benchmark | WiSE-FT, mục 2. Chỉ cho LoRA thường; DoRA bị từ chối vì vector magnitude đã train |

Chọn checkpoint theo **state match trên validation benchmark**, không chỉ theo eval loss: loss của JSON ngắn không
phản ánh hết việc gọi đúng tool.

### 4.3 Dữ liệu tự sinh có kiểm chứng: RFT / ReST-EM

Ý tưởng:

1. cho model lấy mẫu nhiều lần trên câu train;
2. giữ quỹ đạo mà bộ kiểm chứng xác nhận đúng;
3. fine-tune lại.

Bằng chứng:

- ReST-EM vượt fine-tune trên dữ liệu người cho toán và code ([ReST-EM](https://arxiv.org/abs/2312.06585)).
- Theo SDFT, mục tiêu do chính model viết gần phân phối của nó hơn, nên quên ít hơn mà kết quả tác vụ ngang hoặc tốt
  hơn ([SDFT](https://arxiv.org/html/2402.13669v2)).

Ở repo này, devkit là bộ kiểm chứng: state hash cuối phải khớp nhãn.

`self:DIR` mới trong `build_agent_sft.py`:

- **Lấy gì:** lời gọi tool mà agent *tự* thực hiện trong một run đã lưu (`results.jsonl`, chỉ mẫu `state_match`), trên
  đúng đầu vào agent đã thấy (`samples/<id>/trace.jsonl`).
- **Kiểm chứng:** chạy lại các lời gọi đó trong phiên devkit mới; chỉ giữ nếu không lỗi tool và state hash khớp nhãn.
- **Chỉ mẫu fit:** validation và test vẫn chỉ có đích gold.
- **Giới hạn:** `--max-trajectories-per-sample N` giới hạn số quỹ đạo mỗi mẫu, để câu dễ không chiếm trọng số.
- **Giá trị thêm:** quỹ đạo trùng y hệt (cùng đầu vào, cùng lời gọi) chỉ giữ một lần. Cái thêm vào là các đường đúng
  khác gold, ví dụ có thêm bước `list` trước `add`: đúng state, và tự nhiên với model.
- **Cảnh báo về cách chấm:** devkit chỉ kiểm state cuối, không kiểm chuỗi lời gọi. `task.md` liệt kê `tool_calls` là một
  output được kiểm tra.
  - Nếu điểm chính thức so `tool_calls` với gold, đường "đúng state nhưng khác gold" có thể bị trừ điểm. Khi đó, dùng các
    lượt lấy mẫu dưới dạng `run:DIR` (đầu vào đa dạng, đích gold) thay vì `self:DIR`.
  - `summary.json` đếm `kept_self_differs_from_gold` để biết có bao nhiêu đường như vậy.

Chạy (máy GPU; bốn lượt lấy mẫu dùng chung ASR cache):

```bash
for k in 1 2 3 4; do
  OUTPUT_DIR=outputs/rft/pass$k ASR_CACHE_DIR=outputs/dual_asr_qwen/asr_cache ASR_CACHE_ONLY=1 LLM_TEMPERATURE=0.7 \
    ENV_MODE=official ./run.sh all --benchmark 1545 --seed 42
done
```

Thêm các nguồn `self:` vào lệnh build SFT, chạy riêng một lệnh:

```bash
ENV_MODE=official ./run.sh build_agent_sft.py --source reference --source run:outputs/dual_asr_qwen/benchmark_seed42_n1545 \
  --source self:outputs/rft/pass1/benchmark_seed42_n1545 --source self:outputs/rft/pass2/benchmark_seed42_n1545 \
  --source self:outputs/rft/pass3/benchmark_seed42_n1545 --source self:outputs/rft/pass4/benchmark_seed42_n1545 \
  --source generated --source synthetic --drop-examples --max-trajectories-per-sample 6 --out outputs/agent_sft_rft
```

Khi chạy:

- Đặt các cờ `ASR_*` và `AGENT_*` giống run gốc mà cache được tạo, để ASR cache khớp.
- Không có seed cố định cho sampling, nên mỗi lượt khác nhau và không tái lập được từng token.
- Nhiệt độ cũng áp vào bước chuẩn hoá. Vì vậy đầu vào agent giữa các lượt cũng đa dạng hơn.
- Vòng tiếp theo: lấy mẫu bằng model SFT v1, rồi train lại *từ model gốc* trên dữ liệu đã lọc. ReST-EM lặp như vậy.

**Bước sau (chưa cài):**

- DPO trên cặp quỹ đạo sai và đúng của cùng mẫu, tại bước đầu tiên khác nhau. Các run trên đã có sẵn cặp
  ([ETO](https://arxiv.org/abs/2403.02502)).
- GRPO với phần thưởng theo luật (đúng tên tool, tham số, state). Kết quả báo tăng khoảng 15% so với SFT, nhưng
  SFT-rồi-RL không luôn tốt hơn ([ToolRL](https://arxiv.org/abs/2504.13958)).

### 4.4 Prompt, ràng buộc và chuẩn hoá

- **Rút gọn prompt khoảng 11,6k token bằng chọn tool liên quan** (ToolRAG của
  [TinyAgent](https://arxiv.deeppaper.ai/papers/2409.00608v3)). Model nhỏ đỡ bị nhiễu bởi schema không liên quan, và
  nhanh hơn. Nếu làm, SFT phải dựng lại với cùng cách chọn tool.
- **Ràng buộc lúc decode chỉ cho cấu trúc JSON và tên tool.** Ràng buộc ngữ pháp giúp JSON luôn hợp lệ
  ([XGrammar-2](https://arxiv.org/pdf/2601.04426)), nhưng có báo cáo "constraint tax" làm giảm chất lượng nội dung ở
  model nhỏ ([2605.26128](https://arxiv.org/pdf/2605.26128), [2608.13959](https://arxiv.org/pdf/2608.13959)). Đừng ràng
  buộc giá trị tự do.
- **Chuẩn hoá ASR bằng LLM theo kiểu sửa có điều kiện.** Sửa lỗi từ N-best bằng LLM có hiệu quả
  ([HyPoradise](https://arxiv.org/pdf/2309.15701)), nhưng sửa quá tay là rủi ro thật. Chỉ sửa khi ASR không chắc hoặc hai
  ASR bất đồng ([Naderi et al. 2024](https://publications.idiap.ch/attachments/papers/2024/Naderi_INTERSPEECH_2024.pdf),
  [2310.11532](https://arxiv.org/html/2310.11532v2)). Repo đã truyền bằng chứng ASR thô (`AGENT_ASR_EVIDENCE`) và có dữ
  liệu SFT normalize từ N-best + nhãn.

### 4.5 Đo lường cho Qwen

`compare_benchmark_runs.py` nay in `mcnemar_exact_p`, tổng và theo split. Đây là kiểm định McNemar chính xác, hai phía,
chỉ dựa trên các mẫu đổi kết quả.

Với 248 mẫu test, mức tăng ròng tối thiểu để p < 0,05:

| Số mẫu bị làm hỏng | Cần sửa được ít nhất | Tăng ròng | % của 248 |
| ---: | ---: | ---: | ---: |
| 0 | 6 | 6 | 2,4 |
| 5 | 15 | 10 | 4,0 |
| 10 | 23 | 13 | 5,2 |
| 20 | 36 | 16 | 6,5 |

Hệ quả: chọn cấu hình trên validation (231 mẫu), xác nhận một lần trên test, và đừng tin chênh lệch 1–3 điểm.

## 5. Lộ trình đề xuất

| Bước | Việc | Quyết định dựa trên |
| --- | --- | --- |
| A | `./run_finetune.sh hotwords` (CPU, không train) | Validation: WER, recall từ ngoại, false alarm, câu làm hỏng. Nếu có lợi, bật `*_DECODING`/`*_HOTWORDS` và benchmark agent trên validation |
| B | `./run_finetune.sh tts`, rồi `qc`, rồi nghe `listen_sample.csv` | Tỉ lệ bị loại so với mức 12,8% trên giọng thật |
| C | `./run_finetune.sh asr2 gipformer` (và `zipformer`). So thêm với cách một giai đoạn (`./run_finetune.sh gipformer`, thật + TTS cùng lúc). Sau đó `./run_finetune.sh scales …` cho bản tốt nhất | Validation; giữ bản có câu làm hỏng thấp và drift public_test nhỏ |
| D | Qwen SFT v1 trên data hiện có, `QWEN_ARGS="--early-stopping-patience 3"`; merge 1,0 và 0,7 | State match validation |
| E | RFT: 4 lượt lấy mẫu, build với `self:`, SFT v2 từ model gốc | State match validation; McNemar so với v1 |
| F | Chấm test một lần cho cấu hình cuối, kèm `mcnemar_exact_p` | Báo cáo |

## 6. Không nên làm

| Đừng | Thay bằng |
| --- | --- |
| Train hoặc chọn tham số trên test, hay train trên public_test | Chọn trên validation; public_test chỉ để đo drift |
| Lấy danh sách hotword từ validation hoặc test | Chỉ từ split fit (`build_hotwords.py` đã làm vậy) |
| Để TTS chiếm gần hết mỗi epoch (hơn khoảng 2 lần số câu thật), dùng TTS chưa QC, gán nhãn TTS theo chữ viết | `--max-tts-ratio` 1–2 theo epoch (có thể kèm `--real-repeat`), `*_qc.jsonl`, nhãn theo chữ được đọc |
| Để TTS dạy encoder khi audio thật chỉ vài giờ | `./run_finetune.sh asr2`: encoder chỉ học audio thật, TTS chỉ dạy joint network |
| Full fine-tune lr lớn nhiều epoch, không dừng sớm | LoRA, chọn epoch theo WER validation trên audio thật |
| Chọn epoch theo loss trên TTS | Validation chỉ gồm audio thật (đã vậy) |
| Boost từ thông dụng (NHẮN, HỌP) bằng hotword | Để lỗi nhầm động từ lệnh cho LoRA trên audio thật và bước chuẩn hoá |
| Kết luận từ chênh lệch 1–3 điểm trên 248 mẫu | Xem `mcnemar_exact_p` |

## 7. Đã cài trong commit này

| File | Thay đổi | Mặc định |
| --- | --- | --- |
| `asr/gipformer.py`, `asr/zipformer.py`, `asr/__init__.py` | `decoding_options`, `decoding_identity`, `bpe_vocab` (xuất từ `bpe.model` ra thư mục tạm để không đổi fingerprint model); nhận `decoding` | Greedy |
| `main.py`, `cache_asr.py`, `benchmark.py` | Khoá cache và identity gồm cấu hình decode và hash danh sách hotword; `timestamp_word` từ chối beam/hotword | Khoá cũ giữ nguyên |
| `build_hotwords.py`, `outputs/asr_hotwords/` | Danh sách hotword theo model từ split fit | — |
| `eval_asr_onnx.py` | Decode riêng cho mỗi `--model …@…`; chỉ số `foreign_words` | — |
| `asr_lora/lora.py`, `export_asr_lora.py`, `train_asr_lora.py` | `scale_lora` / `--scale`; `--lora-joiner` (lưu trong `lora_config`, export dựng lại đúng). **Sửa lr mặc định của ScaledAdam: 0.0045 → 0.045** (AdamW: 5e-4) | Scale 1, không joiner |
| `scripts/scaled_adam_lora_lr_check.py` | Kiểm tra lr ở trên, chỉ cần torch | — |
| `asr_lora/data.py` (`epoch_cuts`), `train_asr_lora.py` | Trộn thật/TTS theo epoch: `--real-repeat`, TTS mới mỗi epoch (`--fixed-tts-subset`), `--tts-speed-perturb`, `--tts-reverb-prob`; `--lora-targets joiner`; bước `asr2` trong `run_finetune.sh` | Tỉ lệ 1, lặp 1, không tăng cường; lấy TTS mới mỗi epoch |
| `run_finetune.sh` | Bước `hotwords` và `scales KIND` | — |
| `train_qwen_lora.py` | `--neftune-alpha`, `--use-rslora`, `--use-dora`, `--early-stopping-patience` | Tắt |
| `merge_qwen_lora.py` | `--scale` (chỉ LoRA thường) | 1 |
| `build_agent_sft.py` | Nguồn `self:DIR` (rejection sampling, đếm `kept_self_differs_from_gold`); bỏ quỹ đạo trùng; `--max-trajectories-per-sample`; `--keep-duplicates` | Bỏ trùng bật |
| `compare_benchmark_runs.py` | `mcnemar_exact_p` tổng và theo split | — |
| `tests/test_low_data_knobs.py` | Test: decode/cache, hotword, scale và lr ASR, scale Qwen (torch/peft), `self:` end-to-end với devkit giả, McNemar | — |

## 8. Giới hạn của tài liệu này

- **Tôi không đọc trực tiếp được nhiều nguồn.** Các trang arxiv.org, huggingface.co, thinkingmachines.ai và
  developer.nvidia.com bị chặn trong phiên này. Các số liệu từ những nguồn đó lấy qua tóm tắt tìm kiếm hoặc bản mirror.
  - Số liệu icefall đọc từ checkout tài liệu local.
  - Hành vi hotword của sherpa-onnx đọc từ mã nguồn trên GitHub.
  - Muốn kiểm tra lại từng con số, cho phép các host đó trong cài đặt mạng (Network access) của môi trường.
- **Chưa decode thật với hotword hay scale**, vì máy này không có model ASR. Các con số "phủ được" ở mục 3.1 là trần lý
  thuyết tính từ transcript greedy đã lưu.
- **Phần Qwen mới chỉ smoke test** trên một Qwen3.5 thu nhỏ, trọng số ngẫu nhiên, chạy CPU:
  - train với `--neftune-alpha 5 --use-rslora --early-stopping-patience 1`;
  - merge `--scale 0.5`: 25 lớp LoRA được scale, chênh logit 0,008 do bf16.
- **Các thí nghiệm GPU (bước C–E) chưa chạy.** Tài liệu không hứa mức cải thiện cụ thể.
