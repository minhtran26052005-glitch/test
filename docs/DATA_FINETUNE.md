# Data để fine-tune: đã có gì, dùng cho ASR hay agent, còn thiếu gì

Số liệu tính trên snapshot 05/11/2026. Chạy lại: `python audit_finetune_data.py` (ra `outputs/finetune_data_audit/`),
`python eval_fewshot_retrieval.py`. Không script nào gọi model hoặc API.

## 1. Data đã gen (`outputs/asr_datagen_api/generated.csv`) phục vụ được gì

10.980 biến thể text của 1.098 seed. Mỗi biến thể được chấm riêng cho hai mục đích trong
[`outputs/finetune_data_audit/variants.csv`](../outputs/finetune_data_audit/variants.csv).

### Cho ASR: dùng được phần lớn, nhưng mới là text

| Kết quả | Biến thể |
| --- | ---: |
| Dùng được (`asr_use=ok`) | 10.867 |
| Cần sửa nhẹ (đã chấm tay `edit_text`) | 26 |
| Loại: lẫn chữ Nga/Ả Rập/CJK (51), bản đọc hỏng đã xác nhận (32), loại khi chấm tay (47); một câu có thể dính nhiều lý do | 87 |

- Với ASR, lỗi *nghĩa* (đổi báo thức thành lịch hẹn…) không quan trọng. Chỉ cần audio đọc đúng `spoken_text`. Vì vậy đổi slot (tên người lạ, số khác) vẫn có ích: tên mới giúp ASR nghe được nhiều tên hơn.
- **Chưa có audio.** Muốn fine-tune GipFormer/ZipFormer phải TTS `spoken_text` trước (mục 4), rồi trộn với audio thật để model không học thuộc giọng TTS.
- Phân bố đã lệch theo hướng sinh data cho ASR: lượt cuối tập trung ở lịch/nhắc việc (1.110), tài chính (1.040), nhà thông minh (590). Đúng với mục tiêu ban đầu là các cụm ASR hay sai.

### Cho agent: chỉ dùng được một phần nhỏ, và không giải quyết lỗi chính của agent

Data này không có nhãn tool call riêng. Muốn dùng cho agent thì phải **dùng lại nhãn gold của seed**. Điều đó chỉ đúng khi biến thể thuộc lượt cuối, không đổi slot, và giữ được mọi từ khoá mà tham số gold phụ thuộc vào (tên, query, số…; số được so ở dạng lời đọc).

| 3.360 biến thể lượt cuối | Biến thể |
| --- | ---: |
| `label_safe`: dùng lại nhãn gold được | 2.027 |
| `label_safe_free_text`: dùng được, nhưng nhãn có nội dung tự do (tin nhắn/nhắc nhở) có thể lệch chữ với câu đã diễn đạt lại | 822 |
| `needs_new_label`: đổi slot (393) hoặc làm mất từ khoá (72) | 465 |
| Loại vì lỗi ASR/chấm tay | 46 |

Ngoài ra có 7.620 biến thể của lượt history. Chúng dùng được để diễn đạt lại history, vì nhãn của lượt cuối không đổi, nhưng chưa được kiểm tra nghĩa.

Vì sao chỉ có ích một phần:

1. **Độ phủ nhỏ.** Chỉ 336/1.545 mẫu lượt cuối có biến thể, 330 mẫu dùng lại được nhãn; phần `fit` có 1.282 biến thể `label_safe`. **28 app không có biến thể nào**: appliances, door_lock, email, flights, home_automation, home_security, learning, library, maps, music, podcast_radio, subscriptions, tasks, video_meeting…
2. **Text sạch, khác input thật.** Lúc inference agent nhận text đã qua ASR và normalizer, có lỗi. Câu diễn đạt lại bằng văn viết không dạy agent cách chịu lỗi ASR.
3. **Không có observation.** Nhãn chỉ có chuỗi tool call. Muốn dạy agent nhiều bước thì phải phát lại trong devkit để có kết quả tool thật (mục 2).
4. **Không nhắm vào lỗi chính.** Agent sai chủ yếu ở câu nhiều bước (3 lời gọi: 13% đúng; 4+: 5%), ở việc đáng lẽ phải hỏi lại mà tự bịa giá trị (144 ca), và ở việc gọi lặp. Diễn đạt lại cùng một câu không tạo ra tình huống mới cho các lỗi này.

## 2. Data có sẵn để fine-tune ngay (không cần gen thêm)

`build_agent_sft.py` dựng data chat (system / user / assistant) cho đúng con Qwen đang dùng cho cả agent lẫn normalizer:

- **Agent:** phát lại tool call gold trong devkit nên observation là thật. User message là **đúng payload** mà `run_agent` gửi lúc inference (dùng chung `agent.build_payload`). Chỉ giữ mẫu khi mọi lời gọi gold đều được devkit chấp nhận và hash trạng thái cuối khớp nhãn.
- **Normalizer (phần sửa lỗi ASR):** input là các bản ASR đã cache của từng audio, đích là transcript reference. Data này giúp ASR mà không cần TTS: dạy Qwen sửa lỗi ASR và chuẩn hoá số, trong khi normalizer hiện tại đang làm hỏng nhiều hơn sửa (523 ca tệ hơn, 71 ca tốt hơn).

Ước tính trên phần `fit` (1.066 mẫu, trung bình 3,14 bước/mẫu), trước khi lọc trạng thái lệch:

| Nguồn | Bản ghi |
| --- | ---: |
| `reference`: transcript sạch | ~3.342 |
| `run:` từng lượt benchmark đã lưu (Gip, Zip, dual), tức input có lỗi ASR thật | ~3.342 mỗi lượt |
| `generated`: 1.282 biến thể `label_safe` | ~4.758 |
| Normalizer: 3.558 audio `fit`, cả hai ASR (thêm `--normalize-single` thì ×3) | 3.558 |

```bash
ENV_MODE=official AGENT_ASR_EVIDENCE=1 ./run.sh build_agent_sft.py \
  --source reference \
  --source run:outputs/benchmark_seed42_n1545 \
  --source run:outputs/zipformer_150m/benchmark_seed42_n1545 \
  --source run:outputs/dual_asr_qwen/benchmark_seed42_n1545 \
  --source generated --drop-examples \
  --normalize-run outputs/dual_asr_qwen/benchmark_seed42_n1545 --normalize-single \
  --out outputs/agent_sft
```

- Chạy qua `./run.sh` để system prompt và các cờ agent khớp với lúc inference. Nhớ dùng cùng cờ khi benchmark model đã fine-tune.
- Nguồn `run:` cần thư mục `samples/` của archive kết quả (`python scripts/fetch_results.py --extract`) và devkit trong `data/`.
- File ra theo split manifest: `train.jsonl` (fit), `validation.jsonl`, `test.jsonl`. **Không train trên `test`.** Sau khi fine-tune, chỉ con số trên split `test` mới là so sánh công bằng: `compare_benchmark_runs.py` có mục `by_split`.
- `tool_schemas` nằm trong mọi payload nên bản ghi dài. Hãy đo số token trước khi chọn `max_seq_length`; giới hạn của cuộc thi là 16K.
- Gợi ý train: LoRA trên Qwen3.5-4B, chỉ tính loss trên phần assistant, trộn bản ghi agent và normalizer trong cùng một adapter. Có thể dùng TRL `SFTTrainer` hoặc LLaMA-Factory, cả hai đọc được định dạng `messages`. Gộp LoRA vào model thì số tham số không đổi (dưới 5B).

## 3. Nên gen thêm gì

**Cho agent: sinh tác vụ mới trên môi trường có sẵn, không chỉ diễn đạt lại.**
Với mỗi `train_environment/<id>/state.json`, cho LLM thầy đề xuất một yêu cầu kèm chuỗi tool call gold. Giữ lại chỉ khi phát lại trong devkit thành công, theo đúng kiểm tra `replay` của `build_agent_sft.py`. Ưu tiên:
- 28 app chưa có data;
- câu 3–5 việc, câu có điều kiện "nếu… thì… không thì", câu đổi ý giữa chừng;
- tình huống phải hỏi lại: thiếu số tiền, mức âm lượng, giờ; nhiều liên hệ cùng tên. Trong gold, `ask_missing_info` luôn là lời gọi cuối, và 153/209 ca có làm việc khác trước khi hỏi.

**Cho cả ASR và agent cùng lúc: TTS → ASR.**
TTS các câu đã gen, rồi cho chạy qua chính GipFormer/ZipFormer:
- cặp audio + `spoken_text` dùng để fine-tune ASR;
- transcript ASR có lỗi thật thay cho `normalized_text`, dùng làm input SFT cho agent và normalizer.

Như vậy một lần TTS phục vụ được cả hai phía.

## 4. Nguồn ngoài (cần tự kiểm tra license và luật thi)

Chưa tìm thấy văn bản luật chính thức nói về data ngoài và việc fine-tune. Các repo của đội khác ghi giới hạn tổng < 6B tham số, LLM < 5B, context 16K, inference offline. Nhiều đội có fine-tune LoRA cho Qwen và PhoWhisper, nhưng đó là nguồn thứ cấp: **cần đối chiếu với thể lệ chính thức** trên [challenge.zalo.ai](https://challenge.zalo.ai/). (`task.md` ghi "Không fine-tune model" là phạm vi của nhiệm vụ code ban đầu, không phải luật thi.)

| Nguồn | Dùng cho | Quy mô | License (theo tìm kiếm, cần xác minh) |
| --- | --- | --- | --- |
| [Bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) | ASR | ~500 giờ | Không nhất quán: Apache-2.0 / CC BY-NC-SA 4.0, ghi "research only" |
| [VIVOS](https://huggingface.co/datasets/vivos) | ASR | ~15 giờ | CC BY-NC-SA 4.0 |
| [viVoice](https://huggingface.co/datasets/capleaf/viVoice) | ASR/TTS | ~1.017 giờ, 186 kênh YouTube | CC BY-NC-SA 4.0, chỉ cấp cho email tổ chức |
| [VietMed](https://huggingface.co/datasets/leduckhai/VietMed) | ASR y tế (tên thuốc) | 16 giờ có nhãn + ~2.200 giờ không nhãn | Chưa rõ |
| [GigaSpeech 2](https://huggingface.co/datasets/speechcolab/gigaspeech2) (vi) | ASR | ~6.000 giờ | Không nhất quán: Apache-2.0 / CC BY-NC-SA 4.0 |
| [5CD-AI xLAM function calling (vi)](https://huggingface.co/datasets/5CD-AI/Vietnamese-Salesforce-xlam-function-calling-Gemini) | Agent | 10K–100K | Chưa kiểm tra |
| [wannaphong function calling (Qwen format)](https://huggingface.co/datasets/wannaphong/vietnamese-Locutusque-function-calling-qwen) | Agent | — | Chưa kiểm tra |
| [phamhai Vietnamese-Function-Calling-Test](https://huggingface.co/datasets/phamhai/Vietnamese-Function-Calling-Test) | Đánh giá agent | 2.899 mẫu một lượt | Chưa kiểm tra |
| [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) | TTS cho data đã gen | 300M, clone giọng từ 3–5 giây, đọc lẫn Anh–Việt | Apache-2.0 |
| [F5-TTS-Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese) | TTS | — | Xem repo |

Data function-calling công khai thường chỉ có một lượt và tool khác hẳn devkit. Nó chỉ nên là phần phụ để giữ năng lực chung; phần chính vẫn là các quỹ đạo phát lại từ devkit ở mục 2.
