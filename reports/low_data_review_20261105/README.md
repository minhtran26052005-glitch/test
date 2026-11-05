# Log và bằng chứng kiểm tra low-data/TTS

Đọc [báo cáo nghiên cứu và code](../../docs/LOW_DATA_RESEARCH_REVIEW_20261105.md).

- `claim_source_review.csv`: 38 mục nguồn/claim; phân biệt full experiment, abstract, tham chiếu implementation và kết luận chưa chuyển giao được.
- `sources/`: URL, trạng thái tải, hash và metadata tiêu đề; không đóng gói PDF hoặc toàn văn abstract.
- `audio_duration_and_augmentation.json`: đo thời lượng WAV, thử speed và reverb bằng thư viện thật.
- `two_stage/`: lệnh, log, merge/ONNX parity, decoder smoke và đối chiếu encoder/decoder không đổi ở stage 2. Audio mang nhãn TTS trong smoke là audio gốc tái sử dụng làm fixture, không phải audio tổng hợp mới.
- `hotwords_comparison.csv`, `hotwords/`: 64 cặp audio validation, ba decoder cho mỗi ASR; đây là pilot phân tầng, chưa phải benchmark toàn validation hoặc calling.
- `qc_real_audio_baseline.json`: quy tắc QC áp lên audio gốc, cho thấy lỗi tương quan của hai ASR.
- `tts_label_strict_changes.json`: 91 nhãn thay đổi sau bỏ fuzzy relabel; không coi mọi thay đổi là lỗi nghĩa đã xác nhận.
- `actual_scaled_adam_lr.json`: bài fit ma trận 720 bước với optimizer thật; không phải thí nghiệm accuracy ASR. Chạy lại bằng `scripts/scaled_adam_lora_lr_check.py --icefall-dir PATH`.
- `qwen/`: bản hybrid thu nhỏ, tokenizer thật; test NEFTune, rsLoRA, callback early stopping và merge 0,5.
- `agent_sft/summary.json`: 15.767 / 2.937 / 3.176 bản ghi sau dedup; JSONL đầy đủ giữ cục bộ ở outputs của review.
- `unit_tests_final.log`: 145 test pass; profile CI có 27 skip thiếu thư viện model.

Không có trọng số model, audio hoặc token API trong thư mục này. `SHA256_MANIFEST.json` chứa checksum các file bằng chứng.
