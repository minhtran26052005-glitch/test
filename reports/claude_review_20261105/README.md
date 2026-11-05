# Kết quả review CPU

Đọc [báo cáo và các lỗi đã sửa](../../docs/CLAUDE_REVIEW_20261105.md).

- `unit_tests_final.log`: 104 test pass trong venv có thư viện model.
- `unit_tests_ci_profile.log`: profile CI không có thư viện model, 13 test skip.
- `real_audio_training_step.json`, `asr_cli_final.json`: checkpoint thật, audio thật, backward/update và CLI train nhỏ.
- `*_export_merge.json`, `sherpa_smoke_summary.json`: export với adapter tạo riêng để smoke test; không phải kết quả cải thiện sau train.
- `tiny_qwen_*`: train/merge bản Qwen hybrid thu nhỏ dùng tokenizer thật; không chạy Qwen 4B đầy đủ.
- `random_feature_stress_failed.json`: phép stress âm tính đã phát hiện output không hữu hạn. Không tính là pass.
- `split_data_check.json`, `data_integrity_final.json`, `agent_sft_summary.json`: kiểm tra split, nhãn, vocab và số lượng data.
- `artifact_manifest.json`: checksum bản nén SFT và từng file bên trong.

Không có audio, token API hoặc trọng số model trong thư mục này. Data SFT đầy đủ nằm trong `../../artifacts/claude_review_20261105/agent_sft.tar.gz` (khoảng 39 MiB).
