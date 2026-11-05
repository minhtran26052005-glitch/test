# Danh mục kết quả

Các thư mục trong Git chứa bảng và báo cáo để đọc trực tiếp. Log/cache chi tiết từng mẫu được lưu trong [Release](https://github.com/minhtran26052005-glitch/test/releases/tag/snapshot-2026-11-05), có thể khôi phục bằng `python scripts/fetch_results.py --extract` từ gốc repo.

| Mục đích | Thư mục/file |
| --- | --- |
| Benchmark Gip đầy đủ 1.545 mẫu | [`benchmark_seed42_n1545/`](benchmark_seed42_n1545/) |
| Benchmark Zip đầy đủ và đối chiếu Gip | [`zipformer_150m/`](zipformer_150m/) |
| Qwen review hai transcript, gồm history | [`dual_asr_qwen/`](dual_asr_qwen/) |
| Ghép từng từ theo timestamp, selector và so sánh keyword | [`timestamp_word_fusion/`](timestamp_word_fusion/) |
| Rà domain/thuốc/Anh/mã theo hai ASR | [`zipformer_150m/domain_comparison/`](zipformer_150m/domain_comparison/) |
| Từ/cụm ưu tiên gen data | [`asr_datagen_targets/`](asr_datagen_targets/) |
| Chẩn đoán WER sau chuẩn hoá số về lời đọc | [`asr_spoken_eval/`](asr_spoken_eval/) |
| Text API sinh hiện tại | [`asr_datagen_api/generated.csv`](asr_datagen_api/generated.csv) |
| Rà chất lượng và gợi ý sửa text | [`asr_datagen_quality_20261104/`](asr_datagen_quality_20261104/) |

Các file `results.jsonl` là kết quả theo sample của luồng tương ứng; file `summary.json` là tổng hợp. `asr_text`, `normalized_text`, `tool_calls` là ba phần khác nhau của pipeline. Các CSV đối chiếu có reference/nhãn để khảo sát offline; không đưa nhãn vào inference.

Các file top-level `progress.json`/log chép nguyên từ máy nguồn mô tả lúc đang chạy. Số đếm chốt của bản chia sẻ nằm trong [`SHARE_SNAPSHOT.json`](../SHARE_SNAPSHOT.json). File sinh text gộp đã được tái xuất từ các result có trong snapshot, nên có thể mới hơn bản progress thô.

Thư mục pilot/v1 nằm trong archive để giữ lịch sử; dùng `timestamp_word_fusion/` hiện hành cho kết quả ghép từ v2.
