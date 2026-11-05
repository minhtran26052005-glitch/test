# Kết quả tại snapshot 05/11/2026

Snapshot kết quả được chốt lúc **09:06 ngày 05/11/2026 (UTC+7)**. Bộ sinh dữ liệu ở máy nguồn tiếp tục chạy; các số đếm này thuộc bản chụp đã xuất.

## Agent benchmark trên toàn bộ train có nhãn

1.545 mẫu cuối, gồm 5.036 audio khi tính cả các lượt user history.

| Luồng | State match | Exact tool calls | Mẫu lỗi |
| --- | ---: | ---: | ---: |
| GipFormer + Qwen | 452 / 1.545 = 29,26% | 148 / 1.545 = 9,58% | 83 |
| ZipFormer + Qwen | 452 / 1.545 = 29,26% | 151 / 1.545 = 9,77% | 68 |
| Hai ASR, Qwen review cả câu và history | 464 / 1.545 = 30,03% | 149 / 1.545 = 9,64% | 72 |

Nguồn: [`outputs/dual_asr_qwen/comparison_summary.json`](../outputs/dual_asr_qwen/comparison_summary.json). Mẫu lỗi được tính là thất bại. State hash và exact calls là chỉ số local; state match không chứng minh mọi nội dung tự do đều đúng. Lượt dual có thêm review history, nên không quy mọi chênh lệch thành tác động riêng của ghép ASR.

WER GipFormer sau chuyển số về lời đọc đối xứng: **15,2429% trên 5.036 audio**; riêng lượt cuối 15,1341%. Đây là chỉ số chẩn đoán text, không phải tỷ lệ lỗi làm sai calling. Xem [`asr_spoken_eval/summary.json`](../outputs/asr_spoken_eval/summary.json).

## Ghép từng từ theo timestamp

Mỗi ASR lưu BPE token, thời điểm và log-probability. Bộ ghép căn từ theo nội dung/thời gian, bảo vệ các cụm số, phủ định, tên/thuốc; chọn một trong hai nguồn, không dùng reference để viết lại câu ở inference.

| Phạm vi | Lượt từ khoá | Gip: cờ lệch | Zip: cờ lệch | Ghép: cờ lệch |
| --- | ---: | ---: | ---: | ---: |
| Toàn bộ 5.036 audio | 10.805 | 2.346 | 2.168 | 1.717 |
| 829 audio phần test kỹ thuật | 1.765 | 415 | 384 | 308 |

Trên phần test kỹ thuật, có 207 lượt từ khoá mà một ASR giữ được; bộ ghép lấy được 192, khoảng 92,75%. Còn 15 ca chọn nguồn cần rà, và 293 ca cả hai nguồn đều bị cờ lệch.

Đây là **cờ bảo toàn từ khoá**, chưa phải độ chính xác ngữ nghĩa đã chấm tay cho toàn tập, cũng chưa phải hiệu quả calling. Fit/validation/test là 3.558 / 649 / 829 audio trong train; hội thoại và một số template số được gom cùng phân vùng, nhưng cả train đã được khảo sát trước đó. Chưa có benchmark full agent cho `timestamp_word`.

Nguồn và ví dụ sửa/hỏng theo từng từ: [báo cáo ghép từ](../outputs/timestamp_word_fusion/comparison/BAO_CAO_GHEP_TU.md), [keyword_summary.csv](../outputs/timestamp_word_fusion/comparison/keyword_summary.csv), [keyword_evidence.csv](../outputs/timestamp_word_fusion/comparison/keyword_evidence.csv).

## Sinh text bổ sung ASR

- Kế hoạch: 5.036 seed × 10 biến thể = **50.360 câu**.
- Snapshot: **1.098 seed thành công, 10.980 câu**, chia 5.490 hướng Gip và 5.490 hướng Zip.
- **26 seed lỗi kiểm tra** được lưu để rà; chưa hoàn tất toàn bộ corpus.
- Đây là text thô. Chưa TTS hoặc fine-tune hai ASR. `target_model` là nhãn nhóm dữ liệu, không bảo đảm câu có đủ tiếng Anh hoặc đúng ngữ nghĩa.

[`generated.csv`](../outputs/asr_datagen_api/generated.csv) đã được xuất lại từ đúng các result theo seed trong snapshot, thay vì lấy file gộp đang cập nhật dở ở máy nguồn.

## Rà chất lượng text

Đợt rà cố định trước đó dùng snapshot **7.270 câu / 727 seed**, không phải toàn bộ 10.980 câu hiện tại. Đã đọc 252 câu: 120 ngẫu nhiên và 132 chọn theo rủi ro.

| Chấm trên 120 câu ngẫu nhiên | Số câu | Tỷ lệ trong mẫu |
| --- | ---: | ---: |
| Giữ tốt ý nghĩa | 96 | 80% |
| Cần sửa nhẹ | 12 | 10% |
| Cần loại/viết lại | 12 | 10% |

Có lỗi đổi báo thức thành lịch hẹn, chuyển khoản thành thanh toán hoá đơn, đổi vị trí gần đây thành gần nhà, mất giá mua/bán hoặc nội dung nhắn. Đã tách 32 bản đọc hỏng: lẫn chữ Nga/Ả Rập, sai chữ trong SMS và thiếu thành phần VTV. Các gợi ý sửa là đề xuất riêng, không đánh dấu toàn bộ dữ liệu đã sạch.

Trong 60 câu có nội dung thuốc được rà, 57 giữ tốt các trường chính và 3 cần sửa nhẹ. Chưa thấy lỗi lớn ở tên thuốc/giờ/trạng thái đã-chưa uống trong 60 câu đó; mẫu còn hẹp và chưa nghe audio.

Xem [báo cáo chất lượng](../outputs/asr_datagen_quality_20261104/BAO_CAO_CHAT_LUONG.md), [252 câu đã chấm](../outputs/asr_datagen_quality_20261104/reviewed_examples.csv), [32 bản đọc hỏng](../outputs/asr_datagen_quality_20261104/confirmed_spoken_corruption.csv).

## Dữ liệu đầy đủ để debug

Repo chứa bảng đọc nhanh. [Release snapshot-2026-11-05](https://github.com/minhtran26052005-glitch/test/releases/tag/snapshot-2026-11-05) chứa archive toàn bộ outputs đã lưu: cache ASR, timestamp, quyết định ghép từ, trace Qwen/tool, request/prompt/response API, lỗi và pilot.

Archive có checksum SHA-256 và manifest theo file. Dữ liệu audio/devkit gốc, trọng số và API key không thuộc archive này.
