# Khảo sát lỗi ASR trên toàn bộ train

> Cập nhật: đã tính lại WER theo dạng lời nói cho toàn bộ tập. Yêu cầu cuối: 15,13%; history: 15,34%. Các bảng WER/cờ chênh lệch dưới đây vẫn dùng text gốc; xem báo cáo mới tại `../asr_spoken_eval/BAO_CAO_SPOKEN_WER.md` và các CSV spoken để loại khác biệt số/chữ trước khi rà lỗi.

## Phạm vi và cách đo

- 1.545 audio yêu cầu cuối và 3.491 audio user trong history: đủ 5.036/5.036 transcript ASR đã lưu.
- So sánh `audio_text` của train với output GipFormer; không dùng transcript Qwen. Lấy transcript từ cache của benchmark full.
- Căn chỉnh từ bằng khoảng cách Levenshtein sau Unicode NFC, chuyển chữ thường và bỏ dấu câu. `substitution`: thay từ; `deletion`: thiếu từ so với nhãn; `insertion`: thêm từ so với nhãn. Một vùng có thể gồm nhiều phép sửa và có nhiều cách căn chỉnh tối ưu.
- Không nghe lại 5.036 audio và không xác minh từng nhãn, vì vậy các chênh lệch trong file là **ứng viên cần rà soát**, chưa phải toàn bộ lỗi nghe sai thực sự. Khác biệt phương ngữ, dấu thanh, cách viết, số và tên viết tắt có thể làm tăng WER.

| Lượt | Audio | Từ nhãn | Thay | Thiếu | Thừa | WER gốc |
|---|---:|---:|---:|---:|---:|---:|
| Yêu cầu cuối | 1.545 | 34.703 | 4.446 | 690 | 731 | 16,91% |
| History | 3.491 | 37.383 | 4.885 | 623 | 700 | 16,61% |

Chẩn đoán bổ sung: nếu chỉ đổi các số nguyên 0–999 đứng riêng thành chữ tiếng Việt ở cả hai phía, WER là 15,28% cho yêu cầu cuối và 16,16% cho history. **Đây không phải WER chính thức**: quy tắc chưa xử lý đầy đủ số điện thoại, số đầu 0, ngày giờ có dấu, số thập phân, đơn vị hay tên viết tắt. Kết quả không chứng minh mọi số đã được nhận đúng.

## Domain ở lượt yêu cầu cuối

Domain được suy từ app trong `label.tool_calls` — có thể không bao quát toàn bộ nội dung. Một câu có thể thuộc nhiều domain; các số lượng không cộng thành 1.545. Các câu không có nhãn tool được gom riêng.

| Nhóm | Audio | WER gốc | WER thử đổi số |
|---|---:|---:|---:|
| Media | 76 | 20,40% | 20,17% |
| Smart home (app) | 126 | 20,91% | 16,45% |
| Home nói chung | 305 | 17,76% | 15,94% |
| Di chuyển | 117 | 17,41% | 14,95% |
| Thông tin | 221 | 17,67% | 14,98% |
| Lập kế hoạch | 431 | 16,48% | 14,48% |
| Giao tiếp | 429 | 15,89% | 15,22% |
| Tài chính | 290 | 15,51% | 12,92% |

Mẫu theo domain có kích thước khác nhau; không xem các nhóm nhỏ là kết luận chắc chắn về độ khó. Xem mọi app và domain, cả history, trong `domain_summary.csv`.

## Các cụm nên ưu tiên kiểm tra

1. **Lệnh, phủ định và điều kiện.** Ví dụ `đừng → được`, `dời → giờ` (24 căn chỉnh một từ), `đặt → đọc` (14), `nhắn → nhấn` (17), `họp → học` (19). Đây là nhóm dễ đổi hành động hay nhánh điều kiện, nên nghe lại audio và ưu tiên đo riêng theo intent/slot.
2. **Viết tắt và mã.** `SJC → SC` xuất hiện 39 căn chỉnh một từ; còn các biến thể khác trong file chi tiết. `VTV3 → VTV BA`, `RON95 → RON CHÍN LĂM` có thể chỉ là cách viết tiếng nói, cần kiểm tra trước khi coi là lỗi nội dung.
3. **Số, thời gian và đơn vị.** Kiểm tra giờ, ngày, giá, số tiền, phần trăm, nhiệt độ và số điện thoại; đồng thời tách lỗi giá trị khỏi `50 → năm mươi` chỉ khác cách viết. File `review_dinh_dang_so.csv` chỉ nhận diện một phần trường hợp đơn giản.
4. **Tên riêng, địa chỉ, tiếng Anh và thương hiệu.** Ví dụ `Trúc Bạch → CHÚC BẠN`, `họp team → HỌC PHÍM`; nhóm heuristic tiếng Anh dùng từ điển hữu hạn nên có thể bỏ sót tên khác.
5. **Tên thuốc.** Nhóm tự động gắn cờ 2 audio: có `Omeprazole → OMEZON` và các đoạn liên quan `Brightpath`. Bộ phát hiện dựa trên tên trong `medication.add` hoặc mẫu `thuốc + Tên`; con số này **không phải** tổng số lỗi tên thuốc vì có thể bỏ sót.
6. **Khác biệt có thể không đổi nghĩa.** `hông → không` (99 căn chỉnh), `dùm → giùm` (52), `hoà → hòa` (74), `khoá → khóa` (59) đóng góp vào WER gốc; cần nghe/ngữ cảnh trước khi chuẩn hóa biến thể phương ngữ.

## Các file để làm việc

- `utterances.csv`: đủ 5.036 audio, transcript chuẩn và ASR, app/domain, WER, S/D/I, các cụm chênh lệch và đường dẫn audio. `turn=last` là yêu cầu cuối; `turn=history` là lượt user trước đó. `history_index` là chỉ số 0-based trong danh sách history gốc.
- `error_spans.csv`: 8.007 vùng chênh lệch để lọc các cụm sai; `reference_span` và `asr_span` là hai phía, trống là thiếu hoặc thừa. `error_type` có thể chứa nhiều loại nếu một vùng gồm nhiều phép sửa.
- `review_*.csv`: các tập lọc sẵn cho tiếng Anh/thương hiệu, viết tắt/mã, tên thuốc, tên riêng/địa danh, số/thời gian/đơn vị, phủ định/điều kiện và khả năng khác cách viết số. Các nhãn là **nghi vấn heuristic**, chồng lấp và không đầy đủ; đặc biệt số/thời gian bao gồm từ thông dụng liên quan thời gian, không chỉ số thực sự sai.
- `frequent_substitutions.csv`: tần suất thay thế 1 từ ↔ 1 từ. Không bao gồm đầy đủ lỗi nhiều từ; nên đọc cùng `error_spans.csv`.
- `domain_summary.csv`, `tag_summary.csv`, `summary.json`: bảng tổng hợp. Các lần xuất CSV dùng UTF-8 BOM.

Tái tạo từ thư mục dự án: `/home/pc/venv/bin/python asr_error_audit.py` (cần `num2words` cho phân tích số thử nghiệm). Toàn bộ dữ liệu và kết quả nằm local.
