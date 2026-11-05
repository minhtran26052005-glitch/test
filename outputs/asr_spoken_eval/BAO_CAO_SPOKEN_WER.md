# Kiểm tra lại WER theo dạng lời nói — full train

Đã tính lại trên cùng 1.545 yêu cầu cuối và 3.491 lượt user trong history (5.036 transcript cache). Không chạy lại ASR hay tool execution. WER cũ bao gồm khác biệt số/chữ và mã liền/tách.

| Phạm vi | WER chữ viết ban đầu | WER chuẩn hóa số về lời nói | Có thêm dạng rút gọn theo ngữ cảnh |
|---|---:|---:|---:|
| 1.545 yêu cầu cuối | 16,9063% | **15,1341%** | 15,0847% |
| 3.491 lượt history | 16,6065% | **15,3437%** | 15,2157% |
| Toàn bộ 5.036 lượt | 16,7508% | **15,2429%** | 15,1527% |

Sử dụng cột giữa làm số đo spoken cơ sở trong báo cáo này. Cột rút gọn là phép thử riêng: thêm các biến thể như `mươi tư/mươi bốn`, `RON chín lăm/RON chín mươi lăm`, `hai sáu độ/hai mươi sáu độ`. Quy tắc cố định được áp dụng ở cả reference và hypothesis; không chọn cách đọc theo kết quả nào khớp ASR hơn.

## Quy tắc

- Chuẩn hóa Unicode NFC, chữ thường, bỏ dấu câu khi tách từ như baseline cũ.
- Số nguyên được đổi sang cách đọc tiếng Việt; giữ giá trị, không bỏ số và không thay bằng token chung.
- Số đầu 0 và số điện thoại nhóm bằng dấu cách được đọc từng chữ số. Ví dụ `0983 214 6755` thành `không chín tám ba hai một bốn sáu bảy năm năm`.
- `VTV3 → vtv ba`, `HTV7 → htv bảy`, `RON95 → ron chín mươi lăm`, `E5 → e năm`. Prefix khác nhau vẫn là khác nhau: `SJC → SC` vẫn bị tính lỗi.
- Mã chuyến bay như `RV221 → rv hai hai một`; `Omega-3 → omega ba`; số vé sau từ vé đọc theo chữ số. Các mã chữ-số khác dùng cách đọc chữ số cố định và giữ nguyên phần chữ.
- Có xử lý số thập phân kiểu Việt, dấu chấm phân nhóm nghìn, `%`, `h`, ký hiệu tiền, số ngày d/m/yyyy và HH:MM[:SS] bằng cách đọc cố định. Các kiểu đọc khác và các biểu thức số phức tạp chưa bao phủ đầy đủ.
- Giữ các từ phủ định, hành động, tên người, tên thuốc và từ tiếng Anh. Không sửa `đừng → được`, `Minh → mình`, `Omeprazole → Omezon` hoặc `mai → nay`.
- Không đồng nhất phương ngữ (`hông/không`, `tui/tôi`) hay mọi cách phát âm tên chữ cái; những khác biệt này có thể vẫn bị tính vào WER.

## Mức ảnh hưởng của cách viết

- 481/5.036 lượt thay đổi ít nhất một bên sau chuẩn hóa số.
- 79 lượt chỉ khác cách viết sau khi áp dụng quy tắc này và nay khớp hoàn toàn: 16 yêu cầu cuối, 63 lượt history. Xem `format_only_utterances.csv`.
- Số phép sửa từ giảm ở 425 lượt, giữ nguyên ở 4.597 lượt và tăng ở 14 lượt. Tăng có thể xảy ra khi số/mã bị nhận sai thật: một mã viết gộp được mở thành nhiều từ ở reference.
- Tổng phép sửa: từ 12.075 / 72.086 từ tham chiếu xuống 11.111 / 72.893 từ tham chiếu sau chuẩn hóa. Đây là thay đổi cả tử số và mẫu số; không được diễn giải mức giảm WER thành tỷ lệ câu đúng hoặc tỷ lệ lỗi calling do ASR.
- 7.725 vùng chênh lệch còn lại được xuất theo spoken text. Cách căn chỉnh tối ưu có thể có nhiều nghiệm, nên các đoạn cụ thể là hỗ trợ rà soát.

## Minh họa kiểm tra đúng/sai chữ số

- `HTV7` và `HTV BẢY`: nay được coi là tương đương.
- `4 chỗ … 30 phút … 15 phút` và `bốn chỗ … ba mươi phút … mười lăm phút`: nay tương đương nếu những từ khác giống nhau.
- `09351708362` và `không chín ba năm một bảy không tám ba sáu hai`: tương đương.
- `09889990002` và `không chín tám tám chín chín không không hai`: vẫn khác vì thiếu chữ số.
- `30%` và `năm mươi phần trăm`: vẫn khác vì giá trị sai.

## Các file

- `summary.json`: số đo, số lỗi S/D/I, mẫu số, phiên bản quy tắc và hash mã nguồn normalizer.
- `utterances.csv`: đủ 5.036 lượt, reference/ASR gốc và spoken, WER cho từng chế độ.
- `normalization_changed.csv`: 481 lượt đã thay đổi dạng text, để duyệt tính hợp lý của phép chuẩn hóa.
- `format_only_utterances.csv`: 79 lượt không còn lỗi sau chuẩn hóa số.
- `error_spans.csv`: các đoạn còn khác nhau trong chế độ spoken cơ sở, kèm câu gốc và audio.
- `domain_summary.csv`: WER theo tool/app và domain ở lượt cuối hoặc history. Một lượt có thể thuộc nhiều domain, suy từ nhãn tool; không cộng các nhóm lại thành tổng mẫu.

Chạy lại từ thư mục dự án:

```sh
/home/pc/venv/bin/python evaluate_spoken_asr.py
/home/pc/venv/bin/python -m unittest discover -s tests -p 'test_asr_spoken_text.py' -v
```

Kết quả đã được kiểm tra độc lập bằng `jiwer` cho cả 3 chế độ và 2 nhóm audio. Toàn bộ 15 unit/smoke tests đạt. Quy tắc chỉ dùng trong đánh giá; báo cáo calling trước đó vẫn giữ nguyên vì không có inference hay execution mới.

Đây là WER chẩn đoán cục bộ, chưa phải điểm chấm chính thức hoặc transcript spoken do người nghe xác nhận. Cách đọc ngày giờ, số, tiếng địa phương và tên viết tắt vẫn có thể có nhiều biến thể. Các bảng ASR audit trước đó dùng WER chữ viết ban đầu; nên kiểm tra lại các cờ lỗi số/mã bằng bảng spoken này trước khi ưu tiên sửa model.
