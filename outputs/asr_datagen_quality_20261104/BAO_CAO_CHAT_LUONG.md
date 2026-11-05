# Chất lượng text sinh để dựng dữ liệu ASR

**Kết luận: phần lớn câu có thể dùng làm nguyên liệu, nhưng chưa nên đưa toàn bộ vào TTS.** Có lỗi đổi tác vụ và lỗi trong `spoken_text` mà bộ kiểm tra cấu trúc hiện tại chưa bắt được.

Snapshot chụp lúc **22:46 ngày 04/11/2026, UTC+7**: 727 seed đã thành công, 7.270 biến thể. Luồng sinh dữ liệu tiếp tục chạy; báo cáo này dùng snapshot cố định, không suy rộng số đếm sang các câu sinh sau đó.

## Cách kiểm tra và kết quả

Đã kiểm tra cấu trúc toàn bộ 7.270 biến thể, rồi đọc và đối chiếu ngôn ngữ **252 câu** với reference, bản viết, bản đọc và ngữ cảnh/slot thay đổi tương ứng:

- **120 câu lấy ngẫu nhiên đều trên toàn bộ biến thể**, seed lấy mẫu 20261104. Đây là phần dùng để mô tả tỷ lệ trong mẫu.
- **132 câu chọn có chủ đích** để rà thuốc, giờ/số, phủ định, đổi ý, chữ viết tắt, model dự phòng và các chuỗi bất thường. Phần này cố tình tập trung ca khó; không cộng tỷ lệ của nó vào mẫu ngẫu nhiên.

| Chấm trên 120 câu ngẫu nhiên | Số câu | Tỷ lệ trong mẫu |
| --- | ---: | ---: |
| Giữ tốt ý nghĩa, chưa thấy vấn đề text cần sửa | 96 | 80% |
| Cần sửa nhẹ: diễn đạt, độ rõ của slot hoặc cách ghi bản đọc | 12 | 10% |
| Cần loại hoặc viết lại vì đổi/bỏ thông tin quan trọng | 12 | 10% |

Đây là nhận xét ngôn ngữ của trợ lý trên mẫu, chưa có người chấm độc lập. **Không phải xác nhận rằng chính xác 80% toàn bộ corpus đạt, cũng không phải độ chính xác ASR hoặc calling.** Các model sinh có số lượng mẫu và loại câu khác nhau; mẫu này chưa hỗ trợ xếp hạng chất lượng Gemini/Gemma.

## Những lỗi cần sửa trước TTS

### 1. Câu trôi chảy nhưng đổi tác vụ hoặc slot

| Biến thể | Nội dung gốc | Nội dung sinh | Vấn đề |
| --- | --- | --- | --- |
| `train_0143_last_v01` | Đặt **báo thức** 5 giờ sáng mai | Thiết lập một **lịch hẹn** 5 giờ sáng mai | Đổi alarm thành calendar |
| `train_0229_last_v10` | **Chuyển cho người** khoản tiền bằng tiền mạng | **Thanh toán tiền mạng cho người** | Chuyển khoản thành trả hoá đơn; thay tên theo kế hoạch là hợp lệ, đổi tác vụ thì không |
| `train_0221_history_02_v07` | Tìm bún bò **gần đây** | Tìm bún bò **gần nhà tôi** | Tự thay vị trí tìm kiếm |
| `train_0106_last_v09` | **Huỷ lịch rồi nhắn báo** cô giáo Mai | **Nhắn cô giáo Mai huỷ lịch** | Đổi người thực hiện thao tác huỷ |
| `train_0234_last_v07` | Điều kiện theo **giá bán** vàng | Điều kiện theo **giá** vàng | Mất phân biệt mua vào/bán ra |
| `train_0009_last_v02` | **Nạp điện thoại** cho một số cụ thể | **Chuyển vào số** đó | Động từ mới làm lẫn loại giao dịch |

Một số câu còn bỏ nội dung tin nhắn “đã đóng tiền”, bỏ điểm đến của chuyến xe hoặc chuyển “rồi” thành “đồng thời”. CSV ghi rõ lý do và các bản sửa đề xuất cho những lỗi lớn.

### 2. Phát hiện 32 bản đọc hỏng trong snapshot

Đây là **32 dòng cụ thể đã đọc và xác nhận có vấn đề**, không phải số lỗi tối đa của toàn bộ corpus:

| Vấn đề | Số dòng | Ví dụ |
| --- | ---: | --- |
| Từ Nga lọt vào câu Việt | 1 | `Nếu` trong text thành `Если` trong spoken_text |
| Chữ Ả Rập lọt vào cách viết SJC | 10 | `SJC` thành `ét جي xi` |
| SMS bị đổi chữ | 11 | 1 dòng `E-M-S`; 10 dòng `ét em ếch` |
| Thiếu thành phần của VTV | 10 | `VTV6` thành `V vê sáu` |

Nếu dùng các chuỗi này làm đầu vào TTS rồi giữ nhãn gốc, audio có thể không khớp transcript. `confirmed_spoken_corruption.csv` có đủ 32 dòng và cột **suggested_spoken_text**. Các gợi ý sửa được lưu riêng, chưa áp vào kết quả thô.

Các dạng `SMS` → `ét em ét`, `VTV6` → `vê tê vê sáu`, hoặc đổi chữ số thành lời đọc **không tự động bị tính là lỗi**. Một số cách ghi tên chữ còn cần nghe audio TTS trước khi kết luận phát âm đúng hay sai.

### 3. Làm phẳng câu đổi ý, giảm giá trị luyện miền yếu

Ví dụ `train_0002_last` có diễn tiến: “Nếu mai mưa thì huỷ lịch... **À mà thôi**, cứ để lịch đó, chỉ xem thời tiết”. Nhiều biến thể giữ đúng ý cuối nhưng viết lại thành yêu cầu thẳng: “Xem thời tiết, lịch thì không huỷ”.

Về ý định cuối, các câu này phần lớn vẫn đúng. Nhưng chúng làm mất **cấu trúc tự sửa lời** mà ASR cần có dữ liệu luyện. Cần giữ kiểu nói đổi ý trong một phần biến thể, đồng thời giữ các từ khó thật sự cần học như nhắn/nạp/dời/cữ uống thay vì luôn đổi sang từ dễ hơn.

### 4. Nhóm Zip chưa tập trung nhiều vào tiếng Anh

Theo kế hoạch chuẩn bị cho 727 seed đã sinh:

- 509 seed, khoảng **70%**, theo nhánh diễn đạt tiếng Việt, không chủ động thêm tiếng Anh.
- 164 seed theo nhánh giữ từ/tên/mã có sẵn.
- 54 seed theo nhánh thay tên người; có **270 biến thể thay tên nhưng chỉ 25 tên thay thế khác nhau**.

Phần tiếng Việt có thể dùng làm dữ liệu bổ trợ, nhưng nhãn `target_model=zipformer` chưa đồng nghĩa với một câu luyện tiếng Anh. Nếu mục tiêu là Zip chuyên tiếng Anh/tên thuốc/tên quốc tế, cần bổ sung thêm dữ liệu nhắm đúng các từ khó, và theo dõi tỷ lệ này riêng với dữ liệu tiếng Việt giữ năng lực chung. Đây không phải lý do để ép chèn các từ Anh gượng vào câu Việt.

## Những phần đang ổn

- Cả 7.270 dòng đều qua kiểm tra cấu trúc hiện hành; không có nhóm trùng hoàn toàn bản đọc trong cùng một seed.
- Có 7.220 bản đọc khác nhau sau chuẩn hoá so sánh: **50 bản dư do trùng giữa các seed**, khoảng 0,69%. Nếu dùng cùng voice và cấu hình TTS thì có thể tái sử dụng audio cho những bản trùng, giữ lại liên kết tới các seed.
- Đã đọc toàn bộ **60 câu có nội dung thuốc** trong snapshot, thuộc 6 seed: 57 câu giữ tốt thông tin, 3 câu cần sửa nhẹ về cách nói/lý do quên uống; chưa thấy lỗi lớn ở tên thuốc, giờ hoặc trạng thái đã/chưa uống trong 60 câu này. Mẫu thuốc này còn hẹp, không đại diện cho mọi thuốc/domain sau này.
- Các tên **Omeprazole, Clearwater Pharma**, các cụm **viên sắt, tảo Hawaii, dầu cá** và marker **chưa uống** trong các ví dụ được rà đều được giữ.
- `9 giờ kém 15` ↔ `8 giờ 45`, `8 giờ tối` ↔ `20 giờ`, chữ số ↔ số viết bằng lời là các dạng tương đương hợp lệ trong những ca đã đọc; không tính chúng thành lỗi số.

## Bộ kiểm tra hiện tại còn thiếu gì

Các cờ từ vựng chỉ hỗ trợ tìm ca cần đọc. Trên 12 câu ngẫu nhiên bị chấm lỗi lớn, cột `semantic_warnings` cũ chỉ gắn cờ **1 câu**. Ngược lại, có câu đúng vẫn bị cờ vì “nếu” được đổi thành “trường hợp” hoặc cách đọc số/mã khác bề mặt.

Việc cần ưu tiên trước khi dựng audio:

1. Chặn chữ không thuộc ngôn ngữ dự kiến và kiểm tra đủ các thành phần của chữ viết tắt trong bản đọc.
2. Giữ rõ loại tác vụ, vai trò mua/bán, người nhận, điểm đón/đến, phủ định và quan hệ thời gian. Đừng chỉ kiểm tra còn tên và còn số.
3. Với số và mã, dùng quy tắc đọc có kiểm soát; giữ tên Anh đúng chính tả và nghe thử TTS trên tập nhỏ. Không để LLM tự thay chữ bằng phiên âm sai.
4. Giữ một phần câu tự sửa lời/đổi ý và chủ động tăng các cụm ASR đang yếu.
5. Tách nhóm nhắm từ Anh/thuốc/tên riêng khỏi nhóm tiếng Việt bổ trợ khi lấy mẫu huấn luyện; không dùng nhãn model làm bằng chứng câu đã đúng mục tiêu.

## File để tự kiểm tra

- `reviewed_examples.csv`: 252 câu đã chấm, có reference/text/spoken_text, loại lỗi, giải thích, tool nguồn và gợi ý sửa.
- `passed_reviewed.csv`, `needs_edit_reviewed.csv`, `rejected_reviewed.csv`: các nhóm kết quả đã chấm.
- `confirmed_spoken_corruption.csv`: 32 bản đọc hỏng, kèm gợi ý sửa.
- `quality_summary.json`: số đếm của lần rà ngôn ngữ.
- `all_variants_screen.csv`: sàng lọc tự động toàn bộ 7.270 dòng; các cờ không đồng nghĩa với lỗi đã xác nhận.
- `snapshot.jsonl`, `snapshot_summary.json`: snapshot và checksum kết quả nguồn để tái kiểm tra.

Đánh giá này chỉ trên văn bản. Chất lượng audio và mức cải thiện ASR phải kiểm tra sau khi có audio và thử mô hình; chưa có phép đo đó trong báo cáo này.
