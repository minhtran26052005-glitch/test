# Khảo sát nội dung ASR cần sửa để giảm lỗi calling

Mục tiêu: tìm **thông tin cần cho hành động/tool nhưng bị ASR thay, làm mất hoặc làm nhiễu**, rồi chọn từ/cụm và ngữ cảnh để gen data. Không lấy WER làm tiêu chí ưu tiên.

Đã quét 5.036 lượt ASR (1.545 yêu cầu cuối, 3.491 lượt history). Đã loại một số khác biệt vô hại về số, chính tả, phương ngữ và phiên âm rõ ràng. Có 200 mục tiêu trong các nhóm từ đã chọn; danh sách rộng hơn có 482 mục, gồm tên phát hiện tự động và mục y tế chưa ghi nhận hỏng. Có 17 ví dụ được đọc đối chiếu trực tiếp reference, ASR, normalizer và calling, bao gồm ví dụ đối chứng cần sửa bước sau.

**Số đếm tự động là ứng viên cần rà, không phải số calling sai do ASR.** Các nhóm có thể chồng lấp. Chưa nghe lại audio hoặc chạy đối chứng bằng transcript chuẩn. Với tiếng Anh, có trường hợp chỉ khác phiên âm mà agent vẫn hiểu; cần ưu tiên mất thông tin, đổi giá trị và đổi sang từ khác.

## Nhóm nội dung cần gen

| Nhóm | Dấu hiệu trong corpus | Thông tin downstream dễ hỏng | Hướng gen |
|---|---|---|---|
| Động từ/ý định | đặt báo thức → đặt vào thức/đọc báo; dời → giờ/rời; nhắn → nhấn; thư nháp → thư nhắc | Họ tool, bỏ hành động, gửi thay vì để nháp | Đặt/đọc, nhắn/nhắc, dời/xem; câu nhiều tác vụ, sửa ý cuối câu |
| Thuốc và cữ uống | Omeprazole → OMEZON; mất huyết áp; viên sắt → viên sạch; vitamin D → vitamin T; cữ → cửa | name, schedule_id, loại thuốc, cữ sáng/trưa/tối, đã/chưa uống | Tên/nhóm thuốc + lịch/cữ + đã/chưa uống, tên Anh trong câu Việt |
| Tên người Việt | Minh → mình; Bống → bóng; Sếp Trân → xếp chân | contact_id, query danh bạ, người nhận | Tên + quan hệ + gọi/nhắn/chuyển; các tên gần âm |
| Tên người quốc tế | Daniel Lee → Dani; Grace Nakamura → camera; Priya Sharma → visa sama | Người liên hệ/đối tác/tác giả | Họ tên trong câu Việt; Baker/Barker, Shah/Sharma; nhiều cách nhấn âm |
| Địa điểm/tổ chức | Trúc Bạch → Chúc Bạn; The Lemon Tree Café → đường Lê Cà Phê; Nexora Labs → Nesura Las | query, địa chỉ, pickup/dropoff, mốc tham chiếu | Tìm đường/đặt xe/giao hàng/chuyển tiền với tên đầy đủ |
| Mã/viết tắt | SJC → SC; SMS mất/đổi thành em; SCTV9 → SCTV chính; RV221 thiếu chữ số | Kênh, loại vàng/xăng, phương thức nhắn, mã chuyến | Mã kèm số, cách đọc chữ cái; thống nhất nhãn spoken |
| Tiếng Anh nghiệp vụ | check-in/out, invoice, voucher, workshop, book, họp team, release note | Ứng dụng, nội dung query, mốc tham chiếu | Câu Việt-Anh tự nhiên có tên và giờ, không chỉ từ đơn |
| Thiết bị/phòng | phòng khách → phong cách/khoảng cách; phòng ngủ → phòng ốc; máy giặt → máy gạch | device_id, room, type | Phòng ngủ/phòng ngủ nhỏ; phòng/tầng; chế độ máy |
| Giá trị/thời gian | 30 → 50; kém → tối; mai → nay; trước → chắc; mất rưỡi | brightness, temperature, amount, time/ngày | Số gần âm trong cùng cấu trúc; kém/rưỡi, trước/sau |
| Phủ định/điều kiện | đừng → được; mất không/chưa; đổi ý bị nhiễu | Đảo nhánh, làm hành động đã hủy | Nếu A thì B/còn không thì C; đã/chưa; à không/mà thôi |
| Tài chính | ví → phí; tiền điện/nước/mạng bị nhiễu | ewallet/banking, nguồn tiền, hóa đơn | Ví/ngân hàng, thanh toán/tiết kiệm, các loại hóa đơn |

Trong bộ từ đã chọn, dấu hiệu hỏng động từ/ý định có ở 490 lượt audio, thiết bị/phòng ở 112 lượt, viết tắt/mã ở 180 lượt, thuốc/cữ uống ở 19 lượt. Đây là tín hiệu trong corpus và từ điển khảo sát, không là tỷ lệ lỗi thực của cả ngành.

## Riêng y tế

18 từ/cụm có dấu hiệu hỏng ở 19 lượt audio. File medical_targets.csv giữ cả mục có lỗi và mục xuất hiện nhưng chưa thấy hỏng. Cột medical_group chỉ phân nhóm từ vựng corpus.

| Từ/cụm | Nhận dạng quan sát được | Vì sao cần để ý |
|---|---|---|
| Omeprazole | OMEZON | train_1474: medication.add dùng name=Omez thay Omeprazole |
| thuốc huyết áp | thuốc, mất huyết áp | train_1703: log_taken chọn med_72 thay med_318 |
| Clearwater Pharma | CIA WATER PHẠM MA | train_1343: log_taken chọn med_5021 thay med_318 |
| kẽm Brightpath / thuốc Brightpath | CẢM BIẾN PART / THUỘC VINFAST | train_1430: reminders chứa cảm biến/VinFast |
| vitamin D | vitamin T | train_1695: bỏ yêu cầu nhắc vitamin D |
| viên sắt / tảo Hawaii / dầu cá | viên sạch / mất tảo Hawaii / yo cá | train_1510: nhầm tập thuốc đã uống |
| cữ sáng / cữ trưa | cửa sáng / cửa trưa | Mất thông tin phân biệt cữ uống |
| lịch thuốc / lịch uống thuốc | là thuốc / loại thuốc / ly thuốc | Mất ý tra/sửa lịch thuốc |
| siro ho | siro, thiếu ho | Mất thành phần phân biệt |
| thuốc bổ gan | thuốc bộ gan | Đổi từ trong nhóm thuốc |
| thuốc say xe | đoạn thành để phúc xe | Mất loại thuốc trong nội dung yêu cầu |

Các cụm thuốc dạ dày, thuốc bao tử, thuốc tiểu đường, kháng sinh, thuốc dị ứng, thuốc long đờm, vitamin C, Omega-3 có xuất hiện nhưng bộ đối chiếu hiện chưa thấy bản thân cụm đó bị hỏng sau khi bỏ khác biệt vô hại. Phần khác của câu vẫn có thể sai. Không kết luận cả ngành y tế yếu hoặc cần mở rộng hàng trăm tên thuốc chỉ từ vài mẫu tên hiếm.

Các tên/nhãn thuốc được giữ nguyên theo corpus; báo cáo không xác minh hoặc hướng dẫn sử dụng thuốc.

## Bằng chứng có calling và các ví dụ đối chứng

- **train_1474:** tên thuốc sai trong ASR, rồi tool dùng tên sai. Cần gen tên thuốc trong câu.
- **train_0083:** yêu cầu độ sáng 30, ASR thành 50, light_set dùng 50. Cần gen tương phản giá trị.
- **train_0133:** Minh thành mình, agent bỏ phần nhắn Minh. Cần gen tên người/đại từ.
- **train_0097:** mất kém trong 8 giờ kém 15, nhắc được đặt 20:00 thay 19:45.
- **train_1526:** tên quán sai, maps.search_place dùng Lê Cà Phê.
- **train_0256 — riêng giá trị 45 không cần sửa ASR:** ASR nói bốn lăm phút; normalizer đổi thành bốn mươi, agent dùng 40. Mẫu có lỗi ASR khác, nhưng giá trị này vẫn đúng.
- **train_1085 — điện thoại đã đúng:** ASR nhận đủ 09832146755, agent bỏ SMS. Không cần gen để sửa số đó.
- **train_0004 — tên phòng đã đúng:** nhỏ bị normalizer bỏ, không phải ASR.
- **train_1510 — tách lỗi:** tên/tập thuốc bị ASR làm hỏng nhưng mười hai giờ rưỡi vẫn đủ. Tool dùng 12:00 là lỗi phía sau.
- **train_1715 — chỉ kết luận nguy cơ:** đừng thành được, nhưng calling quan sát không mute TV. Không khẳng định chính lỗi này gây calling sai.

Đây là bằng chứng nội dung và calling lệch tương ứng, chưa chứng minh sửa ASR thì agent chắc chắn đúng. Normalizer và agent còn có lỗi độc lập.

## Dùng các file

1. **priority_targets.csv:** 200 seed trong nhóm đã chọn. Xem category, asr_variants, examples, potential_downstream_risk, generation_focus. suspected_bad_audio là số lượt có nghi vấn trong các lượt chứa mục tiêu, không phải số calling sai do ASR.
2. **medical_targets.csv:** 30 mục y tế có phân nhóm. Lọc suspected_bad_mentions > 0 để lấy ứng viên gen; giữ các mục đúng làm đối chứng.
3. **reviewed_examples.csv:** 17 ví dụ đã đọc trực tiếp, có reference, ASR, Qwen normalization, tool kỳ vọng và thực tế.
4. **evidence.csv:** 2.366 lần đối chiếu mục tiêu có nghi vấn, kèm audio và calling. aligned_asr_target lấy từ căn chỉnh; cần đọc cả câu vì token ở biên có thể thuộc đoạn lân cận.
5. **targets.csv:** 482 mục rộng hơn gồm tên tự phát hiện và giá trị số; các mục review_discovered_entity cần rà ngữ nghĩa/phiên âm trước.
6. **generation_brief.csv:** bảng nhóm, rủi ro và cách gen.
7. **unclassified_review.csv:** 5.290 vùng còn khác chưa được từ điển/phép lọc bao phủ đầy đủ; có thể là đại từ/từ thường/phiên âm hoặc đoạn chồng lấp. Không mặc định tất cả là lỗi quan trọng.

Mỗi bằng chứng có id, lượt last/history, chỉ số history, đường dẫn audio. History cần được khảo sát vì tên hoặc mốc tham chiếu có thể nằm ở lượt trước; calling của history chưa được đo như tác vụ riêng.

## Chọn nội dung để sinh

- Gen tên/cụm trong nhiều khung câu thực tế: thêm/tra/xóa thuốc; gọi/nhắn/chuyển; tìm đường/đón trả. Lấy ngữ cảnh gốc từ evidence.
- Có cặp gần âm, giá trị tương phản, phủ định và đổi ý cuối câu; không chỉ lặp một tên trong một template.
- Giữ các ví dụ ASR đã đúng cùng domain làm đối chứng; tách bộ kiểm tra khỏi câu/tên đã sinh.
- Có biến thể vùng miền và tốc độ đọc, nhưng hiện chưa có nhãn người nói/giọng để kết luận giọng nào yếu hơn.
- Hiện chỉ lập danh sách; chưa gen audio, huấn luyện hoặc đổi model inference.

Tái tạo từ thư mục dự án: /home/pc/venv/bin/python asr_datagen_audit.py
