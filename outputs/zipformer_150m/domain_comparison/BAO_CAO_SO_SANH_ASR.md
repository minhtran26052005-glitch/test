# So sánh ASR để chọn mục tiêu gen data

Đối chiếu GipFormer 1.5 68M với hynt ZipFormer 150M trên cùng **5.036 audio** (1.545 lượt cuối + 3.491 lượt history), đọc output đã lưu; không chạy lại model.

**Kết luận từ các trường đã rà:** ZipFormer cải thiện rõ thiết bị/phòng, sửa được một số tên/nghiệp vụ; chưa giải quyết tên thuốc ngoại và tên riêng khó. Có lỗi mới mất hàng chục, đổi ngày và mất phủ định đã/chưa uống. Vì vậy đổi model chưa thay thế được việc gen data theo nhóm thông tin.

## Cách hiểu số liệu

Bảng dưới đếm **audio có ít nhất một cụm thông tin sai/nghi sai** trong bộ thuật ngữ cố định đã khảo sát. Mẫu số là audio chứa cụm mục tiêu của nhóm đó, **không phải toàn bộ audio của lĩnh vực**. Hai model dùng cùng 726 thuật ngữ lấy từ reference, bao gồm cụm GipFormer trước đây nhận đúng để bắt lỗi mới của ZipFormer. Có 718 thuật ngữ thực sự khớp sau khi bỏ các trường hợp sai ngữ cảnh.

Đã loại khác hoa/thường, dấu câu, số viết chữ/số, biến thể số nói tương đương, một số biến thể vùng miền, khác khoảng trắng (SIROHO/siro ho, Han Sen/Hansen), email/mail, Mạ/mẹ, xếp/sếp trước tên người. Cụm được căn theo **từng vị trí**: một lần phòng khách đúng không che lần khác sai trong cùng câu.

**Đây là sàng lọc bảo toàn cụm, không phải tỷ lệ chính xác ngữ nghĩa đã chấm tay toàn tập hoặc % lỗi calling do ASR.** Tên phiên âm gần đúng có thể được Qwen/danh bạ khôi phục; lỗi có thể do căn biên. Các nhóm/cụm chồng lấp nên không cộng các dòng thành tổng audio.

## Đối chiếu theo nhóm

| Nhóm | Audio có mục tiêu | Gip sai/nghi sai | Zip sai/nghi sai | Hết cờ ở Zip | Mới có cờ ở Zip |
|---|---:|---:|---:|---:|---:|
| Thuốc, y tế, cữ uống | 45 | 19 (42.22%) | 18 (40.0%) | 5 | 4 |
| Lệnh và ý định hành động | 1623 | 490 (30.19%) | 468 (28.84%) | 116 | 94 |
| Phủ định, điều kiện, sửa ý | 337 | 44 (13.06%) | 40 (11.87%) | 13 | 9 |
| Thiết bị, phòng, không gian | 420 | 112 (26.67%) | 53 (12.62%) | 64 | 5 |
| Viết tắt, kênh, mã | 371 | 180 (48.52%) | 180 (48.52%) | 42 | 42 |
| Tiếng Anh, từ nghiệp vụ | 344 | 130 (37.79%) | 111 (32.27%) | 46 | 27 |
| Tên người Việt, quan hệ | 331 | 102 (30.82%) | 92 (27.79%) | 27 | 17 |
| Tên người quốc tế | 84 | 84 (100.0%) | 74 (88.1%) | 10 | 0 |
| Địa điểm, quán, tổ chức | 54 | 48 (88.89%) | 44 (81.48%) | 5 | 1 |
| Ngày, giờ, quan hệ trước/sau | 1777 | 253 (14.24%) | 232 (13.06%) | 84 | 63 |
| Tài chính, phương thức thanh toán | 299 | 41 (13.71%) | 46 (15.38%) | 12 | 17 |
| Giá trị số, đơn vị, chữ số | 1076 | 99 (9.2%) | 170 (15.8%) | 27 | 98 |

Chi tiết riêng lượt cuối và history nằm trong category_summary.csv. Tên tự phát hiện từ chữ hoa/query danh bạ được lưu thêm trong CSV, cần rà danh tính trước khi dùng để gen.

## Thuốc và y tế

Trong 45 audio chứa 30 cụm y tế đã theo dõi: cờ nghi sai giảm 19 → 18; 5 audio hết cờ nhưng 4 audio phát sinh cờ. Mẫu số nhỏ và một số cờ là biến thể gần âm, chưa thể suy rộng sang mọi tên thuốc.

- **Vẫn sai ở cả hai:** Omeprazole, Clearwater Pharma, Brightpath, tảo Hawaii, dầu cá, cữ sáng/cữ trưa.
- **Sửa được theo trường:** thuốc huyết áp (train_1703), vitamin D (train_1695), viên sắt (train_1510), thuốc bổ gan ở history.
- **Lỗi mới cần ưu tiên:** viên sắt → men sắt (train_0485), thuốc bao tử → thuốc ai tưởng (train_1125), chưa uống → chú uống (train_1510), đánh dấu đã uống → đến dấu để uống (train_1335).
- **Không tự gen như lỗi mất nội dung:** SIROHO vẫn giữ siro ho; OMEGA BADY có thể nối Omega ba đi, cần rà, không chắc chắn mất số 3.
- **Giữ làm đối chứng:** thuốc dạ dày, kháng sinh, vitamin C, thuốc dị ứng và thuốc tiểu đường không có cờ hỏng cụm ở cả hai trong các lượt được quan sát.

Tên nhãn như Clearwater Pharma/Brightpath là chuỗi trong corpus; báo cáo không xác thực thuốc, hoạt chất hay công dụng y khoa.

| Cụm y tế | Lượt xuất hiện | Gip có cờ | Zip có cờ |
|---|---:|---:|---:|
| lịch thuốc | 10 | 6 | 5 |
| cữ sáng | 2 | 2 | 2 |
| Clearwater Pharma | 1 | 1 | 1 |
| Omega-3 | 1 | 0 | 1 |
| Omeprazole | 1 | 1 | 1 |
| chưa uống | 3 | 0 | 1 |
| cữ thuốc | 1 | 1 | 1 |
| cữ trưa | 1 | 1 | 1 |
| cữ uống buổi tối | 1 | 1 | 1 |
| dầu cá | 1 | 1 | 1 |
| kẽm Brightpath | 1 | 1 | 1 |
| lịch uống thuốc | 3 | 1 | 1 |
| thuốc Brightpath | 1 | 1 | 1 |
| thuốc bao tử | 1 | 0 | 1 |
| thuốc say xe | 2 | 1 | 1 |
| tảo Hawaii | 1 | 1 | 1 |
| viên sắt | 4 | 1 | 1 |
| đánh dấu đã uống | 1 | 0 | 1 |
| kháng sinh | 4 | 0 | 0 |
| siro ho | 1 | 1 | 0 |
| thuốc bổ gan | 1 | 1 | 0 |
| thuốc dạ dày | 7 | 0 | 0 |
| thuốc dị ứng | 1 | 0 | 0 |
| thuốc huyết áp | 1 | 1 | 0 |
| thuốc long đờm | 1 | 0 | 0 |
| thuốc tiểu đường | 1 | 0 | 0 |
| uống rồi | 2 | 0 | 0 |
| vitamin C | 4 | 0 | 0 |
| vitamin D | 3 | 1 | 0 |
| đơn thuốc | 2 | 0 | 0 |

## Các cơ chế sai cần gen

1. **Thiết bị/phòng:** Zip giảm mạnh cờ 112 → 53 trên 420 audio có cụm mục tiêu. Phòng ngủ nhỏ, robot hút bụi, máy giặt tốt hơn; vẫn cần kiểm tra thuộc tính/nhiệt độ ở cuối câu dài.
2. **Tên người:** Zip giữ tốt hơn Minh ở một số câu, Sophie Martin/Kevin Park/Peter Hansen ở một số mẫu. Minh → mình vẫn còn; Daniel Lee còn mất Lee; Grace Nakamura và nhiều tên khác vẫn bị méo. Không coi mọi phiên âm tên là mất danh tính; xem nguyên câu và danh bạ khi chọn data.
3. **Địa điểm/tổ chức:** Trúc Bạch được sửa ở train_0828, calling cũng dùng lại đúng địa chỉ. The Lemon Tree Café còn thành THE LEMANRY CÀ PHÊ; Nexora Labs, Evergreen Foods, Team Design vẫn cần rà.
4. **Viết tắt/mã:** tổng cờ không đổi 180/371, nhưng là các lỗi khác nhau: SMS và kênh TV có cải thiện, SJC → SGC và RON95 còn yếu. SJC → SGC vẫn được Qwen gọi type=sjc ở train_0029; không đồng nhất lỗi mặt chữ với lỗi calling.
5. **Tiếng Anh:** team/check-in ở một số câu được sửa; invoice, check-out trong câu khách sạn và tên ngoại vẫn khó. Zip có lỗi podcast → postcard. Gen từ trong câu Việt và câu nhiều domain.
6. **Lệnh/phủ định:** dời vẫn gần như luôn bị giờ/rời; cần nhắn/nhắc, đặt/đọc, đừng/được, đã/chưa và câu sửa ý. Từ trừ có thể là ngoại lệ hoặc trừ tiền: train_0332 đổi trừ qua ngân hàng thành chuyển qua ngân hàng chưa chắc đổi ý.
7. **Giá trị thực và quan hệ thời gian:** ưu tiên 60 → 6 phút, 30 → 3 phút, 40% → 4%, 123 triệu → một trăm hai triệu, 25 độ → hai độ, mất số 0 trong điện thoại, sáng mai → sáng nay. Đây khác với 45/bốn lăm hoặc chữ số/chữ viết.
8. **Tài chính:** tiền điện có cải thiện, nhưng tiền mạng → tiền mặt/mất mạng, tiền nước → miễn phí nếu, nạp ví và tài khoản thanh toán có lỗi mới. Gen loại hóa đơn và nguồn tiền trong câu có nhánh đủ/thiếu số dư.

## Tách lỗi số và lỗi bước sau

- 45 phút ↔ bốn lăm phút giữ cùng giá trị; không đưa vào tập lỗi ASR chỉ vì khác cách nói.
- Bốn năm phút mơ hồ 4–5/45: đánh dấu cần rà, không tự chốt thành sai giá trị. Mười năm phút trong ngữ cảnh này có thể khôi phục 15; đã loại khỏi cờ bảo toàn cụm.
- Các cờ số trong bảng còn gồm mất đơn vị, mơ hồ và căn biên; numeric_review.csv tách từng loại. Không diễn giải toàn bộ chênh lệch cờ số thành lỗi đổi giá trị đã xác minh.
- train_0004: cả hai ASR giữ phòng ngủ nhỏ; mất nhỏ sau đó thuộc normalizer/agent.
- train_1085: cả hai giữ đủ 09832146755; bỏ SMS sau đó không phải lỗi chữ số ASR.
- train_0392: Zip sửa đặt báo thức nhưng calling vẫn không tạo alarm; không gen tiếp ASR để sửa riêng việc agent bỏ tool này.

## File để kiểm tra và gen data

- reviewed_examples.csv: **60 trường thông tin được rà ngôn ngữ**, có nguyên reference, hai ASR, normalized text và gold/actual calls. Đây là các ví dụ chọn có chủ đích, không dùng để ước lượng tỷ lệ toàn tập.
- datagen_plan.csv: nhóm seed P1/P2 và lý do, dựa trên lỗi còn lại/mới phát sinh của Zip.
- medical_comparison.csv: toàn bộ 30 cụm y tế, biến thể sai hai model và số lượt.
- category_summary.csv: số lượng/tỷ lệ theo mẫu số cụm mục tiêu, riêng all/last/history.
- target_comparison.csv và generation_priorities.csv: thống kê từng cụm; cờ tự động cần rà trước khi gen.
- fixed_errors.csv / remaining_errors.csv / new_errors.csv: từng lần xuất hiện được sửa / vẫn có cờ / mới có cờ.
- all_occurrences.csv: cả cụm hai model giữ đúng để đối chứng; unclassified_review.csv giữ các đoạn ngoài bộ thuật ngữ.

Chạy lại báo cáo bằng /home/pc/venv/bin/python compare_asr_domains.py; chỉ đọc log/cache, không tải hay gọi model.
