# Kết quả ghép theo timestamp ở mức từ

Đã ghép đủ 5036 audio từ token/timestamp/log-probability gốc; cả hai lần nhận dạng khớp transcript benchmark cũ. Bộ ghép chọn từ/cụm tại từng vùng thời gian, có đổi nguồn trong cùng câu; không đưa hai câu cho Qwen viết lại.

Bộ chọn gồm hiệu chỉnh điểm từng model, điểm ngữ cảnh n-gram từ phần fit và chọn đường qua các vùng căn thời gian. Các cụm số, tên Latin, thuốc, phủ định và ngày tương đối được ràng buộc để không trộn mảnh tùy ý. Các lựa chọn đều có nguồn trong hai bản ASR.

## Phạm vi và cách đọc

Audio theo phân vùng: {'test': 829, 'fit': 3558, 'validation': 649}. Chỉ fit/validation dùng để hiệu chỉnh/chọn tham số; test được báo riêng. Tập train từng được khảo sát từ trước, nên đây là kiểm tra kỹ thuật có giữ lại dữ liệu, không phải benchmark mới chưa từng thấy.

Bảng đếm audio có cụm từ khóa sai/nghi sai; mẫu số chỉ gồm audio chứa từ khóa đã khảo sát. Loại khác hoa/thường, số/chữ tương đương và các alias đã công bố. Nhóm/cụm có thể chồng lấp. Tên phiên âm/căn biên còn cần rà bằng ngôn ngữ; không coi đây là % lỗi calling đã chứng minh.

## Toàn bộ audio

| Nhóm | Audio | Gip có cờ | Zip có cờ | Ghép có cờ |
|---|---:|---:|---:|---:|
| Thuốc, y tế, cữ uống | 45 | 19 | 18 | 17 |
| Lệnh và ý định hành động | 1623 | 490 | 468 | 377 |
| Phủ định, điều kiện, sửa ý | 337 | 44 | 40 | 31 |
| Thiết bị, phòng, không gian | 420 | 112 | 53 | 46 |
| Viết tắt, kênh, mã | 371 | 180 | 180 | 131 |
| Tiếng Anh, từ nghiệp vụ | 344 | 130 | 111 | 92 |
| Tên người Việt, quan hệ | 331 | 102 | 92 | 76 |
| Tên người quốc tế | 84 | 84 | 74 | 74 |
| Địa điểm, quán, tổ chức | 54 | 48 | 44 | 44 |
| Ngày, giờ, quan hệ trước/sau | 1777 | 253 | 232 | 182 |
| Tài chính, phương thức thanh toán | 299 | 41 | 46 | 32 |
| Giá trị số, đơn vị, chữ số | 1076 | 99 | 170 | 72 |

## Phần test giữ lại

| Nhóm | Audio | Gip có cờ | Zip có cờ | Ghép có cờ |
|---|---:|---:|---:|---:|
| Thuốc, y tế, cữ uống | 13 | 10 | 8 | 9 |
| Lệnh và ý định hành động | 252 | 84 | 85 | 65 |
| Phủ định, điều kiện, sửa ý | 48 | 7 | 7 | 5 |
| Thiết bị, phòng, không gian | 74 | 20 | 10 | 9 |
| Viết tắt, kênh, mã | 60 | 30 | 28 | 19 |
| Tiếng Anh, từ nghiệp vụ | 55 | 21 | 18 | 17 |
| Tên người Việt, quan hệ | 49 | 15 | 14 | 10 |
| Tên người quốc tế | 13 | 13 | 8 | 8 |
| Địa điểm, quán, tổ chức | 11 | 11 | 11 | 11 |
| Ngày, giờ, quan hệ trước/sau | 297 | 37 | 37 | 27 |
| Tài chính, phương thức thanh toán | 50 | 8 | 10 | 6 |
| Giá trị số, đơn vị, chữ số | 189 | 20 | 28 | 14 |

## Từng từ khóa và quyết định

- keyword_summary.csv: từng cụm, số lượt giữ/sai của ba nhánh, số sửa và lỗi mới so với từng ASR; có toàn tập và test.
- keyword_evidence.csv: từng lần xuất hiện, reference và đoạn cả ba nhánh, câu đầy đủ, tool kỳ vọng và file quyết định.
- medical_keywords.csv: toàn bộ cụm thuốc/y tế, không chỉ những tên đã có lỗi.
- fixed_keywords.csv / new_regressions.csv / remaining_errors.csv: lỗi được sửa / lỗi mới so với một ASR / lỗi còn lại.
- needs_selector_fix.csv: một ASR đã giữ đúng nhưng bộ ghép chọn sai; ưu tiên sửa bộ chọn, không tự quy thành thiếu data ASR.
- both_asr_data_candidates.csv: cả hai ASR còn có cờ, dùng rà/chọn domain cần gen data; vẫn cần kiểm tra biến thể phiên âm/căn biên.
- word_decisions.csv: timestamp, hai ứng viên, nguồn được chọn, điểm chọn, cờ chưa chắc và ràng buộc từng vùng.
- reviewed_examples.csv: 60 trường đã rà trước đây, thêm transcript ghép thực tế để kiểm tra.
- language_review.csv: các kết luận ngôn ngữ đã kiểm tra trực tiếp trên từ khóa và transcript ghép thực tế, gồm sửa được và chọn sai nguồn.
- transcripts.csv: đủ 5.036 audio, reference và ba transcript, phân vùng và nguồn từ.

Có 1288 audio lấy các vùng bất đồng từ cả hai model trong cùng audio. Số này chỉ chứng minh có ghép từ, không phải số audio được sửa đúng.

## Kết luận ở các ca đã rà

- train_0001 thuộc test: ghép lấy một tiếng từ Zip (Gip nghe một chuyến), đồng thời lấy Sếp Dũng từ Gip (Zip nghe Xích Dũng). Đây là bù lỗi ở hai vị trí trong cùng một audio.
- train_0280 thuộc test: Gip giữ nhắn, Zip giữ SMS, ghép được nhắn SMS; tên Minh trong câu đó vẫn sai thành mình.
- Các ví dụ số được giữ từ Gip: 123 triệu, 60 phút, 40 phần trăm; các mẫu minh họa này thuộc phần fit. Hiệu quả trên test được thể hiện ở bảng riêng phía trên.
- Chưa giải quyết tên Omeprazole, Clearwater Pharma, Brightpath khi cả hai nguồn sai.
- Bộ chọn vẫn sai: train_1510 và train_0485 chọn sai viên sắt theo hai hướng khác nhau; train_1703 bỏ huyết áp dù Zip giữ được; train_0083 chọn sai 50 dù Zip giữ 30.
- Thuốc trên phần test chỉ có 13 audio chứa cụm mục tiêu: fusion còn 9 audio có cờ so với Zip 8. Không kết luận cải thiện thuốc nói chung từ con số toàn tập 19/18/17.
- Sàng lọc có dương tính giả: train_0963 bị coi là sửa à không do quy tắc bỏ từ đệm, nhưng chưa xác minh giữ đúng ý sửa câu. Xem ghi chú ngôn ngữ trong language_review.csv.
- Trong quá trình rà đã sửa lỗi cấu trúc căn từ: timestamp phát token có thể lệch hơn một giây dù cùng từ. Bản trước ngưỡng cứng tạo lặp từ; bản v2 dùng mốc từ giống nhau với chi phí thời gian mềm. Bản chẩn đoán cũ lưu riêng tại outputs/timestamp_word_fusion_v1/.

Kết quả dừng ở ASR ghép trước normalization. Chưa chạy lại tool calling cho nhánh timestamp; không suy tỷ lệ calling từ cờ từ khóa.
