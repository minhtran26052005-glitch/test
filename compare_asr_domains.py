"""Paired field/term preservation audit of saved ASR, without model inference.

This screens meaning-bearing terms rather than computing transcript WER.
Automatic mismatches are review candidates; reviewed examples are separate.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from asr_datagen_audit import CATEGORIES, FILLERS, medical_group, semantic_text
from asr_error_audit import align, get_calls

SOURCE = Path('outputs/zipformer_150m/asr_comparison.csv')
LEXICON = Path('outputs/asr_datagen_targets/lexicon.json')
OUT = Path('outputs/zipformer_150m/domain_comparison')
LABELS = {
    'thuoc_y_te': 'Thuốc, y tế, cữ uống',
    'lenh_hanh_dong': 'Lệnh và ý định hành động',
    'phu_dinh_dieu_kien': 'Phủ định, điều kiện, sửa ý',
    'thiet_bi_khong_gian': 'Thiết bị, phòng, không gian',
    'viet_tat_ma': 'Viết tắt, kênh, mã',
    'tieng_anh_cong_viec': 'Tiếng Anh, từ nghiệp vụ',
    'ten_nguoi_viet': 'Tên người Việt, quan hệ',
    'ten_nguoi_quoc_te': 'Tên người quốc tế',
    'dia_diem_to_chuc': 'Địa điểm, quán, tổ chức',
    'thoi_gian_quan_he': 'Ngày, giờ, quan hệ trước/sau',
    'tai_chinh_thanh_toan': 'Tài chính, phương thức thanh toán',
    'gia_tri_so': 'Giá trị số, đơn vị, chữ số',
    'ten_rieng_phat_hien_them': 'Tên riêng tự phát hiện — cần rà',
    'ten_lien_he_phat_hien_them': 'Tên liên hệ tự phát hiện — cần rà',
}

# Field-level language review of saved paired text. Selected examples explain
# mechanisms; this is not a random sample for estimating overall accuracy.
REVIEWS = [
    ('train_1474','last','','thuoc_y_te','Omeprazole','OMEZON: sai định danh','AMERAZON: sai định danh','cả hai còn sai','P1','Tên thuốc ngoại vẫn không được giữ. Cần tên thuốc trong câu thêm lịch, không chỉ đọc từ rời.'),
    ('train_1343','last','','thuoc_y_te','Clearwater Pharma','CIA WATER PHẠM MA: phiên âm sai tên gốc','CIA WATER PHẠM MA: phiên âm sai tên gốc','cả hai còn sai','P1','Hai model gần như giống nhau ở tên này; đổi ASR chưa giải quyết tên nhãn thuốc trong corpus.'),
    ('train_1430','last','','thuoc_y_te','kẽm Brightpath / thuốc Brightpath','CẢM BIẾN PART / VINFAST','mất tên kẽm / WINFRED','cả hai còn sai','P1','Định danh Anh bị đổi hoặc mất. Từ vựng là chuỗi trong dataset, không xác thực tên dược phẩm ngoài thực tế.'),
    ('train_1703','last','','thuoc_y_te','thuốc huyết áp','mất huyết áp','giữ thuốc huyết áp','ZipFormer tốt hơn','P1','Qwen đổi từ med_72 sang med_318 đúng thuốc, nhưng vẫn ghi sai giờ và thiếu nhắc khách; không kết luận cả mẫu đã đúng.'),
    ('train_1695','last','','thuoc_y_te','vitamin D','vitamin T','vitamin D','ZipFormer tốt hơn','P1','Sửa được chữ D. Lịch thuốc vẫn thành liều thuốc, calling vẫn sai; gen vitamin và câu lịch/cữ uống.'),
    ('train_1695','last','','thuoc_y_te','lịch thuốc','ly thuốc','liều thuốc','cả hai còn sai','P1','Danh từ chỉ lịch bị đổi; cần câu hỏi xem lịch rồi nhắc đúng giờ.'),
    ('train_1510','last','','thuoc_y_te','viên sắt','viên sạch','viên sắt','ZipFormer tốt hơn','P1','Sửa được tên viên sắt trong cùng câu nhiều loại thuốc.'),
    ('train_1510','last','','thuoc_y_te','tảo Hawaii / dầu cá','mất tảo Hawaii / YO CÁ','TẠO PHÁ HOẠI / YOU CÁ','cả hai còn sai','P1','Hai thành phần thuốc khác vẫn sai; không coi cả câu đã tốt chỉ vì viên sắt được sửa.'),
    ('train_1510','last','','thuoc_y_te','chưa uống','chưa uống','chú uống','ZipFormer kém hơn','P1','Mất phủ định trong trạng thái dùng thuốc. Còn từ quên hỗ trợ suy luận, nhưng marker chưa không còn được giữ.'),
    ('train_1414','last','','thuoc_y_te','cữ trưa / cữ sáng','cửa trưa / cửa sáng','cửa trưa / cửa sáng','cả hai còn sai','P1','Cần cữ uống cùng thao tác bỏ cữ trưa, giữ cữ sáng; không chỉ gen tên thuốc.'),
    ('train_0485','last','','thuoc_y_te','viên sắt','viên sắt','men sắt','ZipFormer kém hơn','P1','Zip sửa viên sắt ở train_1510 nhưng làm sai ở mẫu này; cần nhiều ngữ cảnh/giọng.'),
    ('train_1125','last','','thuoc_y_te','thuốc bao tử','thuốc bao tử','thuốc ai tưởng','ZipFormer kém hơn','P1','Lỗi mới về nhóm thuốc; đây là từ vựng dạ dày/bao tử.'),
    ('train_1181','last','','thuoc_y_te','lịch thuốc','là thuốc','lịch thuốc','ZipFormer tốt hơn','P2','Giữ được danh từ lịch; từ xóa ở cùng câu vẫn sai thành xá.'),
    ('train_1510','history','2','thuoc_y_te','thuốc bổ gan','thuốc bộ gan','thuốc bổ gan','ZipFormer tốt hơn','P2','Sửa tên nhóm thuốc ở history; phần ngày mua ở cuối câu lại sai. Không suy causal effect riêng cho history.'),
    ('train_1061','last','','thuoc_y_te','siro ho','siro','SIROHO','ZipFormer tốt hơn','không gen chỉ vì nối chữ','SIROHO giữ đủ âm/chữ siro ho; đã loại lỗi cách tách từ khỏi bảng sàng lọc.'),
    ('train_1334','last','','thuoc_y_te','Omega-3 đi','OMEGA BA ĐI','OMEGA BADY','biến thể cần rà','không tính chắc chắn mất số 3','BA và ĐI có dấu hiệu bị nối thành BADY; không gọi đây là ca chắc chắn mất giá trị 3. Qwen tiếp tục đổi thành Omega Body.'),
    ('train_1335','last','','thuoc_y_te','đánh dấu đã uống','đánh dấu đã uống','đến dấu để uống','ZipFormer kém hơn','P1','Đổi thao tác xác nhận đã uống; cả câu còn uống ... xong rồi nên có thể suy ra, nhưng cụm hành động bị hỏng.'),
    ('train_0392','last','','lenh_hanh_dong','đặt báo thức','đặt vào thức','đặt báo thức','ZipFormer tốt hơn','P2','ASR được sửa nhưng Qwen vẫn bỏ alarm.create; đây là ví dụ ASR tốt hơn chưa kéo theo calling đúng.'),
    ('train_0256','last','','lenh_hanh_dong','dời lịch','giờ lịch','giờ lại','cả hai còn sai','P1','Động từ dời rất yếu ở cả hai. Giờ/rời là biến thể dễ nhầm với ý định đổi lịch; cần câu nền có ngày/giờ.'),
    ('train_0039','last','','ten_nguoi_viet','Minh','mình','Minh','ZipFormer tốt hơn','P1','Sửa người nhận thành tên riêng trong khung nhắn Zalo.'),
    ('train_0133','last','','ten_nguoi_viet','Minh','mình','mình','cả hai còn sai','P1','Tên này không được sửa ở mọi ngữ cảnh; đối chiếu train_0039.'),
    ('train_1655','last','','ten_nguoi_quoc_te','Sophie Martin','SOPHI MA TIÊN','Sophie Martin','ZipFormer tốt hơn','P1','Một trong các tên quốc tế được giữ nguyên tốt hơn; vẫn cần nhóm tên ngoại đa dạng.'),
    ('train_1604','last','','ten_nguoi_quoc_te','Kevin Park','CA VINTA','Kevin Park','ZipFormer tốt hơn','P1','Cải thiện rõ định danh người nước ngoài ở mẫu này.'),
    ('train_1368','last','','ten_nguoi_quoc_te','Peter Hansen','BIT HAN SEN','PETER HAN SEN','ZipFormer tốt hơn','không gen chỉ vì Han Sen','Zip sửa được Peter; tách Hansen thành Han Sen không làm mất âm/chữ định danh. Không coi khác khoảng trắng là mất tên.'),
    ('train_1414','last','','ten_nguoi_quoc_te','Daniel Lee','DANI','DANIEL, thiếu Lee','cả hai còn sai','P1','Zip giữ tên gọi tốt hơn nhưng vẫn mất họ/tên phân biệt. Không coi đã nhận đủ Daniel Lee.'),
    ('train_0828','last','','dia_diem_to_chuc','Trúc Bạch','Chúc Bạn','Trúc Bạch','ZipFormer tốt hơn','P1','Log calling đổi từ địa chỉ 18 phố Hàng Cót sang 18 phố Trúc Bạch; đây là cải thiện trường địa chỉ quan sát được.'),
    ('train_1526','last','','dia_diem_to_chuc','The Lemon Tree Café','đường Lê Cà Phê','THE LEMANRY CÀ PHÊ','cả hai còn sai','P1','Zip gần âm hơn nhưng chưa giữ đầy đủ tên quán; không xếp đúng chỉ vì còn Café.'),
    ('train_0029','last','','viet_tat_ma','SJC','SJC','SGC','ZipFormer kém ở mặt chữ mã','P2','Qwen vẫn gọi gold_price.get_price(type=sjc) ở cả hai; đây chưa phải bằng chứng calling sai do lỗi mã.'),
    ('train_0100','last','','viet_tat_ma','SMS','mất SMS','SMS','ZipFormer tốt hơn','P1','Giữ lại chỉ định kênh nhắn tin trong câu thanh toán nhiều bước.'),
    ('train_0069','history','4','viet_tat_ma','SCTV9','SCTV CHÍNH','SCTV CHÍN','ZipFormer tốt hơn','P2','Giữ đúng mã kênh. Chín/chính khác với việc viết số 9 bằng chữ.'),
    ('train_0075','history','4','viet_tat_ma','RON95','RON CHÍN LĂM: tương đương 95','RON CHÍN LÂM','ZipFormer kém ở mặt chữ mã','P2','Dạng chín lăm là tương đương 95 và đã loại; chín lâm là dạng méo âm/chữ khác, có thể còn phục hồi theo miền xăng.'),
    ('train_0608','last','','tieng_anh_cong_viec','podcast','podcast','postcard','ZipFormer kém hơn','P1','Đổi từ chỉ nội dung nghe sang một từ Anh khác. Gen cặp podcast/postcard trong câu nghe, hàng chờ.'),
    ('train_1079','last','','tieng_anh_cong_viec','check-in','TRAKING','CHECK IN','ZipFormer tốt hơn','P1','Đúng cụm nghiệp vụ check-in; nối/gạch ngang không phải lỗi.'),
    ('train_1703','last','','tieng_anh_cong_viec','khách check-in','khách Y','khách RẤT YÊN NHIÊN','cả hai còn sai','P1','Code-switch khách check-in vẫn hỏng; cho thấy cùng một từ có cải thiện không đồng đều theo câu/giọng.'),
    ('train_1695','last','','tieng_anh_cong_viec','khách check-out sớm','KHÁCH CHIẾC ÁO','HẾT CHÍCH ÁO','cả hai còn sai','P1','Từ nghiệp vụ khách sạn trong câu y tế bị đổi; cần câu đa domain.'),
    ('train_0028','last','','tieng_anh_cong_viec','họp team','họp tim','họp team','ZipFormer tốt hơn','P2','Zip giữ đúng từ team; đây là cặp Việt-Anh với từ Việt gần âm.'),
    ('train_0373','last','','phu_dinh_dieu_kien','đừng bật','đùng bận','đừng bật','ZipFormer tốt hơn','P1','Giữ được marker không bật quạt; từ quạt ở cùng câu vẫn sai thành quẹt.'),
    ('train_1715','last','','phu_dinh_dieu_kien','đừng tắt tiếng','được tắt','được cắt thế','cả hai còn sai','P1','Nguy cơ đảo ý định. Không quy mọi lỗi calling của mẫu này cho từ đừng.'),
    ('train_0347','last','','phu_dinh_dieu_kien','trừ hóa đơn trên 500 nghìn','chưa cái nào trên...','trừ cái nào trên...','ZipFormer tốt hơn','P1','Giữ nhánh ngoại lệ; cần gen câu nếu/trừ/chưa thay vì từ rời.'),
    ('train_0332','last','','phu_dinh_dieu_kien','trừ qua tài khoản ngân hàng','trừ qua tài khoản','chuyển qua tài khoản','biến thể cần rà','không coi tự động đảo nhánh','Trong câu trả hóa đơn, chuyển qua ngân hàng vẫn có thể giữ ý dùng nguồn ngân hàng; từ trừ không luôn là ngoại lệ.'),
    ('train_0083','last','','thiet_bi_khong_gian','phòng khách','hai lần phong cách','một phòng khách, một phòng cách','ZipFormer cải thiện một phần','P2','Cùng câu có hai lần nhắc phòng: chỉ một lần được sửa. Không tìm thấy một cụm đúng rồi coi mọi lần đều đúng.'),
    ('train_0117','history','4','thiet_bi_khong_gian','phòng ngủ nhỏ','phòng hồ nhỏ','phòng ngủ nhỏ','ZipFormer tốt hơn','P2','Giữ cả từ phân biệt nhỏ; quan trọng khi có nhiều phòng ngủ.'),
    ('train_0004','last','','thiet_bi_khong_gian','phòng ngủ nhỏ','phòng ngủ nhỏ','phòng ngủ nhỏ','đối chứng lỗi bước sau','không gen cho lỗi ASR ở mẫu này','Cả hai ASR đã giữ đủ phòng ngủ nhỏ. Nếu normalizer/agent chọn sai phòng thì cần sửa bước sau.'),
    ('train_0164','last','','thoi_gian_quan_he','sáng mai','sáng mai','sáng nay','ZipFormer kém hơn','P1','Đổi ngày đặt báo thức; khác biệt có ý nghĩa thời gian, không phải ITN.'),
    ('train_0066','last','','thoi_gian_quan_he','tối mai','tối mai','tối nay','ZipFormer kém hơn','P1','Giờ có thể đúng nhưng sai ngày; gen cặp nay/mai trong cùng câu.'),
    ('train_0097','last','','thoi_gian_quan_he','8 giờ kém 15','tám giờ tối mười lăm','tám giờ tiếng mười lăm','cả hai còn sai','P1','Mất quan hệ kém. Không chỉ gen số 8 hoặc 15; phải giữ cả cụm giờ kém.'),
    ('train_0332','last','','tai_chinh_thanh_toan','tiền mạng','tiền mạng','tiền mặt','ZipFormer kém hơn','P1','Đổi loại hóa đơn thành một khái niệm thanh toán khác; có nguy cơ mất đối tượng cần trả.'),
    ('train_0209','last','','tai_chinh_thanh_toan','tiền nước','tiền nước','miễn phí nếu','ZipFormer kém hơn','P1','Mất loại hóa đơn, dù SMS được sửa tốt hơn ở cùng mẫu.'),
    ('train_0081','last','','tai_chinh_thanh_toan','tiền điện','tiền biết','tiền điện','ZipFormer tốt hơn','P2','Giữ được loại khoản tiền trong câu chuyển cho nhiều người.'),
    ('train_0006','last','','gia_tri_so','123 triệu','một trăm hai mươi ba triệu','một trăm hai triệu','ZipFormer kém hơn','P1','ASR mất mươi ba, chữ hiện tại biểu diễn 102 triệu. Qwen còn đưa 120 triệu vào content; cần tách lỗi ASR với lỗi suy diễn tiếp theo.'),
    ('train_0124','last','','gia_tri_so','60 phút','sáu mươi phút','sáu phút','ZipFormer kém hơn','P1','Mất mươi làm 60 thành 6. Đây là mất giá trị, không phải khác cách viết chữ số.'),
    ('train_0144','last','','gia_tri_so','40 phần trăm','bốn mươi phần trăm','bốn phần trăm','ZipFormer kém hơn','P1','Log gọi light_set brightness=4. Có quan sát lỗi trường giá trị đi vào tool.'),
    ('train_0194','last','','gia_tri_so','30 phút','ba mươi phút','ba phút','ZipFormer kém hơn','P1','Cặp tối thiểu 3/30; cần giữ thành phần mươi.'),
    ('train_0158','last','','gia_tri_so','25 độ','hai mươi lăm độ','hai độ','ZipFormer kém hơn','P1','Mất phần giá trị nhiệt độ; cần số kèm đơn vị trong câu thiết bị.'),
    ('train_0083','last','','gia_tri_so','30 phần trăm','năm mươi phần trăm','ba mươi phần trăm','ZipFormer tốt hơn','P1','Sửa đúng độ sáng 30; nhưng nhiệt độ 26 ở cuối câu lại bị Zip cắt còn hai.'),
    ('train_0083','last','','gia_tri_so','26 độ ở cuối câu','hai mươi sáu, thiếu đơn vị','hai, mất phần còn lại','ZipFormer kém hơn','P1','Lỗi mất cuối câu nhiều tác vụ; không coi cải thiện độ sáng đồng nghĩa mọi slot số đều tốt.'),
    ('train_0657','last','','gia_tri_so','09870011220','đủ chuỗi không chín tám bảy không không một một hai hai không','tháng bảy, thiếu một không','ZipFormer kém hơn','P1','Chuỗi điện thoại bị đổi từ/thất lạc chữ số. Gen chuỗi có nhiều số 0 và ranh giới nhóm.'),
    ('train_1085','last','','gia_tri_so','09832146755','đủ 11 chữ số','đủ 11 chữ số','đối chứng lỗi bước sau','không gen cho số này','Cả hai giữ đủ số. Bỏ sms.send ở bước sau không phải bằng chứng số điện thoại ASR sai.'),
    ('train_0256','last','','gia_tri_so','45 phút','bốn lăm phút: cùng giá trị 45','bốn năm phút: mơ hồ','không gộp thành lỗi số chắc chắn','cần quy ước nhãn và rà ngữ cảnh','Gip không sai giá trị 45. Zip là dạng mơ hồ 4-5/45; normalizer tiếp tục đổi thành 4. Không đếm chắc chắn sai giá trị chỉ bằng chuỗi này.'),
    ('train_0070','last','','gia_tri_so','15 phút','mười lăm phút','mười năm phút','biến thể số khôi phục được','không gen chỉ vì lăm/năm','Với đơn vị phút, mười năm có thể đọc lại cùng số 15; đã loại khỏi lệch bảo toàn cụm. Không quy thành đổi 15 sang một số khác.'),
]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows, fields=None):
    if fields is None:
        if not rows:
            raise ValueError(f'No schema for empty file: {path}')
        fields = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aligned_target(hypothesis, operations, start, end):
    # Keep the occurrence tied to its reference position, including internal
    # insertions. A matching phrase elsewhere must not hide a missing mention.
    selected = [(op, i, j) for op, i, j in operations
                if (op != 'insertion' and start <= i < end)
                or (op == 'insertion' and start < i < end)]
    return ' '.join(hypothesis[j][1] for op, _, j in selected if op != 'deletion')


def matches(target, heard):
    def equivalent(value):
        # These preserve the information needed downstream. Spacing alone in
        # SIROHO / Peter Han Sen is not loss of the medicine/person identifier.
        value = re.sub(r'\bmười năm (?=phút\b|độ\b|phần trăm\b|nghìn\b|triệu\b)', 'mười lăm ', value)
        words = [word for word in value.split() if word not in FILLERS]
        if target.startswith('sếp ') and words and words[0] == 'xếp':
            words[0] = 'sếp'
        words = [{'café': 'cafe', 'email': 'mail', 'mạ': 'mẹ'}.get(word, word) for word in words]
        return ''.join(words)
    return equivalent(target) == equivalent(heard)


DIGITS = {'không': 0, 'một': 1, 'mốt': 1, 'hai': 2, 'ba': 3, 'bốn': 4,
          'tư': 4, 'năm': 5, 'lăm': 5, 'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9}
NUMERIC = set(DIGITS) | {'mười', 'mươi', 'trăm', 'nghìn', 'triệu', 'tỷ', 'tỉ', 'lẻ', 'linh', 'phẩy'}


def cardinal_value(words):
    """Read explicit Vietnamese cardinal structure; reject bare 'bốn năm'."""
    if not words or words[0] not in DIGITS and words[0] != 'mười' or any(word not in NUMERIC for word in words) or 'phẩy' in words:
        return None
    total, group, pending = 0, 0, None
    for word in words:
        if word in DIGITS:
            if pending is not None:
                return None  # A digit sequence / shorthand is ambiguous here.
            pending = DIGITS[word]
        elif word in {'lẻ', 'linh'}:
            if pending is not None:
                return None
        elif word == 'trăm':
            group += (pending if pending is not None else 1) * 100
            pending = None
        elif word in {'mười', 'mươi'}:
            group += (pending if pending is not None else 1) * 10
            pending = None
        else:
            multiplier = {'nghìn': 1000, 'triệu': 1000000, 'tỷ': 1000000000, 'tỉ': 1000000000}[word]
            total += (group + (pending if pending is not None else 0)) * multiplier
            group, pending = 0, None
    return total + group + (pending if pending is not None else 0)


def numeric_difference(target, heard, source):
    if matches(target, heard):
        return 'preserved_value_or_equivalent_format'
    def number_part(value):
        value = value.replace('phần trăm', 'phần_trăm')
        numbers, uncertain = [], False
        for word in value.split():
            if word in NUMERIC:
                numbers.append(word)
            elif word in {'phút', 'giờ', 'tiếng', 'độ', 'phần_trăm', 'đồng', 'đô', 'lít'}:
                break
            else:
                uncertain = True
        return numbers, uncertain
    r, _ = number_part(target)
    h, uncertain = number_part(heard)
    if source == 'spoken_digit_sequence':
        return 'changed_or_missing_identifier_digit'
    if not h:
        return 'no_numeric_value_in_aligned_span_review_context'
    if r == h:
        return 'unit_or_alignment_boundary_only_review_context'
    if r and h and r[-1] == 'lăm' and len(h) == 2 and h[-1] == 'năm' and r[0] == h[0] and 'mươi' in r:
        return 'ambiguous_short_number_not_confirmed_value_change'
    if uncertain:
        return 'uncertain_spoken_number_review_context'
    a, b = cardinal_value(r), cardinal_value(h)
    if a is None or b is None:
        return 'uncertain_spoken_number_review_context'
    return 'same_numeric_value_different_units_or_surface' if a == b else 'different_explicit_numeric_value'


def transition(gip_bad, zip_bad):
    if gip_bad and not zip_bad:
        return 'fixed_by_zipformer'
    if zip_bad and not gip_bad:
        return 'new_in_zipformer'
    return 'both_flagged' if gip_bad else 'both_preserved'


def candidate_occurrences(reference_tokens, index):
    for start, word in enumerate(reference_tokens):
        for termkey, spec in index.get(word, []):
            end = start + len(spec['tokens'])
            if reference_tokens[start:end] != spec['tokens']:
                continue
            category = spec['category']
            previous = reference_tokens[start-1] if start else ''
            if category == 'gia_tri_so' and previous in {'ron', 'rôn', 'xăng', 'rv', 'vtv', 'htv', 'sctv', 'vov'}:
                continue
            if category == 'gia_tri_so' and previous in NUMERIC:
                continue  # 'ba triệu' inside 'một trăm hai mươi ba triệu'.
            if category == 'thoi_gian_quan_he' and spec['term'] == 'mai' and previous in {'cô', 'chị', 'anh', 'bà', 'ông', 'bạn', 'sếp', 'thầy'}:
                continue
            if category == 'thoi_gian_quan_he' and spec['term'] == 'mai' and reference_tokens[max(0, start-2):start] in [['cô', 'giáo'], ['thầy', 'giáo']]:
                continue
            if category == 'thuoc_y_te' and spec['term'] in {'uống rồi', 'chưa uống'} and not re.search(r'thuốc|vitamin|viên|kháng sinh|siro|dầu cá', ' '.join(reference_tokens)):
                continue
            yield start, end, termkey, spec


def reviewed_fields(lookup, cases, runs, roots):
    actual = {}
    def calls(model, sid):
        key = (model, sid)
        if key not in actual:
            result = runs[model][sid]
            value = result.get('predicted_tool_calls')
            if value is None:
                trace = roots[model] / 'samples' / sid / 'trace.jsonl'
                value = [{'name': event['name'], 'arguments': event['arguments']}
                         for event in read_jsonl(trace) if event.get('stage') == 'tool'] if trace.exists() else []
            actual[key] = json.dumps(value, ensure_ascii=False)
        return actual[key]
    rows = []
    for sid, turn, index, category, field, gip, zip_, verdict, priority, note in REVIEWS:
        source = lookup[(sid, turn, index)]
        gold = get_calls(cases[sid], int(index) if index else None)
        old, new = runs['gipformer'][sid], runs['zipformer'][sid]
        rows.append({'id': sid, 'turn': turn, 'history_index': index, 'category': category,
            'category_vi': LABELS[category], 'reviewed_field': field,
            'gipformer_field_review': gip, 'zipformer_field_review': zip_,
            'language_judgment': verdict, 'datagen_priority': priority, 'review_note': note,
            'reference': source['reference'], 'gipformer_asr': source['gipformer_asr'],
            'zipformer_asr': source['zipformer_asr'],
            'gipformer_normalized': old.get('normalized_text', '') if turn == 'last' else '',
            'zipformer_normalized': new.get('normalized_text', '') if turn == 'last' else '',
            'expected_calls_for_this_turn': json.dumps(gold, ensure_ascii=False),
            'gipformer_last_turn_calls': calls('gipformer', sid),
            'zipformer_last_turn_calls': calls('zipformer', sid),
            'calling_scope': 'last_turn_calls_only_history_causality_not_measured' if turn == 'history' else 'same_last_turn_saved_outputs_not_causal_proof',
            'audio': source['audio'], 'review_method': 'field_level_language_review_of_saved_text_and_calls_no_audio_or_counterfactual'})
    return rows


def build_datagen_plan():
    # Seeds below are grounded in the paired examples, not an invented list of
    # medicines or a claim about their medical use.
    groups = [
        ('P1','thuoc_y_te','Tên/nhãn thuốc tiếng Anh','Omeprazole; Clearwater Pharma; kẽm Brightpath; thuốc Brightpath','train_1474; train_1343; train_1430','Hai ASR vẫn phiên âm sai hoặc mất tên; cần tên đầy đủ trong câu thêm/xem/xóa lịch.'),
        ('P1','thuoc_y_te','Viên uống, chất bổ sung','viên sắt; tảo Hawaii; dầu cá; vitamin D; vitamin C','train_1510; train_0485; train_1695','Zip sửa viên sắt/vitamin D ở một số câu, nhưng chưa ổn định; giữ vitamin C làm đối chứng.'),
        ('P1','thuoc_y_te','Lịch, cữ và trạng thái đã/chưa uống','lịch thuốc; lịch uống thuốc; cữ trưa; cữ sáng; chưa uống; đã uống','train_1414; train_1510; train_1695; train_1335','Ưu tiên cặp cữ/cửa và đã/chưa; câu nhiều thuốc cần gắn đúng trạng thái cho từng thuốc.'),
        ('P1','thuoc_y_te','Tên nhóm thuốc theo bệnh/lời nói thường','thuốc bao tử; thuốc dạ dày; thuốc huyết áp; thuốc bổ gan','train_1125; train_1703; train_1510','Bao tử phát sinh lỗi mới. Huyết áp/bổ gan đã có cải thiện; dạ dày dùng kiểm tra giữ chất lượng.'),
        ('P1','gia_tri_so','Mất hàng chục hoặc thành phần tiền','60/6 phút; 30/3 phút; 40/4 phần trăm; 25/2 độ; 123 triệu','train_0124; train_0194; train_0144; train_0158; train_0006','Gen cùng câu nền, đổi giá trị tương phản; đọc đủ và đọc rút gọn có nhãn nhất quán. Tách bốn năm mơ hồ khỏi lỗi giá trị chắc chắn.'),
        ('P1','gia_tri_so','Chuỗi định danh nhiều số 0','09870011220; chuỗi số chia nhóm; nhắc lại/sửa một chữ số','train_0657; train_1085','Có ca Zip mất một số 0. Train_1085 là đối chứng cả hai đúng, không đưa vào tập lỗi.'),
        ('P1','thoi_gian_quan_he','Ngày tương đối và quan hệ giờ','sáng mai/sáng nay; tối mai/tối nay; giờ kém; giờ rưỡi; trước/sau','train_0164; train_0066; train_0097','Giữ ngày và quan hệ chứ không chỉ con số giờ; thêm bối cảnh sửa ý và mốc trong history.'),
        ('P1','tieng_anh_cong_viec','Khách sạn, giao dịch, lịch họp','check-in; check-out; invoice; confirm; cancel; họp team','train_1703; train_1695; train_1079; train_0028','Code-switch Việt-Anh; cùng từ trong câu khách sạn/y tế/lịch. Invoice vẫn không giữ đúng tên gốc ở các lượt có từ này.'),
        ('P1','tieng_anh_cong_viec','Tên loại nội dung nghe','podcast/postcard; tập podcast; hàng chờ nghe','train_0608; train_0491; train_0804','Zip thường đổi podcast thành postcard; gen hai từ trong ngữ cảnh phân biệt.'),
        ('P1','ten_nguoi_quoc_te','Tên người nước ngoài đầy đủ','Grace Nakamura; Daniel Lee; Sophie Martin; Kevin Park; Peter Hansen','train_1414; train_1655; train_1604; train_1368','Zip đã sửa được một số tên; vẫn thiếu thành phần hoặc méo tên khác. Không phạt Hansen/Han Sen chỉ vì khoảng trắng.'),
        ('P1','ten_nguoi_viet','Tên và từ gần âm','Minh/mình; Bống/bóng; Lan; Hương; Sếp Dũng','train_0039; train_0133; train_1130','Đặt trong câu gọi/nhắn/chuyển tiền. Xếp Dũng/Sếp Dũng cùng tên không tự coi là mất định danh.'),
        ('P1','dia_diem_to_chuc','Tên quán/tổ chức tiếng Anh','The Lemon Tree Café; Nexora Labs; Evergreen Foods; Team Design','train_1526; train_1626; train_1742','Tên trong câu tìm đường/lịch/chuyển tiền. Một số biến thể gần âm có thể khôi phục; chỉ gen sau khi rà đủ câu trong evidence.'),
        ('P1','lenh_hanh_dong','Động từ đổi ý định','dời lịch/giờ lịch; nhắn/nhắc; đặt báo thức/đọc báo; hủy/xóa','train_0256; train_0392','Giữ nhãn ý định và động từ; câu nhiều tác vụ. Không coi nhận đúng động từ là đủ nếu Qwen vẫn bỏ tool.'),
        ('P1','phu_dinh_dieu_kien','Phủ định, ngoại lệ, sửa ý','đừng/được; chưa/đã; nếu A thì B; trừ; à không; mà thôi','train_1715; train_0373; train_0347; train_1510','Cặp tối thiểu đảo hành động; phân biệt trừ ngoại lệ với trừ tiền, không gộp mọi từ trừ thành lỗi điều kiện.'),
        ('P1','tai_chinh_thanh_toan','Loại hóa đơn và nguồn thanh toán','tiền mạng/tiền mặt; tiền nước; tiền điện; ví điện tử; nạp ví','train_0332; train_0209; train_0081','Zip kém hơn ở một số loại hóa đơn dù tốt hơn về tiền điện; cần giữ tên đối tượng thanh toán.'),
        ('P2','viet_tat_ma','Chữ cái, mã sản phẩm/kênh','SJC/SGC; RON95; SMS; SCTV9; VTV3; HTV7','train_0029; train_0075; train_0100; train_0069','SMS/kênh có cải thiện, SJC/RON còn yếu. Chuẩn hóa nhãn chữ cái; chín lăm và 95 là cùng giá trị.'),
        ('P2','thiet_bi_khong_gian','Thiết bị và phòng phân biệt','phòng ngủ nhỏ; phòng ngủ; phòng khách; robot hút bụi; máy giặt','train_0117; train_0083; train_0004','Zip cải thiện rõ nhóm này. Giữ một phần data đối chứng, ưu tiên câu cuối dài bị mất nhiệt độ/đơn vị.'),
    ]
    fields = ['priority', 'category', 'generation_group', 'seed_terms', 'evidence_ids', 'generation_focus']
    return [dict(zip(fields, group)) for group in groups]


def write_report(summary, terms, reviews):
    lines = [
        '# So sánh ASR để chọn mục tiêu gen data', '',
        'Đối chiếu GipFormer 1.5 68M với hynt ZipFormer 150M trên cùng **5.036 audio** '
        '(1.545 lượt cuối + 3.491 lượt history), đọc output đã lưu; không chạy lại model.', '',
        '**Kết luận từ các trường đã rà:** ZipFormer cải thiện rõ thiết bị/phòng, sửa được một số tên/nghiệp vụ; '
        'chưa giải quyết tên thuốc ngoại và tên riêng khó. Có lỗi mới mất hàng chục, đổi ngày và mất phủ định đã/chưa uống. '
        'Vì vậy đổi model chưa thay thế được việc gen data theo nhóm thông tin.', '',
        '## Cách hiểu số liệu', '',
        'Bảng dưới đếm **audio có ít nhất một cụm thông tin sai/nghi sai** trong bộ thuật ngữ cố định đã khảo sát. '
        'Mẫu số là audio chứa cụm mục tiêu của nhóm đó, **không phải toàn bộ audio của lĩnh vực**. '
        'Hai model dùng cùng 726 thuật ngữ lấy từ reference, bao gồm cụm GipFormer trước đây nhận đúng để bắt lỗi mới của ZipFormer. '
        'Có 718 thuật ngữ thực sự khớp sau khi bỏ các trường hợp sai ngữ cảnh.', '',
        'Đã loại khác hoa/thường, dấu câu, số viết chữ/số, biến thể số nói tương đương, một số biến thể vùng miền, '
        'khác khoảng trắng (SIROHO/siro ho, Han Sen/Hansen), email/mail, Mạ/mẹ, xếp/sếp trước tên người. '
        'Cụm được căn theo **từng vị trí**: một lần phòng khách đúng không che lần khác sai trong cùng câu.', '',
        '**Đây là sàng lọc bảo toàn cụm, không phải tỷ lệ chính xác ngữ nghĩa đã chấm tay toàn tập hoặc % lỗi calling do ASR.** '
        'Tên phiên âm gần đúng có thể được Qwen/danh bạ khôi phục; lỗi có thể do căn biên. '
        'Các nhóm/cụm chồng lấp nên không cộng các dòng thành tổng audio.', '',
        '## Đối chiếu theo nhóm', '',
        '| Nhóm | Audio có mục tiêu | Gip sai/nghi sai | Zip sai/nghi sai | Hết cờ ở Zip | Mới có cờ ở Zip |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in summary['categories']:
        if row['scope'] != 'fixed_reference_term_inventory':
            continue
        lines.append(f"| {row['category_vi']} | {row['audio_with_reference_target']} | {row['gipformer_flagged_audio']} ({row['gipformer_flagged_percent']}%) | {row['zipformer_flagged_audio']} ({row['zipformer_flagged_percent']}%) | {row['fixed_audio']} | {row['new_error_audio']} |")
    lines += ['', 'Chi tiết riêng lượt cuối và history nằm trong category_summary.csv. Tên tự phát hiện từ chữ hoa/query danh bạ '
              'được lưu thêm trong CSV, cần rà danh tính trước khi dùng để gen.', '',
        '## Thuốc và y tế', '',
        'Trong 45 audio chứa 30 cụm y tế đã theo dõi: cờ nghi sai giảm 19 → 18; '
        '5 audio hết cờ nhưng 4 audio phát sinh cờ. Mẫu số nhỏ và một số cờ là biến thể gần âm, chưa thể suy rộng sang mọi tên thuốc.', '',
        '- **Vẫn sai ở cả hai:** Omeprazole, Clearwater Pharma, Brightpath, tảo Hawaii, dầu cá, cữ sáng/cữ trưa.',
        '- **Sửa được theo trường:** thuốc huyết áp (train_1703), vitamin D (train_1695), viên sắt (train_1510), thuốc bổ gan ở history.',
        '- **Lỗi mới cần ưu tiên:** viên sắt → men sắt (train_0485), thuốc bao tử → thuốc ai tưởng (train_1125), '
        'chưa uống → chú uống (train_1510), đánh dấu đã uống → đến dấu để uống (train_1335).',
        '- **Không tự gen như lỗi mất nội dung:** SIROHO vẫn giữ siro ho; OMEGA BADY có thể nối Omega ba đi, cần rà, không chắc chắn mất số 3.',
        '- **Giữ làm đối chứng:** thuốc dạ dày, kháng sinh, vitamin C, thuốc dị ứng và thuốc tiểu đường không có cờ hỏng cụm ở cả hai trong các lượt được quan sát.', '',
        'Tên nhãn như Clearwater Pharma/Brightpath là chuỗi trong corpus; báo cáo không xác thực thuốc, hoạt chất hay công dụng y khoa.', '',
        '| Cụm y tế | Lượt xuất hiện | Gip có cờ | Zip có cờ |', '|---|---:|---:|---:|',
    ]
    for row in terms:
        if row['category'] == 'thuoc_y_te':
            lines.append(f"| {row['reference_target']} | {row['mentions']} | {row['gipformer_flagged_mentions']} | {row['zipformer_flagged_mentions']} |")
    lines += ['', '## Các cơ chế sai cần gen', '',
        '1. **Thiết bị/phòng:** Zip giảm mạnh cờ 112 → 53 trên 420 audio có cụm mục tiêu. '
        'Phòng ngủ nhỏ, robot hút bụi, máy giặt tốt hơn; vẫn cần kiểm tra thuộc tính/nhiệt độ ở cuối câu dài.',
        '2. **Tên người:** Zip giữ tốt hơn Minh ở một số câu, Sophie Martin/Kevin Park/Peter Hansen ở một số mẫu. '
        'Minh → mình vẫn còn; Daniel Lee còn mất Lee; Grace Nakamura và nhiều tên khác vẫn bị méo. '
        'Không coi mọi phiên âm tên là mất danh tính; xem nguyên câu và danh bạ khi chọn data.',
        '3. **Địa điểm/tổ chức:** Trúc Bạch được sửa ở train_0828, calling cũng dùng lại đúng địa chỉ. '
        'The Lemon Tree Café còn thành THE LEMANRY CÀ PHÊ; Nexora Labs, Evergreen Foods, Team Design vẫn cần rà.',
        '4. **Viết tắt/mã:** tổng cờ không đổi 180/371, nhưng là các lỗi khác nhau: SMS và kênh TV có cải thiện, '
        'SJC → SGC và RON95 còn yếu. SJC → SGC vẫn được Qwen gọi type=sjc ở train_0029; không đồng nhất lỗi mặt chữ với lỗi calling.',
        '5. **Tiếng Anh:** team/check-in ở một số câu được sửa; invoice, check-out trong câu khách sạn và tên ngoại vẫn khó. '
        'Zip có lỗi podcast → postcard. Gen từ trong câu Việt và câu nhiều domain.',
        '6. **Lệnh/phủ định:** dời vẫn gần như luôn bị giờ/rời; cần nhắn/nhắc, đặt/đọc, đừng/được, đã/chưa và câu sửa ý. '
        'Từ trừ có thể là ngoại lệ hoặc trừ tiền: train_0332 đổi trừ qua ngân hàng thành chuyển qua ngân hàng chưa chắc đổi ý.',
        '7. **Giá trị thực và quan hệ thời gian:** ưu tiên 60 → 6 phút, 30 → 3 phút, 40% → 4%, 123 triệu → một trăm hai triệu, '
        '25 độ → hai độ, mất số 0 trong điện thoại, sáng mai → sáng nay. Đây khác với 45/bốn lăm hoặc chữ số/chữ viết.',
        '8. **Tài chính:** tiền điện có cải thiện, nhưng tiền mạng → tiền mặt/mất mạng, tiền nước → miễn phí nếu, '
        'nạp ví và tài khoản thanh toán có lỗi mới. Gen loại hóa đơn và nguồn tiền trong câu có nhánh đủ/thiếu số dư.', '',
        '## Tách lỗi số và lỗi bước sau', '',
        '- 45 phút ↔ bốn lăm phút giữ cùng giá trị; không đưa vào tập lỗi ASR chỉ vì khác cách nói.',
        '- Bốn năm phút mơ hồ 4–5/45: đánh dấu cần rà, không tự chốt thành sai giá trị. '
        'Mười năm phút trong ngữ cảnh này có thể khôi phục 15; đã loại khỏi cờ bảo toàn cụm.',
        '- Các cờ số trong bảng còn gồm mất đơn vị, mơ hồ và căn biên; numeric_review.csv tách từng loại. '
        'Không diễn giải toàn bộ chênh lệch cờ số thành lỗi đổi giá trị đã xác minh.',
        '- train_0004: cả hai ASR giữ phòng ngủ nhỏ; mất nhỏ sau đó thuộc normalizer/agent.',
        '- train_1085: cả hai giữ đủ 09832146755; bỏ SMS sau đó không phải lỗi chữ số ASR.',
        '- train_0392: Zip sửa đặt báo thức nhưng calling vẫn không tạo alarm; không gen tiếp ASR để sửa riêng việc agent bỏ tool này.', '',
        '## File để kiểm tra và gen data', '',
        f'- reviewed_examples.csv: **{len(reviews)} trường thông tin được rà ngôn ngữ**, có nguyên reference, hai ASR, normalized text và gold/actual calls. Đây là các ví dụ chọn có chủ đích, không dùng để ước lượng tỷ lệ toàn tập.',
        '- datagen_plan.csv: nhóm seed P1/P2 và lý do, dựa trên lỗi còn lại/mới phát sinh của Zip.',
        '- medical_comparison.csv: toàn bộ 30 cụm y tế, biến thể sai hai model và số lượt.',
        '- category_summary.csv: số lượng/tỷ lệ theo mẫu số cụm mục tiêu, riêng all/last/history.',
        '- target_comparison.csv và generation_priorities.csv: thống kê từng cụm; cờ tự động cần rà trước khi gen.',
        '- fixed_errors.csv / remaining_errors.csv / new_errors.csv: từng lần xuất hiện được sửa / vẫn có cờ / mới có cờ.',
        '- all_occurrences.csv: cả cụm hai model giữ đúng để đối chứng; unclassified_review.csv giữ các đoạn ngoài bộ thuật ngữ.', '',
        'Chạy lại báo cáo bằng /home/pc/venv/bin/python compare_asr_domains.py; chỉ đọc log/cache, không tải hay gọi model.',
    ]
    (OUT / 'BAO_CAO_SO_SANH_ASR.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    utterances = read_csv(SOURCE)
    lexicon = json.loads(LEXICON.read_text())
    cases = {row['id']: row for row in read_jsonl(Path('data/train+public_test/train/train.jsonl'))}
    old_root = Path('outputs/benchmark_seed42_n1545')
    new_root = Path('outputs/zipformer_150m/benchmark_seed42_n1545')
    runs = {name: {row['id']: row for row in read_jsonl(root / 'results.jsonl')}
            for name, root in [('gipformer', old_root), ('zipformer', new_root)]}
    by_first = defaultdict(list)
    for key, spec in lexicon.items():
        by_first[spec['tokens'][0]].append((key, spec))
    evidence, term_rows, utterance_rows, unclassified = [], [], [], []
    term_occurrences, category_audio = defaultdict(list), defaultdict(dict)
    lookup = {}
    for row in utterances:
        sid = row['id']
        uid = (sid, row['turn'], row['history_index'])
        if uid in lookup:
            raise ValueError(f'Duplicate utterance: {uid}')
        lookup[uid] = row
        ref = semantic_text(row['reference'])
        alignments = {}
        for name in ('gipformer', 'zipformer'):
            r, h, ops, _ = align(ref, semantic_text(row[name + '_asr']))
            alignments[name] = (h, ops)
        rt = [word for _, word in r]
        covered = set()
        local = defaultdict(lambda: [False, False, 0])
        for start, end, key, spec in candidate_occurrences(rt, by_first):
            category = spec['category']
            heard = {name: aligned_target(h, ops, start, end) for name, (h, ops) in alignments.items()}
            bad = {name: not matches(key, value) for name, value in heard.items()}
            status = transition(bad['gipformer'], bad['zipformer'])
            entry = {'id': sid, 'turn': row['turn'], 'history_index': row['history_index'],
                     'reference_position': start, 'category': category, 'category_vi': LABELS[category],
                     'reference_target': spec['term'], 'source': spec['source'],
                     'gipformer_target': heard['gipformer'], 'zipformer_target': heard['zipformer'],
                     'gipformer_flagged': bad['gipformer'], 'zipformer_flagged': bad['zipformer'],
                     'change': status, 'reference_context': ' '.join(rt[max(0, start-5):end+5]),
                     'reference': row['reference'], 'gipformer_asr': row['gipformer_asr'],
                     'zipformer_asr': row['zipformer_asr'], 'audio': row['audio'],
                     'potential_downstream_risk': CATEGORIES[category]['risk'],
                     'assessment': 'term_preservation_screen_not_verified_causal_error'}
            if category == 'gia_tri_so':
                entry['numeric_gipformer_assessment'] = numeric_difference(key, heard['gipformer'], spec['source'])
                entry['numeric_zipformer_assessment'] = numeric_difference(key, heard['zipformer'], spec['source'])
            else:
                entry['numeric_gipformer_assessment'] = entry['numeric_zipformer_assessment'] = ''
            evidence.append(entry)
            term_occurrences[key].append(entry)
            local[category][0] |= bad['gipformer']
            local[category][1] |= bad['zipformer']
            local[category][2] += 1
            covered.update(range(start, end))
        for category, (gip_bad, zip_bad, mentions) in local.items():
            category_audio[category][uid] = (gip_bad, zip_bad)
            utterance_rows.append({'id': sid, 'turn': row['turn'], 'history_index': row['history_index'],
                'category': category, 'category_vi': LABELS[category], 'target_mentions': mentions,
                'gipformer_flagged': gip_bad, 'zipformer_flagged': zip_bad,
                'change': transition(gip_bad, zip_bad), 'reference': row['reference'],
                'gipformer_asr': row['gipformer_asr'], 'zipformer_asr': row['zipformer_asr'], 'audio': row['audio']})
        # Keep omissions/substitutions outside the fixed vocabulary visible.
        for name, (h, ops) in alignments.items():
            blocks, block = [], []
            for op, i, j in ops:
                if op == 'equal':
                    if block:
                        blocks.append(block)
                        block = []
                else:
                    block.append((op, i, j))
            if block:
                blocks.append(block)
            for block in blocks:
                positions = [i for op, i, _ in block if op != 'insertion']
                if positions and all(i in covered for i in positions):
                    continue
                original = ' '.join(r[i][1] for op, i, _ in block if op != 'insertion')
                heard = ' '.join(h[j][1] for op, _, j in block if op != 'deletion')
                if set(original.split() + heard.split()) <= FILLERS:
                    continue
                unclassified.append({'id': sid, 'turn': row['turn'], 'history_index': row['history_index'],
                    'model': name, 'reference_span': original, 'asr_span': heard,
                    'reference': row['reference'], 'gipformer_asr': row['gipformer_asr'],
                    'zipformer_asr': row['zipformer_asr'], 'audio': row['audio']})
    for key, entries in term_occurrences.items():
        spec = lexicon[key]
        counts = Counter(e['change'] for e in entries)
        g_bad = sum(e['gipformer_flagged'] for e in entries)
        z_bad = sum(e['zipformer_flagged'] for e in entries)
        variants = {name: Counter(e[name + '_target'] or '[mất cụm]' for e in entries if e[name + '_flagged'])
                    for name in ('gipformer', 'zipformer')}
        term_rows.append({'category': spec['category'], 'category_vi': LABELS[spec['category']],
            'reference_target': spec['term'], 'source': spec['source'], 'mentions': len(entries),
            'audio_mentions': len({(e['id'], e['turn'], e['history_index']) for e in entries}),
            'gipformer_flagged_mentions': g_bad, 'zipformer_flagged_mentions': z_bad,
            'gipformer_flagged_percent': round(100*g_bad/len(entries), 2),
            'zipformer_flagged_percent': round(100*z_bad/len(entries), 2),
            'fixed_mentions': counts['fixed_by_zipformer'], 'new_error_mentions': counts['new_in_zipformer'],
            'both_flagged_mentions': counts['both_flagged'], 'both_preserved_mentions': counts['both_preserved'],
            'net_fewer_flags_zipformer': g_bad-z_bad,
            'gipformer_error_variants': json.dumps(variants['gipformer'].most_common(8), ensure_ascii=False),
            'zipformer_error_variants': json.dumps(variants['zipformer'].most_common(8), ensure_ascii=False),
            'generation_priority': 'review_new_regression' if counts['new_in_zipformer'] else ('still_needs_data' if z_bad else 'no_remaining_flag_for_term'),
            'generation_focus': CATEGORIES[spec['category']]['gen']})
    term_rows.sort(key=lambda r: (r['category'], -r['zipformer_flagged_mentions'], r['reference_target']))
    categories = []
    for category in CATEGORIES:
        for turn in ('all', 'last', 'history'):
            entries = [values for uid, values in category_audio[category].items() if turn == 'all' or uid[1] == turn]
            counts = Counter(transition(*values) for values in entries)
            g_bad, z_bad = sum(g for g, _ in entries), sum(z for _, z in entries)
            categories.append({'category': category, 'category_vi': LABELS[category], 'turn': turn,
                'audio_with_reference_target': len(entries), 'gipformer_flagged_audio': g_bad,
                'zipformer_flagged_audio': z_bad,
                'gipformer_flagged_percent': round(100*g_bad/len(entries), 2) if entries else '',
                'zipformer_flagged_percent': round(100*z_bad/len(entries), 2) if entries else '',
                'fixed_audio': counts['fixed_by_zipformer'], 'new_error_audio': counts['new_in_zipformer'],
                'both_flagged_audio': counts['both_flagged'], 'both_preserved_audio': counts['both_preserved'],
                'net_fewer_flags_zipformer': g_bad-z_bad,
                'scope': 'auto_discovered_review_candidates' if category.startswith('ten_') and 'phat_hien_them' in category else 'fixed_reference_term_inventory',
                'generation_focus': CATEGORIES[category]['gen']})
    write_csv(OUT / 'category_summary.csv', categories)
    write_csv(OUT / 'target_comparison.csv', term_rows)
    write_csv(OUT / 'medical_comparison.csv', [dict(medical_group=medical_group(r['reference_target']), **r) for r in term_rows if r['category'] == 'thuoc_y_te'])
    write_csv(OUT / 'all_occurrences.csv', evidence)
    flagged = [r for r in evidence if r['change'] != 'both_preserved']
    write_csv(OUT / 'evidence.csv', flagged)
    for name, status in [('fixed_errors.csv', 'fixed_by_zipformer'), ('new_errors.csv', 'new_in_zipformer'), ('remaining_errors.csv', 'both_flagged')]:
        write_csv(OUT / name, [r for r in evidence if r['change'] == status], list(evidence[0]))
    write_csv(OUT / 'utterance_categories.csv', utterance_rows)
    write_csv(OUT / 'numeric_review.csv', [r for r in evidence if r['category'] == 'gia_tri_so' and r['change'] != 'both_preserved'])
    write_csv(OUT / 'unclassified_review.csv', unclassified)
    priorities = [r for r in term_rows if r['zipformer_flagged_mentions']]
    priorities.sort(key=lambda r: (r['source'] != 'curated', -r['zipformer_flagged_mentions'], -r['new_error_mentions']))
    write_csv(OUT / 'generation_priorities.csv', priorities)
    reviews = reviewed_fields(lookup, cases, runs, {'gipformer': old_root, 'zipformer': new_root})
    write_csv(OUT / 'reviewed_examples.csv', reviews)
    write_csv(OUT / 'datagen_plan.csv', build_datagen_plan())
    summary = {'utterances': len(utterances), 'last': sum(r['turn'] == 'last' for r in utterances),
        'history': sum(r['turn'] == 'history' for r in utterances), 'reference_lexicon_terms': len(lexicon),
        'observed_reference_terms': len(term_rows), 'target_occurrences': len(evidence),
        'flagged_occurrences_in_either_model': len(flagged), 'transition_occurrences': dict(Counter(e['change'] for e in evidence)),
        'language_review_fields': len(reviews),
        'language_review_distinct_audio': len({(r['id'], r['turn'], r['history_index']) for r in reviews}),
        'language_review_sampling': 'purposive_field_examples_not_a_representative_accuracy_estimate',
        'numeric_review_classification': {name: dict(Counter(e['numeric_' + name + '_assessment'] for e in evidence if e['category'] == 'gia_tri_so')) for name in ('gipformer', 'zipformer')},
        'categories': [r for r in categories if r['turn'] == 'all'],
        'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [SOURCE, LEXICON, Path(__file__), Path('asr_spoken_text.py'), Path('asr_datagen_audit.py')]},
        'method': 'Same 726-term reference inventory and symmetric semantic_text normalization for both ASRs; occurrence-aligned phrase preservation, not WER. Also exclude spacing-only differences, mail/email, Mạ/mẹ, sếp/xếp before a person name, café/cafe, and unambiguous mười năm phút -> mười lăm phút.',
        'limits': ['Mismatches are semantic-risk candidates, not verified error rates or causal attribution.',
                   'Overlapping targets/categories; counts cannot be summed as unique utterances.',
                   'Names with phonetic renderings may still be resolvable downstream; review before generating data.',
                   'Numerical spelling, benign spelling/dialect aliases, and polite fillers are excluded by existing shared rules.',
                   'Fixed term inventory can miss other errors; unclassified spans are exported for review.',
                   'No model inference, counterfactual calling, or audio listening in this audit.']}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_report(summary, term_rows, reviews)
    print(json.dumps({key: value for key, value in summary.items() if key not in {'categories', 'limits'}}, ensure_ascii=False))
    for row in categories:
        if row['turn'] == 'all':
            print(row['category'], row['audio_with_reference_target'], row['gipformer_flagged_audio'], row['zipformer_flagged_audio'], row['fixed_audio'], row['new_error_audio'])


if __name__ == '__main__':
    main()
