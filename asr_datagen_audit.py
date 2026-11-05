"""Find meaning-bearing ASR corruptions for targeted data generation.

Counts are semantic-risk candidates, not experimentally established causes.
Labels and names are used only in offline analysis.
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from asr_error_audit import align, get_calls
from asr_spoken_text import spoken_text

BASE = Path('outputs/benchmark_seed42_n1545')
DATA = Path('data/train+public_test/train')
OUT = Path('outputs/asr_datagen_targets')

# Fixed benign variants; never correct names, commands, numbers or negation.
EQUIVALENT = {
    'hoà': 'hòa', 'khoá': 'khóa', 'hoá': 'hóa', 'huỷ': 'hủy', 'xoá': 'xóa',
    'khoẻ': 'khỏe', 'toà': 'tòa', 'thuỷ': 'thủy', 'tui': 'tôi', 'hông': 'không',
    'dùm': 'giùm', 'ngàn': 'nghìn', 'bẩy': 'bảy', 'ni': 'nay',
}
FILLERS = set('ờ ừ ờm ừm à ạ nhé nha nhá nghen nghe hen nè giùm với ơi'.split())
CATEGORIES = {
    'thuoc_y_te': {
        'risk': 'Sai thuốc/cữ uống: chọn nhầm thuốc, nhầm lịch uống hoặc mốc nhắc.',
        'terms': 'Omeprazole|Clearwater Pharma|kẽm Brightpath|thuốc Brightpath|viên sắt|tảo Hawaii|dầu cá|thuốc huyết áp|thuốc tiểu đường|thuốc dạ dày|thuốc bao tử|thuốc bổ gan|siro ho|kháng sinh|thuốc dị ứng|thuốc long đờm|thuốc say xe|vitamin C|vitamin D|Omega-3|lịch thuốc|lịch uống thuốc|cữ thuốc|cữ sáng|cữ trưa|cữ uống buổi tối|đơn thuốc|uống rồi|chưa uống|đánh dấu đã uống',
        'gen': 'Tên thuốc/nhóm bệnh + lịch/cữ uống + đã/chưa uống; tên Anh xen câu Việt. Đây là từ vựng corpus, không suy ra đơn thuốc y khoa.'},
    'lenh_hanh_dong': {
        'risk': 'Đổi ý định, bỏ hành động hoặc chọn sai họ tool.',
        'terms': 'đặt báo thức|dời lịch|nhắn tin|nhắn SMS|gửi email|soạn nháp|thư nháp|để nháp|trả lời mail|gửi tiết kiệm|đặt xe|đổi điểm|đặt nhắc|đặt|dời|nhắn|nhắc|hủy|xóa|ngưng|dừng|giao việc|đánh dấu|tạo lại|bật|tắt|hạ|giảm|tăng',
        'gen': 'Đặt báo thức/đọc báo; dời/xem lịch; nhắn/nhắc; hủy/giữ nguyên. Câu nhiều tác vụ, sửa ý cuối câu và giọng vùng miền.'},
    'phu_dinh_dieu_kien': {
        'risk': 'Đảo nhánh điều kiện hoặc thực hiện hành động đã hủy.',
        'terms': 'đừng|không cần|khỏi bật|đừng bật|không bật|chưa giao|giao rồi|chưa trả|đã trả|à không|mà thôi|không cần nữa|để nguyên|giữ nguyên|trừ|nếu|còn không|chỉ giữ',
        'gen': 'Đừng/được; đã/chưa; nếu A thì B còn không thì C; à không; mà thôi khỏi. Phải giữ rõ nhánh/ý cuối cùng.'},
    'thiet_bi_khong_gian': {
        'risk': 'Nhầm thiết bị/phòng, sai device_id hoặc thiếu hành động.',
        'terms': 'phòng khách|phòng ngủ|phòng ngủ nhỏ|phòng làm việc|tầng hai|đèn bếp|điều hòa|máy giặt|bình nóng lạnh|cửa kho|cửa sau|cửa chính|camera gara|đồ len|giặt nhanh|robot hút bụi',
        'gen': 'Thiết bị + phòng/tầng + thuộc tính; phòng ngủ/phòng ngủ nhỏ, cửa chính/cửa kho, bật/tắt và nhiệt độ/độ sáng.'},
    'viet_tat_ma': {
        'risk': 'Sai sản phẩm/kênh/mã chuyến. Bỏ qua số viết gộp và dạng spoken tương đương.',
        'terms': 'SJC|SMS|RON95|E5|VTV1|VTV3|VTV6|HTV7|SCTV9|VOV|IELTS|KTX|QR|SQL|A4',
        'gen': 'Tên chữ cái Việt/Anh và mã kèm số, xen câu Việt; thống nhất dạng transcript. HTV7/HTV bảy phải cùng giá trị.'},
    'tieng_anh_cong_viec': {
        'risk': 'Mất từ khóa nghiệp vụ: sai ứng dụng hoặc nội dung truy vấn.',
        'terms': 'team|họp team|check-in|check-out|workshop|voucher|invoice|email|mail|release note|book|confirm|cancel|online|offline|Speaking Club|Python|Excel|Zalo|TalkAI|croissant|podcast|homestay|work from home|add remind',
        'gen': 'Code-switch Việt-Anh: họp team, gửi invoice, book lớp, khách check-in/out; chèn tên và giờ, không chỉ đọc từ rời.'},
    'ten_nguoi_viet': {
        'risk': 'Nhầm người nhận/khóa tra danh bạ khi gọi, nhắn hoặc chuyển tiền.',
        'terms': 'Minh|Lan|Hương|Phượng|Bống|Sếp Dũng|Sếp Trân|Trần Văn Tuấn|Lê Minh|Nguyễn Thị Lan|Mạ|Bác Ba|Cô Hạnh',
        'gen': 'Tên + quan hệ + gọi/nhắn/chuyển; Minh/mình, Hương/Hường, Bống/bóng; tên trùng cần từ phân biệt.'},
    'ten_nguoi_quoc_te': {
        'risk': 'Sai tên người liên hệ, tác giả hoặc đối tác; nhầm tên gần âm.',
        'terms': 'Nina Kowalski|Grace Nakamura|Sophie Martin|Laura Costa|Sarah Kim|Priya Sharma|Priya Shah|Marcus Hale|Oliver Grant|Peter Hansen|Julian Ward|James Wong|Daniel Lee|Kevin Tran|Kevin Park|Nathan Reed|Meera Kapoor|Jonas Berg|Lucas Bennett|Ben Taylor|Ryan Cooper|Arjun Mehta|Tom Baker|Tom Barker|Hans Becker|Chloe Dubois|Liam Shaw|Olivia Brown|Michael Chen|Andrew Scott|Mark Johnson',
        'gen': 'Tên ngoại trong câu Việt; Baker/Barker, Priya Shah/Sharma; đổi trọng âm/tốc độ và giữ tên gốc trong nhãn.'},
    'dia_diem_to_chuc': {
        'risk': 'Sai query/nơi đón trả/đối tác hoặc mốc tham chiếu trong history.',
        'terms': 'Trúc Bạch|Hải Thượng Lãn Ông|Lê Đức Thọ|Thái Hà|The Lemon Tree Café|The Olive Branch Café|The Copper Kettle Café|Cao Nguyên Coffee|Riverside Coworking|Riverbank Residence|Riverbank Suites|Nexora Labs|Evergreen Foods|Northwind Trading|Kestrel Fiber|Lumina Studio|Crestview Hotels|Harbor View Hotel|Maple Leaf Education|Design Cloud|Bluebird Cafe|Bluebird Café|Skyline Park|Le Petit Bakery|Brightwave Studio|Team Design|Sun Square|Osaka Ramen|Berlin',
        'gen': 'Địa danh Việt và tên quán/tổ chức Anh trong câu tìm đường, đặt xe, chuyển tiền; giữ đầy đủ thành phần tên.'},
    'thoi_gian_quan_he': {
        'risk': 'Sai ngày/giờ hoặc trước/sau làm đặt lịch/nhắc vào sai thời điểm.',
        'terms': 'sáng mai|tối mai|tối nay|hôm qua|ngày kia|mai|nay|trước|sau|kém|rưỡi|muộn hơn|sớm hơn',
        'gen': 'Giờ kém/rưỡi; trước/sau một khoảng; mai/nay/ngày kia. Dùng cùng câu nền với giá trị tương phản.'},
    'tai_chinh_thanh_toan': {
        'risk': 'Nhầm phương thức/nguồn tiền/hóa đơn khi dùng banking hoặc ewallet.',
        'terms': 'ví điện tử|qua ví|tài khoản thanh toán|tài khoản tiết kiệm|tiền điện|tiền nước|tiền mạng|nạp ví|nạp điện thoại|mua vào|bán ra',
        'gen': 'Ví/ngân hàng; thanh toán/tiết kiệm; tiền điện/nước/mạng; nghìn/triệu và điều kiện đủ/thiếu số dư.'},
    'gia_tri_so': {
        'risk': 'Sai giá trị slot giờ, nhiệt độ, độ sáng, tiền hoặc chữ số định danh.',
        'terms': '',
        'gen': 'Cặp giá trị cùng câu: 26/28 độ, 30/50 phần trăm, 45/40 phút, điện thoại nhiều số 0. Không gen lại chỉ vì khác dạng viết số.'},
    'ten_rieng_phat_hien_them': {
        'risk': 'Tên riêng suy từ chữ hoa hoặc query nhãn: cần kiểm tra trước khi gen.',
        'terms': '',
        'gen': 'Rà tên/đối tượng và phiên âm trước; dùng câu gốc trong evidence để chọn vocab.'},
    'ten_lien_he_phat_hien_them': {
        'risk': 'Tên/quan hệ trích từ query danh bạ, có thể Việt hoặc quốc tế; cần kiểm tra đúng đối tượng.',
        'terms': '',
        'gen': 'Rà tên từ query cùng câu gốc; không suy nguồn ngôn ngữ chỉ từ tên query.'},
}

# Explicit language review of source/ASR/normalizer/call traces. This field-level
# evidence is distinct from the exhaustive, rule-assisted candidate inventory.
REVIEWED = [
    ('train_1474','thuoc_y_te','Omeprazole → OMEZON','medication.add dùng tên Omez thay Omeprazole','gen_ten_thuoc','ASR mất định danh thuốc; Qwen tiếp tục biến thành Omez.'),
    ('train_1703','thuoc_y_te','Mất huyết áp; check-in → Y','Chọn med_72 thay med_318 và bỏ nhắc khách check-in','gen_nhom_benh_va_code_switch','Tên nhóm thuốc bị mất; ý check-in cũng bị mất. Có thể có thêm lỗi agent.'),
    ('train_1510','thuoc_y_te','viên sắt → viên sạch; mất tảo Hawaii; dầu cá → yo cá','Ghi thêm thuốc chưa uống; thời gian cũng lệch 12:30 → 12:00','gen_ten_thuoc_nhung_khong_quy_loi_gio_cho_ASR','ASR vẫn giữ mười hai giờ rưỡi; lỗi 12:00 xuất hiện ở bước sau.'),
    ('train_1343','thuoc_y_te','Clearwater Pharma → CIA WATER PHẠM MA','medication.log_taken chọn med_5021 thay med_318','gen_ten_anh_trong_cau_y_te','Tên thuốc/nhãn trong corpus bị phiên âm; calling không chọn đúng schedule_id.'),
    ('train_1430','thuoc_y_te','kẽm Brightpath → CẢM BIẾN PART; thuốc Brightpath → THUỘC VINFAST','Tạo reminders với nội dung cảm biến part/VinFast','gen_ten_thuoc_va_cu_uong','Lỗi định danh quan trọng; reference thiếu giờ nên nhãn mong đợi ask_missing_info.'),
    ('train_1695','thuoc_y_te','vitamin D → vitamin T; lịch thuốc → ly thuốc','Agent kết thúc không gọi tool, bỏ yêu cầu nhắc vitamin D','gen_vitamin_va_lich_thuoc','Lỗi cần kiểm tra bằng câu y tế có tên chữ cái, không chỉ đọc chữ D riêng.'),
    ('train_1414','thuoc_y_te','cữ trưa/cữ sáng → cửa trưa/cửa sáng; Daniel Lee → Dani','Query Dani; thiếu cập nhật lịch thuốc','gen_cu_uong_va_ten_nguoi','Có lỗi agent độc lập về join/create meeting; chưa chứng minh ASR là nguyên nhân duy nhất.'),
    ('train_0083','gia_tri_so','30 phần trăm → năm mươi phần trăm','light_set brightness=50 thay 30','gen_gia_tri_tuong_phan','Sai giá trị thật, khác với chữ số/chữ viết tương đương.'),
    ('train_0133','ten_nguoi_viet','Minh → mình','Bỏ phần nhắn cho Minh; chỉ tắt đèn','gen_ten_nguoi_va_dai_tu','Đặt tên người trong khung nhắn/gọi/chuyển, tương phản với đại từ mình.'),
    ('train_0392','lenh_hanh_dong','đặt báo thức → đặt vào thức','Bỏ alarm.create; còn tự chọn số tiền chuyển','gen_y_dinh_bao_thuc','Bỏ báo thức phù hợp lỗi ASR; việc bịa số tiền là lỗi agent khác.'),
    ('train_0097','thoi_gian_quan_he','8 giờ kém 15 → tám giờ tối mười lăm','Tạo nhắc 20:00 thay 19:45','gen_quan_he_gio_kem','Mất từ kém làm mất quan hệ thời gian.'),
    ('train_1526','dia_diem_to_chuc','The Lemon Tree Café → đường Lê Cà Phê','maps.search_place query=Lê Cà Phê','gen_ten_quan_tieng_anh','Query sau đó dùng biến thể tên sai, cần tên quán Anh trong câu Việt.'),
    ('train_0828','dia_diem_to_chuc','Trúc Bạch → Chúc Bạn','schedule_pickup dùng 18 phố Hàng Cót','gen_dia_danh','ASR sai địa chỉ; Qwen còn tự thay thành địa chỉ khác, không chỉ sao chép lỗi ASR.'),
    ('train_1715','phu_dinh_dieu_kien','Đừng tắt tiếng → được tắt','Agent chỉnh volume=10, nhãn yêu cầu hỏi volume','gen_phu_dinh_nhung_chua_quy_calling_cho_ASR','Có nguy cơ đảo ý; calling không mute, nên không kết luận lỗi calling này do từ đừng.'),
    ('train_0256','doi_chung_loi_buoc_sau','45 phút → bốn lăm phút là đúng giá trị','Normalizer đổi sang bốn mươi; agent cập nhật 10:40 thay 10:45','khong_gen_de_sua_so_45','ASR có lỗi dời/team khác; riêng giá trị 45 không phải lỗi cần sửa bằng data ASR.'),
    ('train_1085','doi_chung_loi_buoc_sau','Số 0983 214 6755 được nhận đủ chữ số','Agent bỏ sms.send','khong_gen_de_sua_so_dien_thoai_nay','Nội dung và số điện thoại đã đủ; cần kiểm tra agent, không quy lỗi số cho ASR.'),
    ('train_0004','doi_chung_loi_buoc_sau','ASR giữ phòng ngủ nhỏ','Normalizer bỏ nhỏ; agent chỉnh sai đèn','khong_gen_de_sua_ten_phong_o_mau_nay','Từ phân biệt phòng bị mất ở normalizer, không phải ở ASR.'),
]


def semantic_text(text):
    result = ' '.join(EQUIVALENT.get(t,t) for t in spoken_text(text,contextual_aliases=True).split())
    for phrase, token in [('ký túc xá', 'ktx'), ('gia lô', 'zalo'), ('da lô', 'zalo')]:
        result = re.sub(r'\b' + phrase + r'\b', token, result)
    # Colloquial 'bốn lăm phút' preserves the value 45. Avoid treating the
    # missing word 'mươi' as a changed time value for data-generation targets.
    result = re.sub(r'\b(hai|ba|bốn|năm|sáu|bảy|tám|chín) lăm (?=phút\b|độ\b|nghìn\b|triệu\b)', r'\1 mươi lăm ', result)
    return result


def extract_lexicon(utterances,cases):
    lexicon={}
    def add(term,category,source):
        normal=semantic_text(term)
        if normal and normal not in lexicon:
            lexicon[normal]={'term':term,'category':category,'source':source,'tokens':normal.split()}
    for category,spec in CATEGORIES.items():
        for term in spec['terms'].split('|'):
            if term:add(term,category,'curated')
    stop=set('anh chị cô chú bác em con tôi mình thế rồi còn nếu này cho gọi nhắn coi xem tìm giúp đặt thêm hủy xoá xóa mở bật tắt đọc ghi ờ mai hôm có không nạp chuyển đánh giờ ủa quanh cái lúc sau trước từ tại tới đến số xăng vàng vé điểm trạm bên cậu'.split())
    number=r'(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|mười|mươi|trăm|nghìn|triệu|tỷ|tỉ|phẩy)'
    digit=r'(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)'
    for row in utterances:
        for m in re.finditer(r'\b[A-ZĐ][\w-]*(?:\s+[A-ZĐ][\w-]*)+',row['reference']):
            words=m.group().split()
            while words and semantic_text(words[0]) in stop:words.pop(0)
            if len(words)>=2 and words[0] not in {'Thứ','Nhóm','Zalo'}:
                add(' '.join(words),'ten_rieng_phat_hien_them','capitalized_reference_candidate')
        for m in re.finditer(r'\brv\d+\b',row['reference'],re.I):add(m.group(),'viet_tat_ma','literal_flight_code')
        ref=semantic_text(row['reference'])
        for m in re.finditer(rf'\b{number}(?: {number}){{0,10}} (?:phần trăm|độ|phút|giờ|tiếng|đồng|đô|lít)\b',ref):
            # 'RON95 giờ lên giá' does not contain a 95-hour time value.
            previous = ref[:m.start()].split()
            if previous and previous[-1] in {'ron', 'rôn', 'rv', 'vtv', 'htv', 'sctv', 'vov'}:continue
            add(m.group(),'gia_tri_so','number_with_explicit_unit')
        for m in re.finditer(rf'\b{number}(?: {number}){{0,8}} (?:nghìn|triệu|tỷ)\b',ref):
            add(m.group(),'gia_tri_so','monetary_value_candidate')
        for m in re.finditer(rf'\b{digit}(?: {digit}){{7,12}}\b',ref):
            add(m.group(),'gia_tri_so','spoken_digit_sequence')
    for case in cases.values():
        calls=list(case['label']['tool_calls'])
        for turn in case['history']:
            if turn.get('role')=='assistant':calls.extend(turn.get('tool_calls',[]))
        for call in calls:
            if call['name']=='contacts.search':
                value=call.get('arguments',{}).get('query','')
                if isinstance(value,str) and 1<=len(value.split())<=6 and semantic_text(value) not in {'mai','hai','ba','bạn','anh','chị','cô','chú'}:
                    add(value,'ten_lien_he_phat_hien_them','contact_query_label_candidate')
    return lexicon


def write_csv(name,rows,fields=None):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def medical_group(term):
    text=semantic_text(term)
    groups=[('huyết áp','nhom_benh_huyet_ap'),('tiểu đường','nhom_benh_tieu_duong'),
            ('dạ dày','tu_vung_da_day_bao_tu'),('bao tử','tu_vung_da_day_bao_tu'),
            ('kháng sinh','nhom_khang_sinh'),('dị ứng','tu_vung_di_ung'),
            ('siro','tu_vung_ho_long_dom'),('long đờm','tu_vung_ho_long_dom'),
            ('say xe','tu_vung_say_xe')]
    for word,group in groups:
        if word in text:return group
    if any(word in text for word in ('vitamin','omega','viên sắt','kẽm','tảo','dầu cá','bổ gan')):
        return 'ten_nhom_vien_uong_bo_sung_trong_corpus'
    if any(word in text for word in ('omeprazole','clearwater','brightpath')):
        return 'ten_tieng_anh_nhan_thuoc_trong_corpus'
    return 'lich_cu_uong_trang_thai_da_chua'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    utterances=list(csv.DictReader(Path('outputs/asr_spoken_eval/utterances.csv').open(encoding='utf-8-sig')))
    cases={r['id']:r for r in map(json.loads,(DATA/'train.jsonl').open(encoding='utf-8'))}
    runs={r['id']:r for r in map(json.loads,(BASE/'results.jsonl').open(encoding='utf-8'))}
    lexicon=extract_lexicon(utterances,cases)
    by_first=defaultdict(list)
    for key,spec in lexicon.items():by_first[spec['tokens'][0]].append((key,spec))
    termstats=defaultdict(lambda:{'mentions':0,'audio':set(),'bad_mentions':0,'bad_audio':set(),'variants':Counter(),'examples':[]})
    evidence=[];unclassified=[];category_audio=defaultdict(set);category_sids=defaultdict(set);predictions={}
    def actual_calls(sid):
        if sid not in predictions:
            value=runs[sid].get('predicted_tool_calls')
            if value is None:
                path=BASE/'samples'/sid/'trace.jsonl'
                value=[{'name':e['name'],'arguments':e['arguments']} for line in path.read_text().splitlines() if (e:=json.loads(line)).get('stage')=='tool'] if path.exists() else []
            predictions[sid]=value
        return predictions[sid]
    for row in utterances:
        sid=row['id'];key=(sid,row['turn'],row['history_index'])
        ref=semantic_text(row['reference']);hyp=semantic_text(row['asr'])
        r,h,ops,_=align(ref,hyp);rt=[t for _,t in r];covered=set()
        for start,word in enumerate(rt):
            for termkey,spec in by_first[word]:
                tt=spec['tokens'];end=start+len(tt)
                if rt[start:end]!=tt:continue
                if spec['category']=='gia_tri_so' and start and rt[start-1] in {'ron','rôn','xăng','rv','vtv','htv','sctv','vov'}:continue
                if spec['category']=='thoi_gian_quan_he' and spec['term']=='mai' and start and rt[start-1] in {'cô','chị','anh','bà','ông','bạn','sếp','thầy'}:continue
                if spec['category']=='thuoc_y_te' and spec['term'] in {'uống rồi','chưa uống'} and not re.search(r'thuốc|vitamin|viên|kháng sinh|siro|dầu cá',ref):continue
                selected=[(op,i,j) for op,i,j in ops if (op!='insertion' and start<=i<end) or (op=='insertion' and start<i<end)]
                heard=' '.join(h[j][1] for op,i,j in selected if op!='deletion')
                stat=termstats[termkey];stat['mentions']+=1;stat['audio'].add(key)
                if heard==termkey:continue
                if [w for w in heard.split() if w not in FILLERS]==[w for w in tt if w not in FILLERS]:continue
                stat['bad_mentions']+=1;stat['bad_audio'].add(key);stat['variants'][heard or '[mất toàn bộ cụm]']+=1
                if len(stat['examples'])<3:stat['examples'].append((sid,row['turn'],row['reference'],heard))
                covered.update(range(start,end));cat=spec['category'];category_audio[cat].add(key);category_sids[cat].add(sid)
                predicted=actual_calls(sid) if row['turn']=='last' else []
                index=int(row['history_index']) if row['turn']=='history' else None
                expected=get_calls(cases[sid],index)
                evidence.append({'id':sid,'turn':row['turn'],'history_index':row['history_index'],'category':cat,'reference_target':spec['term'],'aligned_asr_target':heard,'source':spec['source'],'potential_downstream_risk':CATEGORIES[cat]['risk'],'reference_context':' '.join(rt[max(0,start-5):min(len(rt),end+5)]),'reference_text':row['reference'],'asr_text':row['asr'],'apps':row['apps'],'domains':row['domains'],'audio':row['audio'],'last_calling_exact':runs[sid]['exact_calls'] if row['turn']=='last' else 'not_measured_for_history','expected_tool_calls':json.dumps(expected,ensure_ascii=False),'actual_tool_calls':json.dumps(predicted,ensure_ascii=False) if row['turn']=='last' else '','impact_status':'semantic_risk_candidate_not_causal_proof'})
        blocks=[];block=[]
        for op,i,j in ops:
            if op=='equal':
                if block:blocks.append(block);block=[]
            else:block.append((op,i,j))
        if block:blocks.append(block)
        for block in blocks:
            pos=[i for op,i,j in block if op!='insertion']
            if pos and all(i in covered for i in pos):continue
            a=' '.join(r[i][1] for op,i,j in block if op!='insertion')
            b=' '.join(h[j][1] for op,i,j in block if op!='deletion')
            if set(a.split()+b.split())<=FILLERS:continue
            unclassified.append({'id':sid,'turn':row['turn'],'history_index':row['history_index'],'reference_span':a,'asr_span':b,'reference_text':row['reference'],'asr_text':row['asr'],'apps':row['apps'],'audio':row['audio'],'status':'unclassified_review_needed'})
    targets=[]
    for termkey,stat in termstats.items():
        spec=lexicon[termkey];cat=spec['category']
        if not stat['bad_mentions'] and cat!='thuoc_y_te':continue
        targets.append({'category':cat,'reference_target':spec['term'],'source':spec['source'],'mentions_in_reference':stat['mentions'],'audio_mentions':len(stat['audio']),'suspected_bad_mentions':stat['bad_mentions'],'suspected_bad_audio':len(stat['bad_audio']),'review_priority':'review_discovered_entity' if cat in {'ten_rieng_phat_hien_them','ten_lien_he_phat_hien_them'} else ('targeted_generation_candidate' if stat['bad_mentions'] else 'no_target_corruption_observed'),'asr_variants':json.dumps(stat['variants'].most_common(6),ensure_ascii=False),'examples':json.dumps(stat['examples'],ensure_ascii=False),'potential_downstream_risk':CATEGORIES[cat]['risk'],'generation_focus':CATEGORIES[cat]['gen']})
    targets.sort(key=lambda r:(r['category'],-r['suspected_bad_audio'],r['reference_target']))
    write_csv('targets.csv',targets)
    write_csv('priority_targets.csv',[r for r in targets if r['source']=='curated' and r['suspected_bad_mentions']])
    write_csv('medical_targets.csv',[{'medical_group':medical_group(r['reference_target']),**r} for r in targets if r['category']=='thuoc_y_te'])
    write_csv('evidence.csv',evidence)
    write_csv('unclassified_review.csv',unclassified)
    category_rows=[]
    for cat,spec in CATEGORIES.items():
        ts=sorted((t for t in targets if t['category']==cat and t['suspected_bad_mentions']),key=lambda t:-t['suspected_bad_audio'])
        category_rows.append({'category':cat,'audio_with_candidate':len(category_audio[cat]),'sample_ids_with_candidate':len(category_sids[cat]),'distinct_flagged_targets':len(ts),'top_targets':'; '.join(t['reference_target'] for t in ts[:20]),'risk':spec['risk'],'generation_focus':spec['gen']})
    write_csv('generation_brief.csv',category_rows)
    reviewed=[]
    for sid,cat,error,outcome,action,note in REVIEWED:
        folder=BASE/'samples'/sid
        reviewed.append({'id':sid,'category':cat,'reviewed_content_difference':error,'observed_calling_difference':outcome,'generation_decision':action,'review_note':note,'reference_text':cases[sid]['audio_text'],'asr_text':json.loads((folder/'asr_gipformer.cache.json').read_text())['value'],'qwen_normalized_text':json.loads((folder/'normalize.cache.json').read_text())['value'],'expected_tool_calls':json.dumps(cases[sid]['label']['tool_calls'],ensure_ascii=False),'actual_tool_calls':json.dumps(actual_calls(sid),ensure_ascii=False),'review_method':'language_review_of_saved_text_and_calls_no_audio_or_counterfactual'})
    write_csv('reviewed_examples.csv',reviewed)
    summary={'transcripts_checked':len(utterances),'sample_ids':len(cases),'last':sum(r['turn']=='last' for r in utterances),'history':sum(r['turn']=='history' for r in utterances),'candidate_evidence_rows':len(evidence),'distinct_candidate_audio':len({(r['id'],r['turn'],r['history_index']) for r in evidence}),'target_rows':len(targets),'unclassified_spans':len(unclassified),'categories':category_rows,'limitations':['Lexicon/rule-assisted candidates for semantic risk, not verified accuracy or causal attribution.','No audio listening or gold-transcript counterfactual inference was run.','Categories overlap; names derived from search labels/capitalization may need review.','Spoken numbers, fixed spelling/dialect aliases and polite fillers excluded.','History errors are listed as training candidates, without separate downstream causal measurement.','Medical survey covers corpus terms only, not all medicines or clinical recommendations.']}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'lexicon.json').write_text(json.dumps(lexicon,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in {'categories','limitations'}},ensure_ascii=False))
    for r in category_rows:print(r['category'],r['audio_with_candidate'],r['distinct_flagged_targets'],r['top_targets'][:150])

if __name__=='__main__':main()
