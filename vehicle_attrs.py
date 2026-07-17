"""
vehicle_attrs.py — nhận diện màu xe và đọc biển số bằng EasyOCR.

BẢN CHUYỂN SANG EASYOCR (thay cho Tesseract + dò contour/MSER tự viết):
- Vì sao đổi: sau khi test kỹ trên ảnh thực tế, phần ĐỊNH VỊ vùng biển số bằng
  contour/tỉ lệ khung hình tự viết tỏ ra rất mong manh — dễ vỡ vụn thành nhiều
  mảnh nhỏ, hoặc bắt nhầm nắp capo/kính/lá cây, tuỳ theo góc chụp và ánh sáng.
  Tesseract tự nó ĐỌC rất tốt khi được đưa đúng vùng đã crop khít (đã kiểm
  chứng), nhưng việc tự viết luật để tìm ra "đúng vùng đó" mới là phần khó.
- EasyOCR giải quyết đúng phần khó này: nó có sẵn bộ ĐỊNH VỊ văn bản (CRAFT)
  được huấn luyện sẵn trên hàng triệu ảnh chữ trong ảnh tự nhiên, tự tìm ra MỌI
  vùng có chữ trong cả tấm ảnh xe — không cần dò contour/tỉ lệ khung hình thủ
  công nữa. Ta chỉ cần lọc trong các vùng chữ tìm được, vùng nào khớp định dạng
  biển số VN thì lấy.
- Đánh đổi so với bản Tesseract cũ:
    + Cần cài thêm gói `easyocr` (pip install easyocr).
    + Lần chạy ĐẦU TIÊN cần có mạng để tự tải model nhận diện (~150MB).
    + Chậm hơn Tesseract: khoảng 1-4 giây/ảnh xe trên CPU (nhanh hơn nhiều nếu
      máy có GPU NVIDIA + CUDA, xem ghi chú ở biến gpu=False bên dưới).
    + Model được load MỘT LẦN duy nhất khi khởi động chương trình (giống cách
      YOLO được load 1 lần trong app.py) — không load lại mỗi lần gọi hàm.

Lưu ý về độ tin cậy (không đổi so với các bản trước):
- Màu xe: ước lượng bằng màu chiếm đa số (dominant color) trên vùng thân xe.
  Có thể sai với xe có kính đen lớn, xe 2 tông màu (nóc đen), hoặc bị cây/bóng che.
- Biển số: nếu ảnh gốc chụp quá xa khiến biển số chỉ còn vài chục pixel, không
  OCR nào (kể cả EasyOCR hay model chuyên dụng) đọc chính xác được — đây là giới
  hạn của ảnh đầu vào, không phải lỗi thuật toán. Khi không đọc được, hàm trả về
  None thay vì đoán bừa.
"""

import os
import re
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import easyocr

# ---------------------------------------------------------------------------
# Load model EasyOCR MỘT LẦN khi module được import (không load lại mỗi request)
# ---------------------------------------------------------------------------
# gpu=False: chạy được trên MỌI máy (kể cả không có GPU). Nếu máy bạn có GPU
# NVIDIA + đã cài CUDA-enabled torch (ultralytics thường tự cài sẵn), đổi thành
# gpu=True sẽ nhanh hơn đáng kể (giây -> phần mười giây mỗi ảnh).
print("Đang tải model EasyOCR (chỉ chạy 1 lần lúc khởi động, có thể mất chút thời gian)...")
_OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
print("EasyOCR đã sẵn sàng.")


# ---------------------------------------------------------------------------
# Màu sắc (giữ nguyên hoàn toàn so với bản trước — phần này đã hoạt động tốt)
# ---------------------------------------------------------------------------

def _rgb_to_vn_color(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    hsv = cv2.cvtColor(np.uint8([[[r * 255, g * 255, b * 255]]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]) * 2, hsv[1] / 255 * 100, hsv[2] / 255 * 100
    if v < 25:
        return "đen"
    if s < 15:
        if v > 78:
            return "trắng"
        elif v > 50:
            return "xám/bạc"
        return "xám đậm"
    if h < 15 or h >= 345:
        return "đỏ"
    if h < 45:
        return "cam"
    if h < 65:
        return "vàng"
    if h < 170:
        return "xanh lá"
    if h < 200:
        return "xanh ngọc"
    if h < 260:
        return "xanh dương"
    if h < 290:
        return "tím"
    return "hồng"


def detect_color(crop: Image.Image):
    """Trả về tên màu tiếng Việt, hoặc None nếu không đủ tin cậy (vd. bị lá cây che gần hết)."""
    w, h = crop.size
    body = crop.crop((int(w * 0.12), int(h * 0.30), int(w * 0.88), int(h * 0.90)))
    arr = np.array(body.convert("RGB")).reshape(-1, 3).astype(np.float32)
    if len(arr) < 20:
        return None

    k = min(5, max(2, len(arr) // 20))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(arr)
    counts = np.bincount(km.labels_)
    total = counts.sum()

    black_share, green_share = 0.0, 0.0
    cluster_info = []
    for idx, cnt in enumerate(counts):
        r, g, b = km.cluster_centers_[idx]
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
        hh, ss, vv = int(hsv[0]) * 2, hsv[1] / 255 * 100, hsv[2] / 255 * 100
        share = cnt / total
        is_green_foliage = 65 <= hh < 170 and ss > 20 and vv >= 25
        if vv < 25:
            black_share += share
        if is_green_foliage:
            green_share += share
        cluster_info.append((share, (r, g, b), is_green_foliage))

    if black_share > 0.45:
        return "đen"
    if green_share > 0.40:
        return None  # nhiều khả năng bị lá cây che khuất phần lớn xe

    cluster_info.sort(key=lambda x: -x[0])
    for share, rgb, is_green in cluster_info:
        if is_green:
            continue
        return _rgb_to_vn_color(rgb)
    return _rgb_to_vn_color(cluster_info[0][1])


# ---------------------------------------------------------------------------
# Biển số xe — EASYOCR (định vị + đọc chữ trong 1 bước, không cần dò contour)
# ---------------------------------------------------------------------------

# Mẫu biển số VN phổ biến sau khi loại bỏ khoảng trắng/dấu: 2 số + 1-2 chữ + 4-5 số
_PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,2}\d{4,5}$")

# Chỉ cho phép các ký tự thực sự xuất hiện trên biển số VN (loại bỏ chữ dễ nhầm
# như I, J, O, Q, R, W vốn không dùng trên biển số xe VN) — giúp EasyOCR bớt
# nhầm lẫn ngay từ bước nhận diện ký tự.
_ALLOWLIST = "ABCDEFGHKLMNPSTUVXYZ0123456789"


def _clean_plate_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


# Các cặp ký tự hay bị OCR nhầm lẫn (dựa trên hình dạng giống nhau, đặc biệt với
# font chữ block/stencil trên biển số VN). Dùng để "sửa thử" khi chuỗi đọc được
# gần đúng định dạng nhưng lệch đúng 1 ký tự.
# Các cặp ký tự hay bị OCR nhầm lẫn — CHỈ giữ cặp 1-đối-1 rõ ràng (dựa trên hình
# dạng giống nhau với font block/stencil của biển số VN, và trên lỗi ĐÃ QUAN SÁT
# thực tế: "3" bị đọc thành "D"). Không đưa các ký tự vốn đã bị loại khỏi
# _ALLOWLIST (O, Q, I, J, R, W) vào đây vì EasyOCR sẽ không bao giờ xuất ra chúng.
_CONFUSABLE = {
    "D": "3", "3": "D",
    "B": "8", "8": "B",
    "S": "5", "5": "S",
    "G": "6", "6": "G",
    "Z": "2", "2": "Z",
    "L": "1", "1": "L",
    "A": "4", "4": "A",
}
_PLAUSIBLE_LENGTHS = (7, 8, 9)  # 2+1+4=7, 2+1+5=8 hoặc 2+2+4=8, 2+2+5=9


def _fuzzy_correct(cleaned: str):
    """Thử sửa 1 ký tự dễ nhầm để chuỗi khớp định dạng biển số VN.

    CHỈ chấp nhận nếu việc thử mọi vị trí + mọi ký tự thay thế khả dĩ cho ra
    DUY NHẤT một chuỗi hợp lệ — nếu có từ 2 phương án hợp lệ trở lên (không rõ
    phương án nào đúng), trả về None để tránh đoán bừa, giữ đúng nguyên tắc gốc
    "không chắc thì không đoán" của hệ thống.
    """
    if len(cleaned) not in _PLAUSIBLE_LENGTHS:
        return None
    if _PLATE_RE.match(cleaned):
        return cleaned

    found = set()
    for i, ch in enumerate(cleaned):
        for alt in _CONFUSABLE.get(ch, ""):
            candidate = cleaned[:i] + alt + cleaned[i + 1:]
            if _PLATE_RE.match(candidate):
                found.add(candidate)
    if len(found) == 1:
        return next(iter(found))
    return None


def detect_plate(crop: Image.Image, debug_dir: str | None = None):
    """Dùng EasyOCR quét TOÀN BỘ ảnh xe (không cần crop/dò vùng trước), tìm vùng
    text nào sau khi làm sạch khớp định dạng biển số VN.

    QUY TRÌNH:
    1) Quét cả ảnh xe -> định vị vùng nghi ngờ là biển số.
    2) Với các vùng nghi ngờ, CẮT SÁT + PHÓNG TO rồi đọc lại lần 2 (chính xác
       hơn vì ký tự đã đủ lớn).
    3) Ở CẢ 2 bước trên, ưu tiên tuyệt đối cho kết quả KHỚP CHÍNH XÁC định dạng
       biển số VN — trả về ngay khi tìm thấy.
    4) CHỈ KHI không có kết quả nào khớp chính xác, mới xét đến "sửa ký tự dễ
       nhầm" (fuzzy correction) trên TOÀN BỘ các chuỗi đã thu thập được xuyên
       suốt bước 1+2 — chỉ chấp nhận nếu việc sửa cho ra DUY NHẤT 1 phương án
       hợp lệ trên toàn bộ tập hợp (tránh đoán bừa khi có nhiều khả năng).

    debug_dir: nếu truyền vào, lưu ảnh khoanh vùng text EasyOCR tìm được ở bước 1
    ra debug_dir/easyocr_boxes.png, và các vùng đã phóng to ở bước 2 ra
    debug_dir/zoom_candX.png — hữu ích để xem hệ thống đang "nhìn" vào đâu.
    """
    car_bgr = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = car_bgr.shape[:2]
    if w < 40 or h < 40:
        return None

    try:
        results = _OCR_READER.readtext(
            car_bgr, allowlist=_ALLOWLIST, mag_ratio=2.0, text_threshold=0.6, low_text=0.35,
        )
    except Exception as e:
        print("    [EasyOCR lỗi]", e)
        return None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = car_bgr.copy()
        for bbox, text, conf in results:
            cv2.polylines(vis, [np.array(bbox, dtype=np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "easyocr_boxes.png"), vis)

    results_sorted = sorted(results, key=lambda r: -r[2])
    stage1_texts = []  # list các (text, conf) từ lượt 1 (chưa zoom, độ tin cậy thấp hơn)
    stage2_texts = []  # list các (text, conf) từ lượt 2 (đã zoom, đáng tin hơn nhiều)

    # --- Lượt 1: khớp chính xác ngay nếu có ---
    for bbox, text, conf in results_sorted:
        cleaned = _clean_plate_text(text)
        if not cleaned:
            continue
        if debug_dir:
            print(f"    [EasyOCR-1] '{cleaned}' (conf={conf:.2f})")
        stage1_texts.append((cleaned, conf))
        if _PLATE_RE.match(cleaned):
            return cleaned

    # ghép 2 mảnh cùng hàng gần nhau (vd "98A" + "21933")
    merged_candidates = []  # (bbox_hop_nhat, text_hop_nhat)
    for i in range(len(results_sorted)):
        for j in range(len(results_sorted)):
            if i == j:
                continue
            bbox_i, text_i, _ = results_sorted[i]
            bbox_j, text_j, _ = results_sorted[j]
            yi = (bbox_i[0][1] + bbox_i[2][1]) / 2
            yj = (bbox_j[0][1] + bbox_j[2][1]) / 2
            xi_right, xj_left = bbox_i[1][0], bbox_j[0][0]
            if abs(yi - yj) < max(h * 0.06, 6) and 0 <= (xj_left - xi_right) < w * 0.08:
                merged_text = _clean_plate_text(text_i + text_j)
                if not merged_text:
                    continue
                stage1_texts.append((merged_text, min(results_sorted[i][2], results_sorted[j][2])))
                if _PLATE_RE.match(merged_text):
                    return merged_text
                xs = [p[0] for p in bbox_i] + [p[0] for p in bbox_j]
                ys = [p[1] for p in bbox_i] + [p[1] for p in bbox_j]
                merged_candidates.append(((min(xs), min(ys), max(xs), max(ys)), merged_text))

    # --- Lượt 2: zoom vào các vùng nghi ngờ rồi đọc lại ---
    zoom_targets = []
    for bbox, text, conf in results_sorted[:4]:  # giới hạn top 4 để đỡ chậm
        cleaned = _clean_plate_text(text)
        if 5 <= len(cleaned) <= 10:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            zoom_targets.append((min(xs), min(ys), max(xs), max(ys)))
    for (x1, y1, x2, y2), _ in merged_candidates[:2]:
        zoom_targets.append((x1, y1, x2, y2))

    for idx, (x1, y1, x2, y2) in enumerate(zoom_targets):
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.25) + 4, int(bh * 0.35) + 4
        xx1, yy1 = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
        xx2, yy2 = min(w, int(x2) + pad_x), min(h, int(y2) + pad_y)
        sub = car_bgr[yy1:yy2, xx1:xx2]
        if sub.shape[0] < 5 or sub.shape[1] < 10:
            continue

        sw = sub.shape[1]
        scale = min(8, max(2, 550 // max(sw, 1)))
        big = cv2.resize(sub, (sub.shape[1] * scale, sub.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        # Tăng tương phản (CLAHE) trên ảnh đã zoom — dù mắt thường thấy ảnh đã
        # khá rõ, bộ nhận diện ký tự của EasyOCR vẫn có thể đọc sai nhiều ký tự
        # trên ảnh tương phản thấp/có chút loá sáng. CLAHE giúp làm rõ nét viền
        # ký tự hơn (đã kiểm chứng hiệu quả khi test thủ công trước đó).
        gray_big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_big)

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"zoom_cand{idx}.png"), big)
            cv2.imwrite(os.path.join(debug_dir, f"zoom_cand{idx}_enhanced.png"), enhanced)

        # Thử cả ảnh gốc (màu) VÀ ảnh đã tăng tương phản, với decoder='beamsearch'
        # (giải mã kỹ hơn 'greedy' mặc định, chậm hơn chút nhưng ảnh đã nhỏ nên
        # không đáng kể) — dùng kết quả đầu tiên khớp định dạng.
        zoom_results = []
        for img_variant, variant_name in ((enhanced, "enhanced"), (big, "raw")):
            try:
                r = _OCR_READER.readtext(
                    img_variant, allowlist=_ALLOWLIST, text_threshold=0.5, low_text=0.3,
                    decoder="beamsearch", beamWidth=5,
                )
            except Exception:
                continue
            zoom_results.extend(r)
            for zbbox, ztext, zconf in sorted(r, key=lambda x: -x[2]):
                zcleaned = _clean_plate_text(ztext)
                if not zcleaned:
                    continue
                if debug_dir:
                    print(f"    [EasyOCR-2 zoom{idx}-{variant_name}] '{zcleaned}' (conf={zconf:.2f})")
                stage2_texts.append((zcleaned, zconf))
                if _PLATE_RE.match(zcleaned):
                    return zcleaned

        zsorted = sorted(zoom_results, key=lambda r: -r[2])

        for i in range(len(zsorted)):
            for j in range(len(zsorted)):
                if i == j:
                    continue
                zi, zj = zsorted[i], zsorted[j]
                zyi = (zi[0][0][1] + zi[0][2][1]) / 2
                zyj = (zj[0][0][1] + zj[0][2][1]) / 2
                zxi_right, zxj_left = zi[0][1][0], zj[0][0][0]
                if abs(zyi - zyj) < max(big.shape[0] * 0.15, 8) and 0 <= (zxj_left - zxi_right) < big.shape[1] * 0.15:
                    zmerged = _clean_plate_text(zi[1] + zj[1])
                    if not zmerged:
                        continue
                    stage2_texts.append((zmerged, min(zi[2], zj[2])))
                    if _PLATE_RE.match(zmerged):
                        return zmerged

    # --- Không có gì khớp chính xác -> thử sửa ký tự dễ nhầm ---
    # Ưu tiên tuyệt đối các ứng viên từ LƯỢT 2 (đã zoom, đáng tin cậy hơn nhiều
    # vì ký tự đã đủ lớn) — chỉ dùng lượt 1 làm phương án dự phòng khi lượt 2
    # hoàn toàn không có ứng viên nào (vd ảnh quá nhỏ để zoom được).
    for pool in (stage2_texts, stage1_texts):
        if not pool:
            continue
        # gom theo chuỗi đã sửa -> giữ lại độ tin cậy CAO NHẤT trong các nguồn
        # đã cho ra chuỗi đó (1 chuỗi có thể được nhiều vùng/biến thể cùng gợi ý)
        corrected_conf = {}
        for text, conf in pool:
            corrected = _fuzzy_correct(text)
            if corrected:
                corrected_conf[corrected] = max(corrected_conf.get(corrected, 0.0), conf)
        if debug_dir and corrected_conf:
            print(f"    [Fuzzy] các phương án sau khi sửa ký tự: {corrected_conf}")

        if len(corrected_conf) == 1:
            return next(iter(corrected_conf))

        if len(corrected_conf) > 1:
            # Bước 1: ưu tiên phương án DÀI HƠN — lỗi OCR làm RỚT MẤT 1 ký tự
            # (undersegment do ảnh mờ/nén) phổ biến hơn nhiều so với việc OCR tự
            # "bịa" thêm ký tự thừa.
            by_len = sorted(corrected_conf.items(), key=lambda kv: -len(kv[0]))
            max_len = len(by_len[0][0])
            longest_group = [kv for kv in by_len if len(kv[0]) == max_len]
            if len(longest_group) == 1:
                return longest_group[0][0]

            # Bước 2: vẫn còn nhiều phương án cùng độ dài dài nhất -> phân xử
            # bằng ĐỘ TIN CẬY gốc của EasyOCR — chỉ chấp nhận nếu có 1 phương án
            # vượt trội rõ rệt (chênh lệch >= 0.15) so với phương án kế tiếp,
            # tránh chọn nhầm khi độ tin cậy sát nhau (thực sự mơ hồ).
            by_conf = sorted(longest_group, key=lambda kv: -kv[1])
            if len(by_conf) == 1 or (by_conf[0][1] - by_conf[1][1]) >= 0.15:
                return by_conf[0][0]

            # thực sự mơ hồ (độ dài bằng nhau VÀ độ tin cậy sát nhau) -> không
            # đoán, dừng ở rổ này
            break

    return None


# ---------------------------------------------------------------------------
# Định dạng kết quả theo yêu cầu: "Xe 1: màu xanh - 30AP02425", v.v.
# ---------------------------------------------------------------------------

def format_vehicle_line(index: int, color: str | None, plate: str | None) -> str:
    if color and plate:
        return f"Xe {index}: màu {color} - {plate}"
    if plate and not color:
        return f"Xe {index}: {plate} (không nhận diện được màu)"
    if color and not plate:
        return f"Xe {index}: màu {color} (không nhận diện được biển số xe)"
    return f"Xe {index}: (không nhận diện được màu, không nhận diện được biển số xe)"


def analyze_vehicle(crop: Image.Image, index: int, debug_dir: str | None = None):
    """debug_dir (tuỳ chọn): xem docstring của detect_plate()."""
    color = detect_color(crop)
    plate = detect_plate(crop, debug_dir=debug_dir)
    return {
        "index": index,
        "color": color,
        "plate": plate,
        "label": format_vehicle_line(index, color, plate),
    }