# Đếm xe ô tô tự động (Car Counter Webapp)

Ứng dụng web nhận diện xe ô tô trong ảnh, đếm số lượng, vẽ bounding box, và cố gắng nhận diện **màu xe** + **biển số xe** cho từng xe — sử dụng YOLOv8x để phát hiện xe và EasyOCR để đọc biển số.

Thả 1 tấm ảnh vào web (ảnh chụp bãi xe, đường phố...), hệ thống sẽ tự động:
- Phát hiện và đếm số xe (ô tô, xe tải, xe bus, xe máy)
- Vẽ khung + đánh số thứ tự cho từng xe
- Ước lượng màu xe (tiếng Việt: trắng, đen, xám/bạc, đỏ, xanh dương...)
- Cố gắng đọc biển số xe (định dạng biển số Việt Nam)

## Demo giao diện

Giao diện web đơn giản: kéo thả ảnh vào khung, bấm xử lý, xem kết quả ngay trên trình duyệt kèm danh sách chi tiết từng xe.

## Tính năng chính

- **Phát hiện xe**: dùng model YOLOv8x (`ultralytics`), lọc theo các lớp xe (car, truck, bus, motorcycle), áp dụng NMS (Non-Max Suppression) không phân biệt lớp để tránh trùng lặp box.
- **Nhận diện màu xe**: phân tích màu chiếm đa số (dominant color, dùng KMeans clustering) trên vùng thân xe, tự động loại trừ nhiễu do lá cây/bóng che.
- **Đọc biển số xe (EasyOCR)**:
  - Quét toàn bộ ảnh xe để định vị vùng nghi ngờ là biển số (không cần dò contour thủ công).
  - Cắt sát + phóng to vùng nghi ngờ, đọc lại lần 2 với độ phân giải cao hơn để tăng độ chính xác.
  - Tự động sửa các lỗi ký tự dễ nhầm phổ biến (ví dụ `D↔3`, `B↔8`, `S↔5`, `G↔6`, `Z↔2`, `L↔1`, `A↔4`) — **chỉ chấp nhận khi có duy nhất 1 phương án sửa hợp lệ**, tránh đoán bừa.
  - Trả về `None` (không hiển thị biển số) nếu không đủ tin cậy, thay vì đoán sai.
- **2 cách sử dụng**:
  - **Web app** (`app.py`): xử lý 1 ảnh theo yêu cầu qua giao diện trình duyệt.
  - **Batch script** (`detect_cars.py`): xử lý hàng loạt tất cả ảnh trong thư mục `input/`, xuất báo cáo Excel + JSON.

## Công nghệ sử dụng

| Thành phần | Công cụ |
|---|---|
| Phát hiện xe | [YOLOv8x](https://github.com/ultralytics/ultralytics) (`yolov8x.pt`) |
| Đọc biển số | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Nhận diện màu | OpenCV + scikit-learn (KMeans) |
| Web server | Flask |
| Xử lý ảnh | Pillow, OpenCV |
| Xuất báo cáo | pandas + openpyxl (Excel) |

## Cài đặt

### Yêu cầu
- Python 3.10+
- Có kết nối mạng ở lần chạy đầu tiên (để tự động tải trọng số YOLOv8x `~130MB` và model EasyOCR `~150MB`)

### Các bước

```bash
# 1. Clone repo
git clone <link-repo-cua-ban>
cd car-counter-webapp-v2

# 2. Tạo virtual environment (khuyến khích)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Cài thư viện
pip install -r requirements.txt
```

> **Lưu ý**: Lần chạy đầu tiên, `ultralytics` sẽ tự tải `yolov8x.pt` (~130MB) và `EasyOCR` sẽ tự tải model nhận diện (~150MB) — cần có mạng, có thể mất vài phút tuỳ tốc độ đường truyền. Các lần chạy sau sẽ dùng lại model đã tải, không cần mạng nữa (trừ khi model bị xoá).

## Cách sử dụng

### 1. Chạy web app

```bash
python app.py
```

Đợi đến khi console hiện:
```
Đang tải mô hình YOLOv8x, vui lòng chờ...
Đã sẵn sàng nhận ảnh.
Đang tải model EasyOCR (chỉ chạy 1 lần lúc khởi động, có thể mất chút thời gian)...
EasyOCR đã sẵn sàng.
```

Sau đó mở trình duyệt vào:
```
http://127.0.0.1:5000
```

Thả ảnh vào khung, hệ thống sẽ tự xử lý và hiển thị kết quả (ảnh đã vẽ khung + danh sách từng xe kèm màu/biển số).

⏱️ **Thời gian xử lý**: dao động khoảng 15 giây đến vài phút tuỳ số lượng xe trong ảnh và cấu hình máy (xem mục [Giới hạn & lưu ý](#giới-hạn--lưu-ý) bên dưới).

### 2. Chạy batch xử lý hàng loạt

Đưa các ảnh cần xử lý vào thư mục `input/`, sau đó chạy:

```bash
python detect_cars.py
```

Kết quả xuất ra thư mục `output/`:
- `<ten_anh>_boxed.png`: ảnh đã vẽ khung + đánh số cho từng ảnh gốc
- `So_luong_xe.xlsx`: báo cáo Excel gồm 2 sheet — `Tong_hop` (số xe mỗi ảnh) và `Chi_tiet_tung_xe` (màu + biển số từng xe)
- `results.json`: dữ liệu chi tiết dạng JSON (dùng để dựng báo cáo tuỳ ý)

## Cấu trúc thư mục

```
car-counter-webapp-v2/
├── app.py                 # Web app Flask
├── detect_cars.py         # Script xử lý hàng loạt ảnh trong input/
├── vehicle_attrs.py        # Module nhận diện màu xe + đọc biển số (EasyOCR)
├── requirements.txt        # Danh sách thư viện cần cài
├── yolov8x.pt              # Trọng số YOLOv8x (tự tải nếu chưa có)
├── templates/
│   └── index.html          # Giao diện web
├── input/                  # Ảnh đầu vào cho detect_cars.py
└── output/                 # Kết quả xuất ra từ detect_cars.py
    ├── debug/               # (tuỳ chọn) ảnh debug quá trình OCR biển số
    ├── *_boxed.png
    ├── So_luong_xe.xlsx
    └── results.json
```

## Giới hạn & lưu ý

- **Biển số ở xa / ảnh chụp toàn cảnh**: nếu biển số trong ảnh gốc chỉ còn vài chục pixel (ảnh chụp bãi xe từ xa), khả năng đọc được rất thấp — đây là giới hạn vật lý của ảnh đầu vào, không phải lỗi thuật toán. Để có kết quả tốt nhất, nên dùng ảnh chụp cận cảnh, biển số rõ nét.
- **Tốc độ xử lý**: mỗi xe có thể chạy nhiều lượt OCR (đọc lần đầu + zoom lại + thử nhiều biến thể ảnh) để tăng độ chính xác, nên thời gian xử lý không nhanh — đặc biệt với ảnh có nhiều xe. Chạy trên CPU sẽ chậm hơn đáng kể so với GPU (mặc định `gpu=False` để chạy được trên mọi máy).
- **Độ chính xác không tuyệt đối**: hệ thống ưu tiên **không đoán bừa** hơn là cố đưa ra kết quả — khi không đủ tin cậy, biển số sẽ hiển thị "không nhận diện được" thay vì đoán sai.

## License

_(Thêm license phù hợp, ví dụ MIT, nếu bạn muốn công khai mã nguồn)_
