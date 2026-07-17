# Car Counter Web App

Đếm xe ô tô, vẽ bounding box, nhận diện màu xe và đọc biển số (best-effort) từ ảnh — chạy local qua trình duyệt hoặc terminal.

## 1. Cài đặt

### 1.1. Thư viện Python
```bash
pip install -r requirements.txt
```

### 1.2. Tesseract OCR (bắt buộc riêng, KHÔNG cài qua pip)
Tính năng đọc biển số dùng thư viện `pytesseract`, nhưng đây chỉ là lớp gọi tới engine
**Tesseract OCR** — cần cài engine này riêng trên máy:

- **Windows**: tải và cài từ https://github.com/UB-Mannheim/tesseract/wiki (bản UB-Mannheim).
  Sau khi cài, thêm dòng sau vào đầu `vehicle_attrs.py` nếu Tesseract không tự vào PATH:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```
- **macOS**: `brew install tesseract`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`

Nếu không cài Tesseract, ứng dụng **vẫn chạy bình thường** (đếm xe + vẽ bounding box + nhận
diện màu vẫn hoạt động) — chỉ riêng phần đọc biển số sẽ luôn trả về "không nhận diện được".

### 1.3. Thư viện Node.js (chỉ cần nếu dùng build_report.js để xuất báo cáo Word)
```bash
npm install
```

## 2. Chạy

### Web app (khuyến nghị — kéo thả ảnh, xem kết quả ngay trên trình duyệt)
```bash
python app.py
```
Mở trình duyệt vào `http://127.0.0.1:5000`.

### Chạy hàng loạt qua terminal (không cần trình duyệt)
```bash
python detect_cars.py
```
Xử lý toàn bộ ảnh trong `input/`, lưu ảnh + Excel (`So_luong_xe.xlsx`, có 2 sheet:
tổng hợp và chi tiết từng xe) vào `output/`.

### Xuất báo cáo Word hoàn chỉnh
```bash
node build_report.js
```

## 3. Giới hạn cần biết

- **Màu xe**: ước lượng bằng màu chiếm đa số trên vùng thân xe — có thể sai với xe
  kính đen lớn, xe 2 tông màu (nóc đen), hoặc bị cây/bóng che khuất nhiều.
- **Biển số**: với ảnh chụp góc cao/xa, biển số thường quá nhỏ hoặc mờ để OCR đọc
  chính xác — đây là giới hạn thực tế của ảnh đầu vào, không phải lỗi thuật toán.
  Khi không đọc được, hệ thống trả về "không nhận diện được" thay vì đoán bừa.
- Ảnh càng rõ nét, càng gần, càng ít bị che khuất thì độ chính xác màu/biển số càng cao.
