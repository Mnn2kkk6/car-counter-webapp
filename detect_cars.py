import os
import json
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from vehicle_attrs import analyze_vehicle

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
CONF_THRES = 0.35
IOU_THRES = 0.5

# Lần chạy đầu tiên, ultralytics sẽ tự tải trọng số yolov8x.pt (~130MB) về thư mục hiện tại
model = YOLO("yolov8x.pt")

INPUT_DIR = "input"
OUTPUT_DIR = "output"

images = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

os.makedirs(OUTPUT_DIR, exist_ok=True)
results_rows = []

for orig_name in images:
    fname = os.path.join(INPUT_DIR, orig_name)
    r = model.predict(fname, conf=CONF_THRES, iou=0.5, imgsz=1280, verbose=False)[0]
    names = r.names

    boxes, scores = [], []
    for box in r.boxes:
        cls_name = names[int(box.cls[0])]
        if cls_name not in VEHICLE_CLASSES:
            continue
        boxes.append(box.xyxy[0].tolist())
        scores.append(float(box.conf[0]))

    if boxes:
        boxes_t = torch.tensor(boxes)
        scores_t = torch.tensor(scores)
        keep = nms(boxes_t, scores_t, IOU_THRES)
        final_boxes = boxes_t[keep].tolist()
    else:
        final_boxes = []

    im = Image.open(fname).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    count = 0
    vehicle_details = []
    for box in final_boxes:
        count += 1
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        label = str(count)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 8, y1], fill=(255, 0, 0))
        draw.text((x1 + 4, y1 - th - 6), label, fill=(255, 255, 255), font=font)

        # Cắt riêng từng xe để nhận diện màu + biển số
        crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
        debug_path = os.path.join(OUTPUT_DIR, "debug", f"{os.path.splitext(orig_name)[0]}_xe{count}")
        info = analyze_vehicle(crop, count, debug_dir=debug_path)
        vehicle_details.append(info)
        print("  ", info["label"])

    boxed_name = f"{os.path.splitext(orig_name)[0]}_boxed.png"
    out_path = os.path.join(OUTPUT_DIR, boxed_name)
    im.save(out_path)
    results_rows.append({
        "STT": len(results_rows) + 1,
        "Ten_anh": orig_name,
        "So_luong_xe": count,
        "Anh_ket_qua": boxed_name,
        "Width": im.width,
        "Height": im.height,
        "Chi_tiet_xe": vehicle_details,
    })
    print(orig_name, "->", count, "vehicles")

# Bảng Excel: 1 dòng/ảnh + 1 bảng chi tiết từng xe (màu, biển số)
summary_df = pd.DataFrame([
    {"STT": r["STT"], "Ten_anh": r["Ten_anh"], "So_luong_xe": r["So_luong_xe"]}
    for r in results_rows
])

detail_rows = []
for r in results_rows:
    for v in r["Chi_tiet_xe"]:
        detail_rows.append({
            "Ten_anh": r["Ten_anh"],
            "Xe_so": v["index"],
            "Mau_xe": v["color"] or "Không xác định",
            "Bien_so": v["plate"] or "Không nhận diện được",
        })
detail_df = pd.DataFrame(detail_rows)

with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "So_luong_xe.xlsx")) as writer:
    summary_df.to_excel(writer, sheet_name="Tong_hop", index=False)
    detail_df.to_excel(writer, sheet_name="Chi_tiet_tung_xe", index=False)

# Ghi kèm results.json để build_report.js tự đọc và dựng báo cáo cho MỌI ảnh
# trong input/, không cần sửa code khi thêm/bớt ảnh.
with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results_rows, f, ensure_ascii=False, indent=2)

print(summary_df)
