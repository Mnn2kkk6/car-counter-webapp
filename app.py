import base64
import io
import os

import torch
from flask import Flask, request, jsonify, render_template
from torchvision.ops import nms
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from vehicle_attrs import analyze_vehicle

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
CONF_THRES = 0.35
IOU_THRES = 0.5

app = Flask(__name__)

# Load model MỘT LẦN khi khởi động server (không load lại mỗi request -> nhanh)
print("Đang tải mô hình YOLOv8x, vui lòng chờ...")
model = YOLO("yolov8x.pt")
print("Đã sẵn sàng nhận ảnh.")

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except Exception:
    FONT = ImageFont.load_default()


def detect_and_draw(im: Image.Image):
    """Chạy YOLOv8x + NMS không phân biệt lớp, vẽ bounding box, nhận diện màu + biển số từng xe.
    Trả về (ảnh_đã_vẽ, danh_sách_thông_tin_từng_xe)."""
    im = im.convert("RGB")
    r = model.predict(im, conf=CONF_THRES, iou=0.5, imgsz=1280, verbose=False)[0]
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

    draw = ImageDraw.Draw(im)
    count = 0
    vehicles = []
    for box in final_boxes:
        count += 1
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        label = str(count)
        tb = draw.textbbox((0, 0), label, font=FONT)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 8, y1], fill=(255, 0, 0))
        draw.text((x1 + 4, y1 - th - 6), label, fill=(255, 255, 255), font=FONT)

        crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
        info = analyze_vehicle(crop, count)
        vehicles.append(info)

    return im, vehicles


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh được gửi lên"}), 400

    file = request.files["image"]
    try:
        im = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "File không phải ảnh hợp lệ"}), 400

    result_im, vehicles = detect_and_draw(im)

    buf = io.BytesIO()
    result_im.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "count": len(vehicles),
        "image": f"data:image/png;base64,{b64_img}",
        "vehicles": [
            {"index": v["index"], "color": v["color"], "plate": v["plate"], "label": v["label"]}
            for v in vehicles
        ],
    })


if __name__ == "__main__":
    # Chạy local, truy cập tại http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
