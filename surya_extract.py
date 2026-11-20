import sys
import json
import os
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from surya.settings import settings

def normalizeOCRText(text):
    return (text or '') \
        .replace('|', 'I') \
        .replace('\r', '\n') \
        .replace('\n+', '\n') \
        .replace(' +', ' ') \
        .strip()

image_path = sys.argv[1]
zones_file = sys.argv[2]

if os.path.exists(zones_file):
    with open(zones_file, 'r', encoding='utf-8') as f:
        zones = json.load(f)
else:
    zones = {"vin": {}, "matricule": {}}

image = Image.open(image_path).convert("RGB")
w, h = image.size

detector = DetectionPredictor()
recognizer = RecognitionPredictor(FoundationPredictor())

det_results = detector.batch_detection([image])
rec_results = recognizer.batch Recognition(det_results, [image])

all_texts = []
for result in rec_results:
    for line in result.text_lines:
        text = line.text if hasattr(line, 'text') else str(line)
        conf = line.confidence if hasattr(line, 'confidence') else 0.5
        bbox = getattr(line, 'bbox', None)
        if text and bbox:
            clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
            all_texts.append({"text": text, "confidence": float(conf), "bbox": clean_bbox})

vin_zone = zones.get("vin") or {}
matricule_zone = zones.get("matricule") or {}

def get_zone_bounds(zone):
    return {
        "left": int(w * zone.get("left", 0)),
        "top": int(h * zone.get("top", 0)),
        "width": int(w * zone.get("width", 1)),
        "height": int(h * zone.get("height", 1))
    }

vin_bounds = get_zone_bounds(vin_zone)
mat_bounds = get_zone_bounds(matricule_zone)

vin_texts = []
mat_texts = []

for item in all_texts:
    text = item["text"]
    bbox = item.get("bbox")
    if bbox and len(bbox) == 4:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if vin_bounds["left"] <= cx <= vin_bounds["left"] + vin_bounds["width"] and \
           vin_bounds["top"] <= cy <= vin_bounds["top"] + vin_bounds["height"]:
            vin_texts.append(text)
        if mat_bounds["left"] <= cx <= mat_bounds["left"] + mat_bounds["width"] and \
           mat_bounds["top"] <= cy <= mat_bounds["top"] + mat_bounds["height"]:
            mat_texts.append(text)

def avg_conf(items, text_list):
    if not text_list: return 0.0
    return float(sum(t["confidence"] for t in items if t["text"] in text_list) / max(1, len(text_list)))

payload = {
    "ok": True,
    "vin": {
        "text": normalizeOCRText(" ".join(vin_texts)),
        "confidence": avg_conf(all_texts, vin_texts)
    },
    "matricule": {
        "text": normalizeOCRText(" ".join(mat_texts)),
        "confidence": avg_conf(all_texts, mat_texts),
        "candidates": []
    },
    "all_texts": all_texts
}

print(json.dumps(payload, ensure_ascii=False))