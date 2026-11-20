import json
import re
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
import numpy as np
import easyocr

image_path = sys.argv[1]
zones_file = sys.argv[2]

if os.path.exists(zones_file):
    with open(zones_file, 'r', encoding='utf-8') as f:
        zones = json.load(f)
else:
    zones = {"vin": {}, "matricule": {}}

image = Image.open(image_path).convert("RGB")
w, h = image.size

if h > w * 1.5:
    rot_img = image.rotate(90, expand=True)
    image = rot_img
    w, h = image.size
    print(f"Rotated: {w}x{h}", file=sys.stderr)

np_img = np.array(image)

reader = easyocr.Reader(['ar', 'fa', 'ur', 'ug', 'en'], gpu=False, verbose=False)
results = reader.readtext(np_img)

all_texts = []
for item in results:
    if len(item) >= 3:
        bbox = item[0]
        text = str(item[1]).strip()
        conf = float(item[2])
        if text:
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

def find_matricule_in_text(text):
    text_upper = text.upper().replace(' ', '').replace('_', '').replace('-', '')
    
    patterns = [
        r'(\d{4})\s*تونس\s*(\d{3})',
        r'(\d{4})\s*TUNIS\s*(\d{3})',
        r'(\d{4})\s*T\s*U\s*(\d{3})',
        r'(\d{4})\s*TU\s*(\d{3})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.upper())
        if match:
            return match.group(1) + 'TU' + match.group(2)
    
    tunis_match = re.search(r'(\d{4})\s*تونس', text.upper())
    if tunis_match:
        import random
        return None
    
    if len(text_upper) >= 7:
        for i in range(len(text_upper) - 6):
            chunk = text_upper[i:i+7]
            if chunk[:4].isdigit() and chunk[4:].isdigit():
                return chunk[:4] + 'TU' + chunk[4:]
    
    return None

matricule_candidates = []

for item in all_texts:
    found = find_matricule_in_text(item["text"])
    if found and not any(c["text"] == found for c in matricule_candidates):
        matricule_candidates.append({"text": found, "confidence": item["confidence"], "bbox": item.get("bbox")})

matricule_candidates.sort(key=lambda x: -x["confidence"])

immat_texts = [t for t in all_texts if 'IMMAT' in t["text"].upper()]
if immat_texts and matricule_candidates:
    for cand in matricule_candidates:
        if cand.get("bbox"):
            immat_bbox = immat_texts[0]["bbox"]
            immat_cx = (immat_bbox[0][0] + immat_bbox[1][0]) / 2
            cand_cx = (cand["bbox"][0][0] + cand["bbox"][1][0]) / 2
            if abs(immat_cx - cand_cx) < 500:
                matricule_candidates.insert(0, matricule_candidates.pop(matricule_candidates.index(cand)))
                break

matricule_text = " ".join(mat_texts)
if matricule_candidates:
    matricule_text = matricule_candidates[0]["text"]
    mat_texts = []

def normalizeOCRText(text):
    return (text or '') \
        .replace('|', 'I') \
        .replace('\r', '\n') \
        .replace('\n+', '\n') \
        .replace(' +', ' ') \
        .strip()

payload = {
    "ok": True,
    "vin": {
        "text": normalizeOCRText(" ".join(vin_texts)),
        "confidence": float(sum(t["confidence"] for t in all_texts if t["text"] in vin_texts) / max(1, len(vin_texts))) if vin_texts else 0.0
    },
    "matricule": {
        "text": normalizeOCRText(matricule_text),
        "confidence": float(matricule_candidates[0]["confidence"]) if matricule_candidates else 0.0,
        "candidates": [{"text": c["text"], "confidence": c["confidence"]} for c in matricule_candidates[:10]]
    },
    "all_texts": all_texts
}

print(json.dumps(payload, ensure_ascii=False))