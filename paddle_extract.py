import sys
import io
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def crop_percent_zone(image: Image.Image, zone: Dict[str, float]) -> Image.Image:
    width, height = image.size

    left = int(round(width * float(zone.get("left", 0))))
    top = int(round(height * float(zone.get("top", 0))))
    zone_width = int(round(width * float(zone.get("width", 1))))
    zone_height = int(round(height * float(zone.get("height", 1))))

    left = clamp(left, 0, max(width - 1, 0))
    top = clamp(top, 0, max(height - 1, 0))
    right = clamp(left + max(zone_width, 1), left + 1, width)
    bottom = clamp(top + max(zone_height, 1), top + 1, height)

    return image.crop((left, top, right, bottom))


def _collect_from_legacy(lines, texts: List[str], confidences: List[float]) -> None:
    # Legacy format: [[bbox, [text, conf]], ...]
    for line in lines or []:
        if not line or len(line) < 2:
            continue
        rec = line[1]
        if not rec or len(rec) < 2:
            continue
        text = str(rec[0]).strip()
        conf = float(rec[1])
        if text:
            texts.append(text)
            confidences.append(conf)


def _collect_from_new(item, texts: List[str], confidences: List[float]) -> bool:
    # Newer PaddleOCR result objects may expose rec_texts/rec_scores.
    rec_texts = getattr(item, "rec_texts", None)
    rec_scores = getattr(item, "rec_scores", None)

    if rec_texts is None and isinstance(item, dict):
        rec_texts = item.get("rec_texts")
        rec_scores = item.get("rec_scores")

    if not rec_texts:
        return False

    if rec_scores is None:
        rec_scores = [1.0] * len(rec_texts)

    for idx, text in enumerate(rec_texts):
        clean = str(text).strip()
        if not clean:
            continue
        score = float(rec_scores[idx]) if idx < len(rec_scores) else 1.0
        texts.append(clean)
        confidences.append(score)

    return True


def ocr_text(ocr: PaddleOCR, pil_img: Image.Image) -> Tuple[str, float]:
    np_img = np.array(pil_img)
    if np_img.size == 0:
        return "", 0.0

    try:
        results = ocr.predict(np_img)
    except Exception:
        # Legacy API fallback.
        results = ocr.ocr(np_img)

    texts: List[str] = []
    confidences: List[float] = []

    if isinstance(results, list) and results and isinstance(results[0], list):
        _collect_from_legacy(results[0], texts, confidences)
    elif isinstance(results, list):
        for item in results:
            if _collect_from_new(item, texts, confidences):
                continue
            if isinstance(item, list):
                _collect_from_legacy(item, texts, confidences)

    merged = " ".join(texts).strip()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return merged, avg_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR on carte grise zones.")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument("zones_json", help="JSON object with vin and matricule zone percentages")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        zones = json.loads(args.zones_json)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Invalid zones json: {exc}"}))
        return 1

    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Unable to open image: {exc}"}))
        return 1

    try:
        # English model works well for VIN and Latin labels.
        try:
            ocr = PaddleOCR(use_textline_orientation=True, lang="en")
        except TypeError:
            ocr = PaddleOCR(use_angle_cls=True, lang="en")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"PaddleOCR init failed: {exc}"}))
        return 1

    vin_zone = zones.get("vin") or {}
    matricule_zone = zones.get("matricule") or {}

    vin_crop = crop_percent_zone(image, vin_zone)
    vin_text, vin_conf = ocr_text(ocr, vin_crop)

    left_crop = crop_percent_zone(image, matricule_zone)
    left_variants = [
        left_crop,
        left_crop.rotate(90, expand=True),
        left_crop.rotate(270, expand=True),
    ]

    left_text_candidates: List[Tuple[str, float]] = []
    for variant in left_variants:
        text, conf = ocr_text(ocr, variant)
        if text:
            left_text_candidates.append((text, conf))

    left_text_candidates.sort(key=lambda x: x[1], reverse=True)
    left_best_text = left_text_candidates[0][0] if left_text_candidates else ""
    left_best_conf = left_text_candidates[0][1] if left_text_candidates else 0.0

    payload = {
        "ok": True,
        "vin": {
            "text": vin_text,
            "confidence": vin_conf,
        },
        "matricule": {
            "text": left_best_text,
            "confidence": left_best_conf,
            "candidates": [
                {"text": text, "confidence": conf} for text, conf in left_text_candidates
            ],
        },
    }

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
