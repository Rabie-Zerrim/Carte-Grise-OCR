import sys
import json
import os
sys.stdout.reconfigure(encoding='utf-8')

import re
from docstrange import DocumentExtractor

image_path = sys.argv[1]

if not os.path.splitext(image_path)[1].lower():
    new_path = image_path + '.jpg'
    os.rename(image_path, new_path)
    image_path = new_path

extractor = DocumentExtractor()

result = extractor.extract(file_path=image_path)

text = result.extract_text() if hasattr(result, 'extract_text') else str(result.content or '')

import re
vin = None
matricule = None

vin_match = re.search(r'([A-HJ-NPR-Z0-9]{17})', text.upper())
if vin_match:
    vin = vin_match.group(1)

mat_match = re.search(r'(\d{4})\s*(?:تونس|TUNIS|TU)\s*(\d{3})', text.upper())
if mat_match:
    matricule = mat_match.group(1) + 'TU' + mat_match.group(2)

payload = {
    "ok": True,
    "vin": {"text": vin or "", "confidence": 0.95},
    "matricule": {"text": matricule or "", "confidence": 0.95, "candidates": []},
    "all_texts": text
}

print(json.dumps(payload, ensure_ascii=False))