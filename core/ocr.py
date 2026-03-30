from __future__ import annotations

from io import BytesIO
from typing import Any

import fitz
from PIL import Image
import pytesseract

from core.pdf_utils import normalize_text


OCR_DATA_KEYS = ["text", "left", "top", "width", "height", "conf"]


def page_to_image(page: fitz.Page, dpi: int = 220) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.open(BytesIO(pix.tobytes("png")))


def extract_ocr_blocks(page: fitz.Page, lang: str = "jpn+eng") -> list[dict[str, Any]]:
    image = page_to_image(page)
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    blocks: list[dict[str, Any]] = []
    for i, text in enumerate(data.get("text", [])):
        cleaned = normalize_text(text)
        if not cleaned:
            continue
        bbox = [
            float(data["left"][i]),
            float(data["top"][i]),
            float(data["left"][i] + data["width"][i]),
            float(data["top"][i] + data["height"][i]),
        ]
        blocks.append(
            {
                "text": cleaned,
                "bbox": bbox,
                "confidence": float(data["conf"][i]) if str(data["conf"][i]).strip() not in {"", "-1"} else -1.0,
            }
        )
    return blocks
