from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u3000", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def save_uploaded_pdf(uploaded_file, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    target = Path(output_dir) / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def _extract_candidate_value(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"\d+[\d,]*(?:\.\d+)?\s?(?:円|kW|KW|V|A|W|mm|cm|kg|%|型)",
        r"[A-Z]{1,5}[\-A-Z0-9]{2,}",
        r"\d+[\d,]*(?:\.\d+)?",
    ]
    for pattern in patterns:
        matched = re.findall(pattern, text)
        if matched:
            return normalize_text(matched[0])
    return normalize_text(text[:80])


def verify_payload_against_pdf(pdf_path: str, payload: dict[str, Any], max_passes: int = 3) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    try:
        page_no = max(int(payload.get("page_no", 1)) - 1, 0)
        page = doc[page_no]
        bbox = payload.get("bbox")
        value = normalize_text(payload.get("value"))
        row_label = normalize_text(payload.get("row_label"))
        col_label = normalize_text(payload.get("col_label"))

        clip_text = ""
        neighborhood_text = ""
        if bbox and len(bbox) == 4:
            rect = fitz.Rect(*bbox)
            expanded = rect + (-20, -12, 20, 12)
            clip_text = normalize_text(page.get_textbox(rect))
            neighborhood_text = normalize_text(page.get_textbox(expanded))

        whole_page = normalize_text(page.get_text("text"))
        pass_results: list[bool] = []

        pass_results.append(bool(value) and value in clip_text)
        pass_results.append(bool(value) and any(value in normalize_text(page.get_textbox(r)) for r in page.search_for(value)))
        pass_results.append(
            (not row_label or row_label in neighborhood_text or row_label in whole_page)
            and (not col_label or col_label in neighborhood_text or col_label in whole_page)
        )

        corrected_value = value
        if not pass_results[0] and clip_text:
            corrected_value = _extract_candidate_value(clip_text)

        return {
            "verified": sum(bool(x) for x in pass_results[:max_passes]) >= 2,
            "verification_summary": f"passes={pass_results[:max_passes]}",
            "corrected_value": corrected_value,
            "clip_text": clip_text,
            "neighborhood_text": neighborhood_text,
        }
    finally:
        doc.close()


def build_highlighted_pdf_bytes(pdf_path: str, payload: dict[str, Any]) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page_no = max(int(payload.get("page_no", 1)) - 1, 0)
        page = doc[page_no]
        target_value = normalize_text(payload.get("value"))
        bbox = payload.get("bbox")

        if target_value:
            quads = page.search_for(target_value, quads=True)
            if quads:
                page.add_highlight_annot(quads)
        elif bbox and len(bbox) == 4:
            rect = fitz.Rect(*bbox)
            page.add_rect_annot(rect)

        return doc.tobytes()
    finally:
        doc.close()
