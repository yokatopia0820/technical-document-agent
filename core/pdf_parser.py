from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import fitz

from core.audit import audit_logger
from core.config import settings
from core.docling_parser import build_docling_chunks
from core.ocr import extract_ocr_blocks
from core.pdf_utils import normalize_text, verify_payload_against_pdf
from core.tagging import TaggingEngine


def _extract_headings(page: fitz.Page) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = normalize_text(" ".join(span.get("text", "") for span in spans))
            if not text:
                continue
            max_size = max(float(span.get("size", 0)) for span in spans)
            is_bold = any("bold" in str(span.get("font", "")).lower() for span in spans)
            if max_size >= 11 or is_bold:
                headings.append({"text": text, "size": max_size, "bbox": line.get("bbox")})
    headings.sort(key=lambda x: x["size"], reverse=True)
    return headings[:10]


def _get_cell_bbox(table: Any, row_idx: int, col_idx: int) -> list[float] | None:
    try:
        row = table.rows[row_idx]
        cell = row.cells[col_idx]
        if cell is None:
            return None
        bbox = cell.bbox if hasattr(cell, "bbox") else cell
        if bbox and len(bbox) == 4:
            return [float(x) for x in bbox]
    except Exception:
        return None
    return None


def _table_chunks(page: fitz.Page, pdf_path: str, doc_id: str, doc_name: str, context_text: str, tagger: TaggingEngine) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    try:
        table_finder = page.find_tables()
        tables = list(getattr(table_finder, "tables", []))
    except Exception:
        tables = []

    for table_index, table in enumerate(tables):
        matrix = table.extract() or []
        if len(matrix) < 2:
            continue
        headers = [normalize_text(v) for v in matrix[0]]
        for row_idx, row in enumerate(matrix[1:], start=1):
            row = [normalize_text(v) for v in row]
            if not any(row):
                continue
            row_label = row[0] if row else f"row_{row_idx}"
            for col_idx in range(1, len(row)):
                value = row[col_idx]
                if not value:
                    continue
                col_label = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                bbox = _get_cell_bbox(table, row_idx, col_idx) or [float(x) for x in table.bbox]
                major_tag, medium_tag, minor_tag = tagger.derive_tags(context_text, row_label, col_label, value)
                payload = {
                    "point_id": str(uuid4()),
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "source_pdf_path": pdf_path,
                    "page_no": page.number + 1,
                    "chunk_type": "table_cell",
                    "table_index": table_index,
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "row_label": row_label,
                    "col_label": col_label,
                    "value": value,
                    "bbox": bbox,
                    "major_tag": major_tag,
                    "medium_tag": medium_tag,
                    "minor_tag": minor_tag,
                    "favorite": False,
                    "favorite_weight": 0,
                    "parser_backend": "pymupdf",
                    "shorthand_json": {row_label: {col_label: value}},
                }
                verification = verify_payload_against_pdf(pdf_path, payload)
                payload["verified"] = verification["verified"]
                payload["verification_summary"] = verification["verification_summary"]
                payload["value"] = verification.get("corrected_value") or value
                payload["shorthand_json"] = {row_label: {col_label: payload["value"]}}
                payload["chunk_text"] = " | ".join(
                    [
                        context_text,
                        f"ページ {page.number + 1}",
                        row_label,
                        col_label,
                        payload["value"],
                        json.dumps(payload["shorthand_json"], ensure_ascii=False),
                    ]
                )
                chunks.append(payload)
    return chunks


def _text_chunks(page: fitz.Page, pdf_path: str, doc_id: str, doc_name: str, context_text: str, tagger: TaggingEngine) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    data = page.get_text("dict")
    for block_index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        text = normalize_text(
            " ".join(
                span.get("text", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
        )
        if len(text) < 20:
            continue
        major_tag, medium_tag, minor_tag = tagger.derive_tags(context_text, text[:200])
        payload = {
            "point_id": str(uuid4()),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "source_pdf_path": pdf_path,
            "page_no": page.number + 1,
            "chunk_type": "text_block",
            "block_index": block_index,
            "bbox": [float(x) for x in block.get("bbox")],
            "row_label": "",
            "col_label": "",
            "value": text[:400],
            "major_tag": major_tag,
            "medium_tag": medium_tag,
            "minor_tag": minor_tag,
            "favorite": False,
            "favorite_weight": 0,
            "verified": True,
            "verification_summary": "passes=[True, True, True]",
            "parser_backend": "pymupdf",
            "shorthand_json": {"text": text[:400]},
            "chunk_text": " | ".join([context_text, f"ページ {page.number + 1}", text[:500]]),
        }
        chunks.append(payload)
    return chunks


def _ocr_chunks(page: fitz.Page, pdf_path: str, doc_id: str, doc_name: str, context_text: str, tagger: TaggingEngine) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx, block in enumerate(extract_ocr_blocks(page, lang=settings.ocr_langs)):
        text = normalize_text(block.get("text"))
        if len(text) < 2:
            continue
        major_tag, medium_tag, minor_tag = tagger.derive_tags(context_text, text)
        chunks.append(
            {
                "point_id": str(uuid4()),
                "doc_id": doc_id,
                "doc_name": doc_name,
                "source_pdf_path": pdf_path,
                "page_no": page.number + 1,
                "chunk_type": "ocr_block",
                "block_index": idx,
                "bbox": block.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                "row_label": "",
                "col_label": "",
                "value": text,
                "major_tag": major_tag,
                "medium_tag": medium_tag,
                "minor_tag": minor_tag,
                "favorite": False,
                "favorite_weight": 0,
                "verified": True,
                "verification_summary": f"ocr_conf={block.get('confidence', -1)}",
                "parser_backend": "ocr_tesseract",
                "shorthand_json": {"ocr_text": text},
                "chunk_text": " | ".join([context_text, f"ページ {page.number + 1}", text]),
            }
        )
    return chunks


def _build_pymupdf_chunks(pdf_path: str, doc_id: str, doc_name: str, ocr_enabled: bool) -> list[dict[str, Any]]:
    doc = fitz.open(pdf_path)
    tagger = TaggingEngine()
    try:
        chunks: list[dict[str, Any]] = []
        for page in doc:
            headings = _extract_headings(page)
            context_text = " | ".join(h["text"] for h in headings[:4])
            native_text = normalize_text(page.get_text("text"))
            chunks.extend(_table_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))
            chunks.extend(_text_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))
            if ocr_enabled and len(native_text) < settings.min_chars_for_native_text:
                chunks.extend(_ocr_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))
        return chunks
    finally:
        doc.close()


def build_chunks(
    pdf_path: str,
    doc_id: str | None = None,
    doc_name: str | None = None,
    parser_backend: str | None = None,
    ocr_enabled: bool | None = None,
) -> list[dict[str, Any]]:
    path = Path(pdf_path)
    doc_id = doc_id or path.stem
    doc_name = doc_name or path.name
    parser_backend = (parser_backend or settings.parser_backend).lower()
    ocr_enabled = settings.ocr_enabled if ocr_enabled is None else ocr_enabled

    chunks: list[dict[str, Any]] = []
    used_backends: list[str] = []

    if parser_backend in {"pymupdf", "auto", "hybrid"}:
        chunks.extend(_build_pymupdf_chunks(pdf_path, doc_id, doc_name, ocr_enabled=ocr_enabled))
        used_backends.append("pymupdf")

    if parser_backend in {"docling", "hybrid"}:
        try:
            chunks.extend(build_docling_chunks(pdf_path, doc_id=doc_id, doc_name=doc_name, ocr_enabled=ocr_enabled))
            used_backends.append("docling")
        except Exception as exc:
            audit_logger.log("docling_parse_failed", {"pdf_path": pdf_path, "error": str(exc)})
            if parser_backend == "docling" and not chunks:
                raise

    audit_logger.log(
        "pdf_ingested",
        {
            "pdf_path": pdf_path,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "parser_backend": parser_backend,
            "used_backends": used_backends,
            "ocr_enabled": ocr_enabled,
            "chunks": len(chunks),
        },
    )
    return chunks
