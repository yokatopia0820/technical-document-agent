from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from core.audit import audit_logger
from core.pdf_utils import normalize_text
from core.tagging import TaggingEngine


def build_docling_chunks(pdf_path: str, doc_id: str | None = None, doc_name: str | None = None, ocr_enabled: bool = True) -> list[dict[str, Any]]:
    from docling.document_converter import DocumentConverter

    path = Path(pdf_path)
    doc_id = doc_id or path.stem
    doc_name = doc_name or path.name
    tagger = TaggingEngine()

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    document = result.document

    chunks: list[dict[str, Any]] = []
    markdown = document.export_to_markdown()
    for index, raw_chunk in enumerate(markdown.split("\n\n")):
        text = normalize_text(raw_chunk)
        if len(text) < 20:
            continue
        major_tag, medium_tag, minor_tag = tagger.derive_tags(text)
        chunks.append(
            {
                "point_id": str(uuid4()),
                "doc_id": doc_id,
                "doc_name": doc_name,
                "source_pdf_path": pdf_path,
                "page_no": 1,
                "chunk_type": "docling_markdown",
                "block_index": index,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "row_label": "",
                "col_label": "",
                "value": text[:500],
                "major_tag": major_tag,
                "medium_tag": medium_tag,
                "minor_tag": minor_tag,
                "favorite": False,
                "favorite_weight": 0,
                "verified": True,
                "verification_summary": "docling-structured-export",
                "shorthand_json": {"text": text[:500]},
                "parser_backend": "docling",
                "chunk_text": text[:800],
            }
        )
    audit_logger.log("docling_parse_complete", {"pdf_path": pdf_path, "chunks": len(chunks), "ocr_enabled": ocr_enabled})
    return chunks
