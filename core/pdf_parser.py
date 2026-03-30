from __future__ import annotations

import json
from collections.abc import Callable
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
    try:
        data = page.get_text("dict")
    except Exception:
        return headings

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


def _table_chunks(
    page: fitz.Page,
    pdf_path: str,
    doc_id: str,
    doc_name: str,
    context_text: str,
    tagger: TaggingEngine,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    try:
        table_finder = page.find_tables()
        tables = list(getattr(table_finder, "tables", []))
    except Exception:
        tables = []

    for table_index, table in enumerate(tables):
        try:
            matrix = table.extract() or []
        except Exception:
            matrix = []

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
                bbox = _get_cell_bbox(table, row_idx, col_idx)
                if bbox is None:
                    try:
                        bbox = [float(x) for x in table.bbox]
                    except Exception:
                        bbox = [0.0, 0.0, 0.0, 0.0]

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

                try:
                    verification = verify_payload_against_pdf(pdf_path, payload)
                    payload["verified"] = verification["verified"]
                    payload["verification_summary"] = verification["verification_summary"]
                    payload["value"] = verification.get("corrected_value") or value
                    payload["shorthand_json"] = {row_label: {col_label: payload["value"]}}
                except Exception as exc:
                    payload["verified"] = False
                    payload["verification_summary"] = f"verify_failed: {exc}"

                payload["chunk_text"] = " | ".join(
                    [
                        context_text,
                        f"{page.number + 1}",
                        row_label,
                        col_label,
                        str(payload["value"]),
                        json.dumps(payload["shorthand_json"], ensure_ascii=False),
                    ]
                )
                chunks.append(payload)

    return chunks


def _text_chunks(
    page: fitz.Page,
    pdf_path: str,
    doc_id: str,
    doc_name: str,
    context_text: str,
    tagger: TaggingEngine,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    try:
        data = page.get_text("dict")
    except Exception:
        return chunks

    for block_index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue

        try:
            text = normalize_text(
                " ".join(
                    span.get("text", "")
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                )
            )
        except Exception:
            continue

        if len(text) < 20:
            continue

        major_tag, medium_tag, minor_tag = tagger.derive_tags(context_text, text[:200])

        bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        try:
            bbox = [float(x) for x in bbox]
        except Exception:
            bbox = [0.0, 0.0, 0.0, 0.0]

        payload = {
            "point_id": str(uuid4()),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "source_pdf_path": pdf_path,
            "page_no": page.number + 1,
            "chunk_type": "text_block",
            "block_index": block_index,
            "bbox": bbox,
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
            "chunk_text": " | ".join([context_text, f"{page.number + 1}", text[:500]]),
        }
        chunks.append(payload)

    return chunks


def _ocr_chunks(
    page: fitz.Page,
    pdf_path: str,
    doc_id: str,
    doc_name: str,
    context_text: str,
    tagger: TaggingEngine,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    try:
        ocr_blocks = extract_ocr_blocks(page, lang=settings.ocr_langs)
    except Exception as exc:
        audit_logger.log(
            "ocr_extract_failed",
            {
                "pdf_path": pdf_path,
                "doc_name": doc_name,
                "page_no": page.number + 1,
                "error": str(exc),
            },
        )
        return chunks

    for idx, block in enumerate(ocr_blocks):
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
                "chunk_text": " | ".join([context_text, f"{page.number + 1}", text]),
            }
        )

    return chunks


def _build_pymupdf_chunks(
    pdf_path: str,
    doc_id: str,
    doc_name: str,
    ocr_enabled: bool,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    doc = fitz.open(pdf_path)
    tagger = TaggingEngine()

    try:
        chunks: list[dict[str, Any]] = []
        total_pages = len(doc)

        for idx, page in enumerate(doc, start=1):
            try:
                headings = _extract_headings(page)
                context_text = " | ".join(h["text"] for h in headings[:4])

                try:
                    native_text = normalize_text(page.get_text("text"))
                except Exception:
                    native_text = ""

                chunks.extend(_table_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))
                chunks.extend(_text_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))

                if ocr_enabled and len(native_text) < settings.min_chars_for_native_text:
                    chunks.extend(_ocr_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))

            except Exception as exc:
                audit_logger.log(
                    "pymupdf_page_failed",
                    {
                        "pdf_path": pdf_path,
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "page_no": idx,
                        "error": str(exc),
                    },
                )
            finally:
                if progress_callback:
                    progress_callback(idx, total_pages, f"PDF P.{idx}/{total_pages}")

        return chunks
    finally:
        doc.close()


def build_chunks(
    pdf_path: str,
    doc_id: str | None = None,
    doc_name: str | None = None,
    parser_backend: str | None = None,
    ocr_enabled: bool | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    path = Path(pdf_path)
    doc_id = doc_id or path.stem
    doc_name = doc_name or path.name
    parser_backend = (parser_backend or settings.parser_backend).lower()
    ocr_enabled = settings.ocr_enabled if ocr_enabled is None else ocr_enabled

    chunks: list[dict[str, Any]] = []
    used_backends: list[str] = []

    if parser_backend in {"pymupdf", "auto", "hybrid"}:
        chunks.extend(
            _build_pymupdf_chunks(
                pdf_path,
                doc_id,
                doc_name,
                ocr_enabled=ocr_enabled,
                progress_callback=progress_callback,
            )
        )
        used_backends.append("pymupdf")

    if parser_backend in {"docling", "hybrid"}:
        if progress_callback:
            progress_callback(1, 1, "Docling")
        try:
            chunks.extend(build_docling_chunks(pdf_path, doc_id=doc_id, doc_name=doc_name, ocr_enabled=ocr_enabled))
            used_backends.append("docling")
            if progress_callback:
                progress_callback(1, 1, "Docling")
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
Copy
添付画像はこちらです。
画面キャプチャ

これで終わりです。


logs-yokatopia0820-technical-document-agent-main-app.py-2026-03-30T14_23_23.352Z.txt
108.92 KB
ツールを使用する
|
大きな文書を要約
このログの今回の起動失敗の直接原因だけを特定してください。最後のTracebackの要点、例外名、発生ファイルと行番号、原因の説明を日本語で簡潔にまとめてください。
エージェントを表示
ツールを使用する
|
読み取り
https://raw.githubusercontent.com/yokatopia0820/technical-document-agent/main/core/qdrant_store.py
エージェントを表示
ツールを使用する
|
読み取り
https://raw.githubusercontent.com/yokatopia0820/technical-document-agent/main/app.py
エージェントを表示
原因はこれです。
core/qdrant_store.py が壊れており、自分自身を import しているため、循環インポートで起動失敗しています。
ログの直接原因は ImportError: cannot import name 'QdrantStore' from partially initialized module 'core.qdrant_store'、発生箇所は core/qdrant_store.py 15行目です。Source

今は core/qdrant_store.py だけ 直せばよいです。
現在の GitHub 上の core/qdrant_store.py は、Qdrant 用コードではなく app.py のような内容になっていて、from core.qdrant_store import QdrantStore を含んでいます。これが起動不能の原因です。Source

以下を core/qdrant_store.py の全文 としてそのまま使ってください。

Copyfrom __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient, models

from core.audit import audit_logger
from core.config import settings
from core.embeddings import Embedder


@dataclass
class SearchHit:
    point_id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.collection_name = settings.qdrant_collection

        if settings.qdrant_mode == "local":
            self.client = QdrantClient(path=settings.qdrant_local_path)
        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=False,
            )

        self.ensure_collection()

    def ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )

        index_fields = [
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("doc_name", models.PayloadSchemaType.KEYWORD),
            ("major_tag", models.PayloadSchemaType.KEYWORD),
            ("medium_tag", models.PayloadSchemaType.KEYWORD),
            ("minor_tag", models.PayloadSchemaType.KEYWORD),
            ("page_no", models.PayloadSchemaType.INTEGER),
            ("favorite", models.PayloadSchemaType.BOOL),
            ("favorite_weight", models.PayloadSchemaType.INTEGER),
            ("verified", models.PayloadSchemaType.BOOL),
            ("parser_backend", models.PayloadSchemaType.KEYWORD),
            ("chunk_type", models.PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                pass

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 64,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        if not chunks:
            return 0

        total = len(chunks)
        inserted = 0

        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            texts = [str(chunk.get("chunk_text", "")) for chunk in batch]
            vectors = self.embedder.embed(texts)

            if not isinstance(vectors, list):
                vectors = list(vectors)

            points: list[models.PointStruct] = []

            for idx, chunk in enumerate(batch):
                if idx >= len(vectors):
                    continue

                vector = vectors[idx]
                point_id = str(chunk.get("point_id") or uuid4())
                chunk["point_id"] = point_id

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=chunk,
                    )
                )

            if not points:
                continue

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            inserted += len(points)

            if progress_callback:
                progress_callback(inserted, total, "Qdrant登録中")

        audit_logger.log(
            "qdrant_upsert",
            {
                "collection": self.collection_name,
                "points": inserted,
                "mode": settings.qdrant_mode,
            },
        )
        return inserted

    def search(
        self,
        query: str,
        tag_filters: dict[str, list[str]] | None = None,
        limit: int = 12,
    ) -> list[SearchHit]:
        vector = self.embedder.embed([query])[0]
        query_filter = self._build_filter(tag_filters or {})

        raw_points = None

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            raw_points = getattr(response, "points", response)

        elif hasattr(self.client, "search"):
            raw_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        else:
            raise AttributeError("Qdrant client に検索APIが見つかりません。")

        hits: list[SearchHit] = []

        for point in raw_points or []:
            payload = dict(getattr(point, "payload", {}) or {})
            score = float(getattr(point, "score", 0.0) or 0.0)
            point_id = str(getattr(point, "id", ""))

            hits.append(
                SearchHit(
                    point_id=point_id,
                    score=score,
                    payload=payload,
                )
            )

        return hits

    def _build_filter(self, tag_filters: dict[str, list[str]]) -> models.Filter | None:
        must_conditions: list[models.Condition] = []

        for key in ("major_tag", "medium_tag", "minor_tag", "doc_id", "doc_name", "parser_backend", "chunk_type"):
            values = [v for v in tag_filters.get(key, []) if v]
            if not values:
                continue

            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=values),
                )
            )

        page_range = tag_filters.get("page_range", [])
        if len(page_range) == 2:
            must_conditions.append(
                models.FieldCondition(
                    key="page_no",
                    range=models.Range(
                        gte=int(page_range[0]),
                        lte=int(page_range[1]),
                    ),
                )
            )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def _document_filter(self, doc_name: str | None = None, doc_id: str | None = None) -> models.Filter:
        must_conditions: list[models.Condition] = []

        if doc_name:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_name",
                    match=models.MatchValue(value=doc_name),
                )
            )

        if doc_id:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                )
            )

        if not must_conditions:
            raise ValueError("doc_name または doc_id の少なくとも一方が必要です。")

        return models.Filter(must=must_conditions)

    def count_document_points(self, doc_name: str | None = None, doc_id: str | None = None) -> int:
        doc_filter = self._document_filter(doc_name=doc_name, doc_id=doc_id)
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=doc_filter,
            exact=True,
        )
        return int(getattr(result, "count", 0))

    def delete_document(self, doc_name: str | None = None, doc_id: str | None = None) -> int:
        doc_filter = self._document_filter(doc_name=doc_name, doc_id=doc_id)
        count_before = self.count_document_points(doc_name=doc_name, doc_id=doc_id)

        if count_before == 0:
            audit_logger.log(
                "document_delete_skipped",
                {
                    "collection": self.collection_name,
                    "doc_name": doc_name,
                    "doc_id": doc_id,
                    "reason": "no_matching_points",
                },
            )
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=doc_filter),
            wait=True,
        )

        audit_logger.log(
            "document_deleted",
            {
                "collection": self.collection_name,
                "doc_name": doc_name,
                "doc_id": doc_id,
                "deleted_points": count_before,
                "mode": settings.qdrant_mode,
            },
        )
        return count_before

    def mark_favorite(self, point_id: str) -> None:
        current = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
        )
        if not current:
            return

        payload = dict(current[0].payload or {})
        current_weight = int(payload.get("favorite_weight", 0))

        self.client.set_payload(
            collection_name=self.collection_name,
            points=[point_id],
            payload={
                "favorite": True,
                "favorite_weight": current_weight + 1,
            },
        )

        audit_logger.log(
            "favorite_marked",
            {
                "collection": self.collection_name,
                "point_id": point_id,
                "favorite_weight": current_weight + 1,
                "doc_name": payload.get("doc_name"),
                "page_no": payload.get("page_no"),
            },
        )
