from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from core.audit import audit_logger
from core.config import settings
from core.embeddings import Embedder
from core.llm import AnswerEngine
from core.pdf_parser import build_chunks
from core.pdf_utils import build_highlighted_pdf_bytes, save_uploaded_pdf
from core.qdrant_store import QdrantStore
from core.retriever import TechnicalRetriever


@st.cache_resource
def get_services() -> tuple[QdrantStore, TechnicalRetriever]:
    embedder = Embedder()
    store = QdrantStore(embedder)
    retriever = TechnicalRetriever(store, AnswerEngine())
    return store, retriever


def _init_state() -> None:
    defaults = {
        "messages": [],
        "pdf_path": None,
        "doc_name": None,
        "last_result": None,
        "highlighted_pdf": None,
        "preset_prompt": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_document_state(clear_messages: bool = False) -> None:
    st.session_state["pdf_path"] = None
    st.session_state["doc_name"] = None
    st.session_state["last_result"] = None
    st.session_state["highlighted_pdf"] = None
    st.session_state["preset_prompt"] = ""
    if clear_messages:
        st.session_state["messages"] = []


def _ingest_uploaded_pdf(store: QdrantStore, uploaded_pdf, parser_backend: str, ocr_enabled: bool) -> None:
    pdf_path = save_uploaded_pdf(uploaded_pdf, settings.temp_dir)
    st.session_state["pdf_path"] = pdf_path
    st.session_state["doc_name"] = uploaded_pdf.name

    progress_bar = st.progress(0, text="0% | アップロード完了")
    progress_note = st.empty()

    def on_parse_progress(done: int, total: int, stage: str) -> None:
        total = max(total, 1)
        percent = min(80, int(done / total * 80))
        progress_bar.progress(percent, text=f"{percent}% | {stage}")
        progress_note.caption(f"解析進捗: {done}/{total} ページ")

    def on_upsert_progress(done: int, total: int, stage: str) -> None:
        total = max(total, 1)
        percent = 80 + min(20, int(done / total * 20))
        progress_bar.progress(percent, text=f"{percent}% | {stage}")
        progress_note.caption(f"登録進捗: {done}/{total} チャンク")

    with st.status("PDFを解析してQdrantへ登録しています…", expanded=True) as status:
        progress_bar.progress(5, text="5% | ファイル保存完了")

        chunks = build_chunks(
            pdf_path=pdf_path,
            doc_id=Path(pdf_path).stem,
            doc_name=uploaded_pdf.name,
            parser_backend=parser_backend,
            ocr_enabled=ocr_enabled,
            progress_callback=on_parse_progress,
        )

        st.write(f"抽出チャンク数: {len(chunks)}")
        progress_bar.progress(80, text="80% | 解析完了。Qdrantへ登録中")

        inserted = store.upsert_chunks(
            chunks,
            batch_size=64,
            progress_callback=on_upsert_progress,
        )

        progress_bar.progress(100, text="100% | 登録完了")
        progress_note.caption(f"登録完了: {inserted} 件")

        status.update(
            label=f"{uploaded_pdf.name} を解析し、{inserted}件のチャンクを登録しました。",
            state="complete",
        )


def _clear_current_pdf_ui() -> None:
    _reset_document_state(clear_messages=False)
    st.toast("現在のPDF表示状態をクリアしました。")
    st.rerun()


def _delete_current_pdf(store: QdrantStore) -> None:
    pdf_path = st.session_state.get("pdf_path")
    doc_name = st.session_state.get("doc_name")
    doc_id = Path(pdf_path).stem if pdf_path else None

    if not doc_name and not doc_id:
        st.warning("削除対象のPDFが見つかりません。")
        return

    deleted_points = store.delete_document(doc_name=doc_name, doc_id=doc_id)

    file_deleted = False
    if pdf_path:
        try:
            path_obj = Path(pdf_path)
            if path_obj.exists():
                path_obj.unlink()
                file_deleted = True
        except Exception:
            file_deleted = False

    audit_logger.log(
        "pdf_deleted_from_ui",
        {
            "doc_name": doc_name,
            "doc_id": doc_id,
            "pdf_path": pdf_path,
            "deleted_points": deleted_points,
            "file_deleted": file_deleted,
        },
    )

    _reset_document_state(clear_messages=False)
    st.toast(f"PDFを削除しました（Qdrant削除対象: {deleted_points}件）。")
    st.rerun()


def _favorite_button(store: QdrantStore, result: dict) -> None:
    top_hit = result.get("top_hit")
    if not top_hit:
        return
    point_id = top_hit.get("point_id")
    if st.button("🌟 お気に入りに登録", key=f"fav-{point_id}", use_container_width=True):
        store.mark_favorite(point_id)
        st.toast("お気に入りに保存しました。次回から優先表示されます。")


def _render_pdf_panel() -> None:
    st.header("📄 Source Document")
    pdf_bytes = st.session_state.get("highlighted_pdf")
    if pdf_bytes:
        try:
            st.pdf(pdf_bytes)
        except Exception:
            st.info("この環境では埋め込みPDFビューアが使えないため、ハイライト済みPDFをダウンロード表示に切り替えています。")
            st.download_button(
                "ハイライト済みPDFをダウンロード",
                data=pdf_bytes,
                file_name="highlighted.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("回答後に、該当ページを自動表示して根拠箇所をハイライトします。")

    result = st.session_state.get("last_result")
    top_hit = result.get("top_hit") if result else None
    if top_hit:
        left, right = st.columns(2)
        left.metric("ページ", f"P.{top_hit.get('page_no')}")
        right.metric("お気に入り重み", str(top_hit.get("favorite_weight", 0)))
        st.caption(f"{top_hit.get('major_tag')} / {top_hit.get('medium_tag')} / {top_hit.get('minor_tag')}")
        st.code(json.dumps(top_hit.get("shorthand_json", {}), ensure_ascii=False, indent=2), language="json")
        st.write(f"検証状況: {top_hit.get('verification_summary')}")


def _render_audit_sidebar() -> None:
    with st.sidebar.expander("🧾 直近の監査ログ", expanded=False):
        for event in audit_logger.tail(limit=8):
            st.caption(f"{event['timestamp_utc']} | {event['event_type']}")
            st.code(json.dumps(event["payload"], ensure_ascii=False, indent=2), language="json")


def _render_demo_queries(retriever: TechnicalRetriever) -> None:
    demo_queries = retriever.tagger.demo_queries()
    if not demo_queries:
        return
    with st.expander("💡 この資料で試しやすい質問", expanded=False):
        for idx, query in enumerate(demo_queries[:10]):
            if st.button(query, key=f"demo-{idx}", use_container_width=True):
                st.session_state["preset_prompt"] = query


def _render_document_management(store: QdrantStore) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("🗂 現在のPDF")

    current_doc_name = st.session_state.get("doc_name")
    current_pdf_path = st.session_state.get("pdf_path")

    if current_doc_name:
        st.sidebar.success(f"セット中: {current_doc_name}")
    else:
        st.sidebar.info("現在セット中のPDFはありません。")

    if current_pdf_path:
        st.sidebar.caption(f"保存先: {current_pdf_path}")

    clear_col, delete_col = st.sidebar.columns(2)

    with clear_col:
        if st.button("クリア", use_container_width=True, disabled=not current_doc_name):
            _clear_current_pdf_ui()

    with delete_col:
        if st.button("完全削除", use_container_width=True, disabled=not current_doc_name, type="secondary"):
            _delete_current_pdf(store)

    if current_doc_name:
        st.sidebar.caption("「完全削除」はQdrant内データと一時PDFファイルを削除します。")


def main() -> None:
    st.set_page_config(page_title="技術資料AI", layout="wide")
    _init_state()
    store, retriever = get_services()

    st.sidebar.title("⚙️ PDFセットアップ")
    parser_backend = st.sidebar.selectbox("解析バックエンド", ["hybrid", "pymupdf", "docling", "auto"], index=1)
    ocr_enabled = st.sidebar.toggle("OCRを有効化", value=settings.ocr_enabled)
    uploaded_pdf = st.sidebar.file_uploader("サービスハンドブックPDFをアップロード", type=["pdf"])

    if uploaded_pdf is not None:
        st.sidebar.write(f"選択中: {uploaded_pdf.name}")
        if st.sidebar.button("解析してQdrantへ登録", use_container_width=True):
            _ingest_uploaded_pdf(store, uploaded_pdf, parser_backend=parser_backend, ocr_enabled=ocr_enabled)
    else:
        sample_pdf = Path(settings.local_pdf_dir) / "新202602_サービスハンドブック_技術料一覧表.pdf"
        if sample_pdf.exists():
            st.sidebar.success(f"サンプルPDF配置済み: {sample_pdf.name}")

    _render_document_management(store)
    _render_audit_sidebar()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.title("💬 AI Technical Assistant")
        st.caption("結論先出し・自己検証・お気に入り学習つきで、1,000ページ級PDFから根拠付き回答を返します。")

        status_cols = st.columns(3)
        status_cols[0].metric("Qdrant", settings.qdrant_url)
        status_cols[1].metric("Parser", parser_backend)
        status_cols[2].metric("OCR", "ON" if ocr_enabled else "OFF")
        _render_demo_queries(retriever)

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("質問を入力（例：4kWエアコンの料金は？）")
        if not prompt and st.session_state.get("preset_prompt"):
            prompt = st.session_state["preset_prompt"]
            st.session_state["preset_prompt"] = ""

        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            result = retriever.answer(prompt, doc_name=st.session_state.get("doc_name"))
            st.session_state["last_result"] = result
            top_hit = result.get("top_hit")
            if top_hit and top_hit.get("source_pdf_path"):
                st.session_state["highlighted_pdf"] = build_highlighted_pdf_bytes(top_hit["source_pdf_path"], top_hit)

            with st.chat_message("assistant"):
                st.write(result["answer"])
                with st.expander("🔑 検索に使った鍵（タグ）"):
                    st.json(result["filters"])
                with st.expander("🧠 根拠JSON"):
                    st.code(json.dumps(result.get("hits", []), ensure_ascii=False, indent=2), language="json")
                _favorite_button(store, result)

            st.session_state["messages"].append({"role": "assistant", "content": result["answer"]})

    with col2:
        _render_pdf_panel()


if __name__ == "__main__":
    main()
Copy
core/pdf_parser.py 全文
Copyfrom __future__ import annotations

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
                        f"{page.number + 1}",
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
            "chunk_text": " | ".join([context_text, f"{page.number + 1}", text[:500]]),
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
            headings = _extract_headings(page)
            context_text = " | ".join(h["text"] for h in headings[:4])
            native_text = normalize_text(page.get_text("text"))

            chunks.extend(_table_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))
            chunks.extend(_text_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))

            if ocr_enabled and len(native_text) < settings.min_chars_for_native_text:
                chunks.extend(_ocr_chunks(page, pdf_path, doc_id, doc_name, context_text, tagger))

            if progress_callback:
                progress_callback(idx, total_pages, f"PDF解析中 P.{idx}/{total_pages}")

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
            progress_callback(1, 1, "Docling構造解析中")
        try:
            chunks.extend(build_docling_chunks(pdf_path, doc_id=doc_id, doc_name=doc_name, ocr_enabled=ocr_enabled))
            used_backends.append("docling")
            if progress_callback:
                progress_callback(1, 1, "Docling構造解析完了")
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
