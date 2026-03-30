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
