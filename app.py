from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import streamlit as st

from core.audit import audit_logger
from core.config import settings
from core.embeddings import Embedder
from core.llm import AnswerEngine
from core.pdf_parser import build_chunks
from core.pdf_utils import build_highlighted_pdf_bytes, save_uploaded_pdf
from core.qdrant_store import QdrantStore
from core.retriever import TechnicalRetriever


class UploadedPDFBuffer(io.BytesIO):
    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name
        self.type = "application/pdf"

    def getbuffer(self):
        return memoryview(self.getvalue())


@st.cache_resource
def get_services() -> tuple[QdrantStore, TechnicalRetriever]:
    embedder = Embedder()
    store = QdrantStore(embedder)
    retriever = TechnicalRetriever(store, AnswerEngine())
    return store, retriever


def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.01em;
        }

        .app-hero {
            padding: 1.05rem 1.15rem 1rem 1.15rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.34), rgba(15, 23, 42, 0.12));
            margin-bottom: 0.9rem;
        }

        .status-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.2rem 0 0.8rem 0;
            background: rgba(15, 23, 42, 0.34);
        }

        .status-card.is-muted {
            opacity: 0.72;
            filter: grayscale(0.15);
        }

        .status-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .status-sub {
            font-size: 0.85rem;
            color: #cbd5e1;
            word-break: break-word;
        }

        .status-icon {
            font-size: 1.15rem;
            margin-right: 0.35rem;
        }

        div.stButton > button {
            border-radius: 12px;
            min-height: 2.75rem;
            font-weight: 700;
            transition: all 0.15s ease;
        }

        div.stButton > button:hover:not(:disabled) {
            transform: translateY(-1px);
            border-color: rgba(96, 165, 250, 0.9);
        }

        div.stButton > button:disabled {
            opacity: 0.55;
            cursor: not-allowed;
            transform: none !important;
            border-color: rgba(148, 163, 184, 0.18) !important;
            box-shadow: none !important;
        }

        div.stButton > button:disabled:hover {
            transform: none !important;
            border-color: rgba(148, 163, 184, 0.18) !important;
        }

        section[data-testid="stFileUploader"] small {
            display: none !important;
        }

        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] > div {
            display: none !important;
        }

        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content: "PDFファイルをアップロードしてください";
            display: block;
            text-align: center;
            font-weight: 700;
            font-size: 1rem;
            color: #e5e7eb;
            margin-bottom: 0.25rem;
        }

        section[data-testid="stFileUploader"] button {
            font-size: 0 !important;
        }

        section[data-testid="stFileUploader"] button::after {
            content: "ファイルを開く";
            font-size: 0.95rem;
            font-weight: 700;
        }

        div[data-testid="stChatInput"] textarea::placeholder {
            color: #94a3b8;
        }

        .evidence-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.65rem;
            background: rgba(15, 23, 42, 0.24);
        }

        .evidence-title {
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .helper-note {
            color: #cbd5e1;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults = {
        "messages": [],
        "pdf_path": None,
        "doc_name": None,
        "last_result": None,
        "highlighted_pdf": None,
        "preset_prompt": "",
        "uploaded_pdf_bytes": None,
        "uploaded_pdf_name": None,
        "is_processing": False,
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _bump_uploader_key() -> None:
    st.session_state["uploader_key"] = int(st.session_state.get("uploader_key", 0)) + 1


def _clear_staged_upload() -> None:
    st.session_state["uploaded_pdf_bytes"] = None
    st.session_state["uploaded_pdf_name"] = None
    _bump_uploader_key()


def _reset_document_state(clear_messages: bool = True) -> None:
    st.session_state["pdf_path"] = None
    st.session_state["doc_name"] = None
    st.session_state["last_result"] = None
    st.session_state["highlighted_pdf"] = None
    st.session_state["preset_prompt"] = ""
    _clear_staged_upload()
    if clear_messages:
        st.session_state["messages"] = []


def _store_uploaded_pdf_selection(uploaded_pdf) -> None:
    st.session_state["uploaded_pdf_bytes"] = uploaded_pdf.getvalue()
    st.session_state["uploaded_pdf_name"] = uploaded_pdf.name


def _build_staged_pdf_file() -> UploadedPDFBuffer | None:
    data = st.session_state.get("uploaded_pdf_bytes")
    name = st.session_state.get("uploaded_pdf_name")
    if not data or not name:
        return None
    return UploadedPDFBuffer(data=data, name=name)


def _render_status_card(title: str, subtext: str = "", icon: str = "📄", muted: bool = False) -> None:
    css_class = "status-card is-muted" if muted else "status-card"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="status-title"><span class="status-icon">{icon}</span>{title}</div>
            <div class="status-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_current_pdf_ui() -> None:
    _reset_document_state(clear_messages=True)
    st.toast("現在のPDFをクリアしました。")
    st.rerun()


def _delete_current_pdf(store: QdrantStore) -> None:
    pdf_path = st.session_state.get("pdf_path")
    doc_name = st.session_state.get("doc_name")
    doc_id = Path(pdf_path).stem if pdf_path else None

    if not doc_name and not doc_id:
        st.warning("削除するPDFが見つかりません。")
        return

    deleted_points = 0
    try:
        deleted_points = store.delete_document(doc_name=doc_name, doc_id=doc_id)
    except Exception as exc:
        audit_logger.log(
            "document_delete_failed",
            {
                "doc_name": doc_name,
                "doc_id": doc_id,
                "error": str(exc),
            },
        )

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

    _reset_document_state(clear_messages=True)
    st.toast(f"PDFを削除しました（削除対象: {deleted_points}件）。")
    st.rerun()


def _favorite_button(store: QdrantStore, result: dict[str, Any]) -> None:
    top_hit = result.get("top_hit")
    if not top_hit:
        return

    point_id = top_hit.get("point_id")
    if not point_id:
        return

    if st.button("この回答を優先表示する", key=f"fav-{point_id}", use_container_width=True):
        try:
            store.mark_favorite(point_id)
            st.toast("次回からこの回答を優先しやすくしました。")
        except Exception as exc:
            audit_logger.log("favorite_mark_failed", {"point_id": point_id, "error": str(exc)})
            st.warning("優先表示の保存に失敗しました。")


def _render_answer_evidence(result: dict[str, Any]) -> None:
    hits = result.get("hits", []) or []
    if not hits:
        return

    with st.expander("回答の根拠を見る", expanded=False):
        for idx, hit in enumerate(hits[:4], start=1):
            page_no = hit.get("page_no", "-")
            verified = "確認済み" if hit.get("verified") else "要確認"

            row_label = str(hit.get("row_label") or "").strip()
            col_label = str(hit.get("col_label") or "").strip()
            value = str(hit.get("value") or "").strip()

            if row_label and col_label and value:
                summary = f"{row_label} / {col_label} / {value}"
            elif value:
                summary = value
            else:
                summary = "該当箇所の要約を表示できませんでした。"

            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-title">{idx}. ページ {page_no} ・ {verified}</div>
                    <div>{summary}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_pdf_panel() -> None:
    st.subheader("現在のPDF")

    current_doc_name = st.session_state.get("doc_name")
    pdf_path = st.session_state.get("pdf_path")
    highlighted_pdf = st.session_state.get("highlighted_pdf")

    if current_doc_name:
        st.caption(current_doc_name)
    else:
        st.info("まだPDFが読み込まれていません。左側からPDFをアップロードしてください。")
        return

    pdf_bytes: bytes | None = None

    if highlighted_pdf:
        pdf_bytes = highlighted_pdf
        st.caption("AIが参照した箇所がある場合は、ハイライト表示します。")
    elif pdf_path and Path(pdf_path).exists():
        try:
            pdf_bytes = Path(pdf_path).read_bytes()
        except Exception:
            pdf_bytes = None

    if pdf_bytes:
        try:
            st.pdf(pdf_bytes)
        except Exception:
            st.download_button(
                "PDFをダウンロード",
                data=pdf_bytes,
                file_name=current_doc_name,
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.warning("PDF表示の準備に失敗しました。")

    result = st.session_state.get("last_result")
    top_hit = result.get("top_hit") if result else None
    if top_hit:
        st.divider()
        st.markdown("**参照した箇所**")
        st.write(f"ページ: P.{top_hit.get('page_no', '-')}")
        if top_hit.get("value"):
            st.write(f"内容: {top_hit.get('value')}")


def _render_demo_queries(retriever: TechnicalRetriever, disabled: bool) -> None:
    demo_queries = retriever.tagger.demo_queries()
    if not demo_queries:
        return

    st.markdown("**質問例**")
    cols = st.columns(2)
    for idx, query in enumerate(demo_queries[:6]):
        with cols[idx % 2]:
            if st.button(query, key=f"demo-{idx}", use_container_width=True, disabled=disabled):
                st.session_state["preset_prompt"] = query
                st.rerun()


def _ingest_uploaded_pdf(store: QdrantStore, uploaded_pdf, parser_backend: str, ocr_enabled: bool) -> None:
    st.session_state["is_processing"] = True
    st.session_state["last_result"] = None
    st.session_state["highlighted_pdf"] = None

    progress_bar = st.progress(0, text="0% | 準備中")
    progress_note = st.empty()

    try:
        pdf_path = save_uploaded_pdf(uploaded_pdf, settings.temp_dir)
        st.session_state["pdf_path"] = pdf_path
        st.session_state["doc_name"] = uploaded_pdf.name

        def on_parse_progress(done: int, total: int, stage: str) -> None:
            total = max(total, 1)
            percent = min(80, int(done / total * 80))
            progress_bar.progress(percent, text=f"{percent}% | {stage}")
            progress_note.caption(f"解析済み: {done}/{total} ページ")

        def on_upsert_progress(done: int, total: int, stage: str) -> None:
            total = max(total, 1)
            percent = 80 + min(20, int(done / total * 20))
            progress_bar.progress(percent, text=f"{percent}% | {stage}")
            progress_note.caption(f"登録済み: {done}/{total} チャンク")

        with st.status("PDFを読み込んでいます…", expanded=True) as status:
            progress_bar.progress(5, text="5% | ファイル保存完了")

            chunks = build_chunks(
                pdf_path=pdf_path,
                doc_id=Path(pdf_path).stem,
                doc_name=uploaded_pdf.name,
                parser_backend=parser_backend,
                ocr_enabled=ocr_enabled,
                progress_callback=on_parse_progress,
            )

            progress_bar.progress(80, text="80% | 解析完了・登録中")
            st.write(f"抽出件数: {len(chunks)}")

            inserted = store.upsert_chunks(
                chunks,
                batch_size=64,
                progress_callback=on_upsert_progress,
            )

            progress_bar.progress(100, text="100% | 読込完了")
            progress_note.caption(f"登録完了: {inserted} 件")

            status.update(
                label=f"{uploaded_pdf.name} の読込が完了しました。",
                state="complete",
            )

        _clear_staged_upload()

    except Exception as exc:
        audit_logger.log(
            "pdf_ingest_failed",
            {
                "doc_name": getattr(uploaded_pdf, "name", None),
                "error": str(exc),
            },
        )
        st.error("PDFの読込中にエラーが発生しました。ログを確認してください。")
        raise

    finally:
        st.session_state["is_processing"] = False


def _render_sidebar(store: QdrantStore) -> tuple[str, bool]:
    staged_name = st.session_state.get("uploaded_pdf_name")
    current_doc_name = st.session_state.get("doc_name")
    is_processing = bool(st.session_state.get("is_processing"))

    parser_backend = settings.parser_backend
    ocr_enabled = settings.ocr_enabled

    st.sidebar.subheader("PDFを準備")
    st.sidebar.caption("PDFを選んでから『読込開始』を押してください。")

    with st.sidebar.expander("詳細設定（上級者向け）", expanded=False):
        parser_backend = st.selectbox(
            "読取方式",
            ["auto", "pymupdf", "hybrid", "docling"],
            index=["auto", "pymupdf", "hybrid", "docling"].index("pymupdf" if settings.parser_backend == "pymupdf" else "auto"),
            help="通常は変更不要です。",
        )
        ocr_enabled = st.toggle(
            "画像内の文字も読む",
            value=settings.ocr_enabled,
            help="スキャンPDFや画像文字が含まれる場合に有効です。通常はONのままで問題ありません。",
        )

    st.sidebar.markdown("**1. アップロード**")

    if not staged_name and not current_doc_name and not is_processing:
        uploaded_pdf = st.sidebar.file_uploader(
            "PDFファイルをアップロードしてください",
            type=["pdf"],
            key=f"pdf_uploader_{st.session_state.get('uploader_key', 0)}",
            label_visibility="collapsed",
        )
        if uploaded_pdf is not None:
            _store_uploaded_pdf_selection(uploaded_pdf)
            st.rerun()
    else:
        card_name = staged_name or current_doc_name or "PDFファイル"
        card_label = "アップロードOK" if not is_processing else "アップロード済み"
        _render_status_card(card_label, card_name, icon="📎", muted=True)

    st.sidebar.markdown("**2. 読込**")

    can_start = bool(staged_name) and not is_processing and not current_doc_name
    start_help = None
    if current_doc_name:
        start_help = "すでに読み込み済みです。別のPDFに差し替える場合は下のボタンを使ってください。"
    elif not staged_name:
        start_help = "先にPDFをアップロードしてください。"

    if st.sidebar.button("読込開始", use_container_width=True, disabled=not can_start, help=start_help, type="primary"):
        staged_file = _build_staged_pdf_file()
        if staged_file is not None:
            _ingest_uploaded_pdf(
                store=store,
                uploaded_pdf=staged_file,
                parser_backend=parser_backend,
                ocr_enabled=ocr_enabled,
            )
            st.rerun()

    if is_processing:
        _render_status_card("読込中", "この間は他の操作ができません。完了までそのままお待ちください。", icon="⏳")

    st.sidebar.divider()
    st.sidebar.markdown("**現在のPDF**")

    if current_doc_name:
        _render_status_card("アップロードOK", current_doc_name, icon="📄", muted=True)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("差し替える", use_container_width=True, disabled=is_processing):
                _clear_current_pdf_ui()
        with col2:
            if st.button("削除する", use_container_width=True, disabled=is_processing):
                _delete_current_pdf(store)
        st.sidebar.caption("データも消したい場合は『削除する』を使ってください。")
    elif staged_name and not is_processing:
        _render_status_card("アップロードOK", staged_name, icon="📎", muted=True)
        if st.sidebar.button("選び直す", use_container_width=True):
            _clear_staged_upload()
            st.rerun()
    else:
        st.sidebar.info("まだPDFはセットされていません。")

    return parser_backend, ocr_enabled


def main() -> None:
    st.set_page_config(page_title="PDFに質問", layout="wide")
    _inject_ui_styles()
    _init_state()

    store, retriever = get_services()
    _render_sidebar(store)

    col1, col2 = st.columns([1.02, 0.98])

    with col1:
        st.markdown(
            """
            <div class="app-hero">
                <h1 style="margin:0 0 0.35rem 0;">PDFに質問</h1>
                <div class="helper-note">
                    PDFを読み込むと、料金・保証・型番などを質問できます。<br>
                    回答には、参照したページや該当箇所もあわせて表示します。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_doc_name = st.session_state.get("doc_name")
        is_processing = bool(st.session_state.get("is_processing"))

        if current_doc_name:
            st.caption(f"現在のPDF: {current_doc_name}")
        else:
            st.info("左側でPDFをアップロードして『読込開始』を押してください。")

        _render_demo_queries(retriever, disabled=(not current_doc_name or is_processing))

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        chat_disabled = not current_doc_name or is_processing
        chat_placeholder = "PDFについて質問してください"
        if is_processing:
            chat_placeholder = "読込中です。完了までお待ちください。"
        elif not current_doc_name:
            chat_placeholder = "先にPDFを読み込んでください"

        prompt = st.chat_input(chat_placeholder, disabled=chat_disabled)

        if not prompt and st.session_state.get("preset_prompt") and not chat_disabled:
            prompt = st.session_state["preset_prompt"]
            st.session_state["preset_prompt"] = ""

        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            try:
                result = retriever.answer(prompt, doc_name=st.session_state.get("doc_name"))
                st.session_state["last_result"] = result

                top_hit = result.get("top_hit")
                if top_hit and top_hit.get("source_pdf_path"):
                    try:
                        st.session_state["highlighted_pdf"] = build_highlighted_pdf_bytes(
                            top_hit["source_pdf_path"],
                            top_hit,
                        )
                    except Exception as exc:
                        audit_logger.log("highlight_pdf_failed", {"error": str(exc)})

                with st.chat_message("assistant"):
                    st.write(result.get("answer", "回答を生成できませんでした。"))
                    _render_answer_evidence(result)
                    _favorite_button(store, result)

                st.session_state["messages"].append(
                    {"role": "assistant", "content": result.get("answer", "")}
                )

            except Exception as exc:
                audit_logger.log(
                    "chat_query_failed",
                    {
                        "query": prompt,
                        "doc_name": st.session_state.get("doc_name"),
                        "error": str(exc),
                    },
                )
                with st.chat_message("assistant"):
                    st.error("検索中にエラーが発生しました。管理画面のログを確認してください。")

    with col2:
        _render_pdf_panel()


if __name__ == "__main__":
    main()
