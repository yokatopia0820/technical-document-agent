from __future__ import annotations

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
            padding-top: 1.6rem !important;
            padding-bottom: 1rem !important;
            max-width: 1400px;
        }

        h1, h2, h3 {
            letter-spacing: -0.01em;
            line-height: 1.2 !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        .app-hero {
            padding: 1.15rem 1.2rem 1.05rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.28), rgba(15, 23, 42, 0.12));
            margin-bottom: 1rem;
        }

        .app-hero-title {
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
        }

        .helper-note {
            color: #cbd5e1;
            font-size: 0.94rem;
            line-height: 1.7;
        }

        .status-card {
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.15rem 0 0.8rem 0;
            background: rgba(15, 23, 42, 0.28);
        }

        .status-card.is-muted {
            opacity: 0.72;
            filter: grayscale(0.18);
        }

        .status-title {
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 0.22rem;
        }

        .status-sub {
            font-size: 0.88rem;
            color: #cbd5e1;
            line-height: 1.6;
            word-break: break-word;
        }

        .settings-note {
            color: #cbd5e1;
            font-size: 0.84rem;
            line-height: 1.65;
            margin: 0.2rem 0 0.55rem 0;
        }

        div.stButton > button {
            border-radius: 12px !important;
            min-height: 2.8rem !important;
            font-weight: 800 !important;
            transition: all 0.15s ease !important;
            outline: none !important;
            box-shadow: none !important;
        }

        div.stButton > button:hover:not(:disabled) {
            transform: translateY(-1px);
            outline: none !important;
            box-shadow: none !important;
        }

        div.stButton > button:focus,
        div.stButton > button:focus-visible,
        div.stButton > button:active {
            outline: none !important;
            box-shadow: none !important;
        }

        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="baseButton-primary"] {
            background: #ff5a5f !important;
            color: #ffffff !important;
            border: 1px solid #ff5a5f !important;
            outline: none !important;
            box-shadow: none !important;
        }

        div.stButton > button[kind="primary"]:hover:not(:disabled),
        div.stButton > button[data-testid="baseButton-primary"]:hover:not(:disabled) {
            background: #ff474d !important;
            color: #ffffff !important;
            border: 1px solid #ff474d !important;
            outline: none !important;
            box-shadow: none !important;
        }

        div.stButton > button[kind="primary"]:focus,
        div.stButton > button[kind="primary"]:focus-visible,
        div.stButton > button[kind="primary"]:active,
        div.stButton > button[data-testid="baseButton-primary"]:focus,
        div.stButton > button[data-testid="baseButton-primary"]:focus-visible,
        div.stButton > button[data-testid="baseButton-primary"]:active {
            background: #ff5a5f !important;
            color: #ffffff !important;
            border: 1px solid #ff5a5f !important;
            outline: none !important;
            box-shadow: none !important;
        }

        div.stButton > button:disabled {
            opacity: 0.58 !important;
            cursor: not-allowed !important;
            transform: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        section[data-testid="stFileUploader"] small {
            display: none !important;
        }

        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            position: relative;
        }

        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
            min-height: 68px;
        }

        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] div,
        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span,
        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small,
        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] p {
            display: none !important;
        }

        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]::before {
            content: "PDFファイルをここに置くか、下のボタンで選択してください";
            display: block;
            text-align: center;
            font-weight: 800;
            font-size: 0.98rem;
            color: #e5e7eb;
            margin-bottom: 0.3rem;
            line-height: 1.6;
        }

        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content: "1ファイル / 最大1GB / PDF";
            display: block;
            text-align: center;
            font-size: 0.84rem;
            color: #94a3b8;
        }

        section[data-testid="stFileUploader"] button {
            position: relative;
            font-size: 0 !important;
            color: transparent !important;
        }

        section[data-testid="stFileUploader"] button * {
            display: none !important;
        }

        section[data-testid="stFileUploader"] button::after {
            content: "ファイルを選択";
            display: inline-block;
            font-size: 0.95rem;
            font-weight: 800;
            color: #ffffff;
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
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "current_doc": None,
        "last_ingested_signature": None,
        "last_answer_result": None,
        "is_ingesting": False,
        "uploader_key": 0,
        "settings_parser_backend": settings.parser_backend,
        "settings_ocr_enabled": settings.ocr_enabled,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _bump_uploader_key() -> None:
    st.session_state.uploader_key += 1


def _uploaded_signature(uploaded_file) -> tuple[str, int] | None:
    if uploaded_file is None:
        return None
    size = int(getattr(uploaded_file, "size", len(uploaded_file.getbuffer())))
    return (uploaded_file.name, size)


def _render_status_card(title: str, body: str, muted: bool = False) -> None:
    klass = "status-card is-muted" if muted else "status-card"
    st.markdown(
        f"""
        <div class="{klass}">
            <div class="status-title">{title}</div>
            <div class="status-sub">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_get_services() -> tuple[QdrantStore, TechnicalRetriever] | None:
    try:
        return get_services()
    except Exception as exc:
        st.error("Qdrant の初期化に失敗しました。")
        st.code(str(exc))
        st.info(
            "Streamlit Cloud では QDRANT_MODE=server に切り替え、"
            "QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION を Secrets に設定してください。"
        )
        audit_logger.log("service_init_failed", {"error": str(exc)})
        return None


def _render_header() -> None:
    gear_col, hero_col = st.columns([1, 9], vertical_alignment="top")

    with gear_col:
        with st.popover("⚙️"):
            st.markdown("**詳細設定**")
            st.caption("初心者向けではない設定は、ここを開いたときだけ表示します。")
            st.selectbox(
                "解析方式",
                options=["pymupdf", "docling", "hybrid"],
                key="settings_parser_backend",
            )
            st.checkbox(
                "OCR を使う",
                key="settings_ocr_enabled",
            )
            st.markdown(
                f"""
                <div class="settings-note">
                現在の Qdrant モード: <b>{settings.qdrant_mode}</b><br>
                コレクション: <b>{settings.qdrant_collection}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with hero_col:
        st.markdown(
            """
            <div class="app-hero">
                <div class="app-hero-title">PDFに質問</div>
                <div class="helper-note">
                    先に PDF を取り込み、その後に質問してください。<br>
                    詳細設定は歯車アイコンを押したときだけ表示されます。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _delete_current_document() -> None:
    current_doc = st.session_state.current_doc
    if not current_doc:
        return

    services = _safe_get_services()
    if not services:
        return

    store, _ = services

    try:
        store.delete_document(
            doc_name=current_doc.get("doc_name"),
            doc_id=current_doc.get("doc_id"),
        )
    except Exception as exc:
        st.error("取り込み済みデータの削除に失敗しました。")
        st.code(str(exc))
        audit_logger.log("document_delete_failed", {"error": str(exc), "doc": current_doc})
        return

    st.session_state.current_doc = None
    st.session_state.last_ingested_signature = None
    st.session_state.last_answer_result = None
    st.session_state.messages = []
    _bump_uploader_key()
    st.rerun()


def _ingest_uploaded_pdf(uploaded_file) -> None:
    if uploaded_file is None:
        return

    services = _safe_get_services()
    if not services:
        return

    store, _ = services
    signature = _uploaded_signature(uploaded_file)

    st.session_state.is_ingesting = True
    progress_box = st.container()
    progress_bar = progress_box.progress(0, text="進捗 0%")
    stage_placeholder = progress_box.empty()

    def set_progress(value: int, label: str) -> None:
        value = max(0, min(100, int(value)))
        progress_bar.progress(value, text=f"進捗 {value}%")
        stage_placeholder.caption(label)

    try:
        pdf_path = save_uploaded_pdf(uploaded_file, settings.temp_dir)
        file_size = int(getattr(uploaded_file, "size", len(uploaded_file.getbuffer())))
        doc_name = uploaded_file.name
        doc_id = f"{Path(uploaded_file.name).stem}-{file_size}"

        try:
            existing_count = store.count_document_points(doc_name=doc_name)
            if existing_count > 0:
                store.delete_document(doc_name=doc_name)
        except Exception:
            pass

        def parse_progress(current: int, total: int, label: str) -> None:
            ratio = (current / total) if total else 0.0
            set_progress(max(1, int(ratio * 70)), f"処理段階: {label}")

        chunks = build_chunks(
            pdf_path=pdf_path,
            doc_id=doc_id,
            doc_name=doc_name,
            parser_backend=st.session_state.settings_parser_backend,
            ocr_enabled=bool(st.session_state.settings_ocr_enabled),
            progress_callback=parse_progress,
        )

        if not chunks:
            set_progress(100, "処理完了")
            st.error("PDF から取り込める情報が見つかりませんでした。")
            return

        def upsert_progress(current: int, total: int, label: str) -> None:
            ratio = (current / total) if total else 0.0
            set_progress(70 + int(ratio * 30), f"処理段階: {label}")

        inserted = store.upsert_chunks(
            chunks,
            progress_callback=upsert_progress,
        )

        set_progress(100, "処理完了")

        st.session_state.current_doc = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "source_pdf_path": pdf_path,
            "chunk_count": len(chunks),
            "point_count": inserted,
        }
        st.session_state.last_ingested_signature = signature
        st.session_state.last_answer_result = None
        st.session_state.messages = []

        st.success(f"「{doc_name}」の取り込みが完了しました。")
        audit_logger.log(
            "document_ingested_via_ui",
            {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_count": len(chunks),
                "point_count": inserted,
                "parser_backend": st.session_state.settings_parser_backend,
                "ocr_enabled": bool(st.session_state.settings_ocr_enabled),
            },
        )

    except Exception as exc:
        st.error("PDF の取り込みに失敗しました。")
        st.code(str(exc))
        audit_logger.log("document_ingest_failed", {"error": str(exc), "file_name": uploaded_file.name})
    finally:
        st.session_state.is_ingesting = False


def _render_upload_panel() -> None:
    current_doc = st.session_state.current_doc
    if current_doc:
        _render_status_card(
            "取り込み済みPDF",
            f"{current_doc['doc_name']}<br>登録件数: {current_doc['point_count']} 件",
        )
    else:
        _render_status_card(
            "取り込み済みPDF",
            "まだありません。先に PDF を選んでください。",
            muted=True,
        )

    uploaded_file = st.file_uploader(
        "PDFアップロード",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    signature = _uploaded_signature(uploaded_file)
    already_ingested = (
        uploaded_file is not None
        and st.session_state.last_ingested_signature is not None
        and signature == st.session_state.last_ingested_signature
        and st.session_state.current_doc is not None
    )

    if already_ingested:
        st.info("この PDF はすでに取り込み済みです。別ファイルを選ぶか、削除してから再実行してください。")

    start_disabled = (
        uploaded_file is None
        or st.session_state.is_ingesting
        or already_ingested
    )

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button(
            "読み込み開始",
            type="primary",
            use_container_width=True,
            disabled=start_disabled,
        )
    with col2:
        delete_clicked = st.button(
            "取り込み削除",
            use_container_width=True,
            disabled=st.session_state.is_ingesting or st.session_state.current_doc is None,
        )

    if delete_clicked:
        _delete_current_document()

    if start_clicked and uploaded_file is not None:
        _ingest_uploaded_pdf(uploaded_file)


def _render_hit(hit: dict[str, Any], index: int, key_prefix: str) -> None:
    page_no = hit.get("page_no", "-")
    verified = "検証済み" if hit.get("verified") else "要確認"
    label_parts = [v for v in [hit.get("row_label"), hit.get("col_label")] if v]
    tag_parts = [v for v in [hit.get("major_tag"), hit.get("medium_tag"), hit.get("minor_tag")] if v]

    with st.expander(f"根拠 {index + 1} | P.{page_no} | {verified}", expanded=(index == 0)):
        st.markdown(f"**値**: {hit.get('value', '-')}")
        if label_parts:
            st.markdown(f"**行・列**: {' / '.join(label_parts)}")
        if tag_parts:
            st.markdown(f"**タグ**: {' > '.join(tag_parts)}")
        if hit.get("verification_summary"):
            st.caption(f"照合: {hit.get('verification_summary')}")

        pdf_path = hit.get("source_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            try:
                highlighted_pdf = build_highlighted_pdf_bytes(pdf_path, hit)
                st.download_button(
                    "該当箇所を強調した PDF をダウンロード",
                    data=highlighted_pdf,
                    file_name=f"highlight_p{page_no}_{index + 1}.pdf",
                    mime="application/pdf",
                    key=f"{key_prefix}_download_{index}",
                    use_container_width=True,
                )
            except Exception:
                pass


def _run_query(query: str) -> None:
    current_doc = st.session_state.current_doc
    if not current_doc:
        return

    services = _safe_get_services()
    if not services:
        return

    _, retriever = services

    try:
        result = retriever.answer(query, doc_name=current_doc["doc_name"])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "result": result,
            }
        )
        st.session_state.last_answer_result = result
        audit_logger.log(
            "query_executed_via_ui",
            {
                "doc_name": current_doc["doc_name"],
                "query": query,
                "hit_count": len(result.get("hits", [])),
            },
        )
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "回答の生成に失敗しました。",
                "result": None,
            }
        )
        st.error("回答の生成に失敗しました。")
        st.code(str(exc))
        audit_logger.log("query_failed", {"error": str(exc), "query": query})


def _render_chat_panel() -> None:
    current_doc = st.session_state.current_doc

    if not current_doc:
        _render_status_card(
            "質問エリア",
            "先に左側で PDF を取り込んでください。",
            muted=True,
        )
        st.chat_input("先に PDF を取り込んでください", disabled=True)
        return

    _render_status_card(
        "質問対象",
        f"{current_doc['doc_name']}<br>質問すると、根拠ページつきで回答します。",
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-title">質問例</div>
                <div class="status-sub">
                    定格電圧は？<br>
                    安全上の注意は？<br>
                    型式は？<br>
                    仕様表の該当ページを教えて
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            result = message.get("result")
            if message["role"] == "assistant" and result and result.get("hits"):
                for hit_index, hit in enumerate(result["hits"][:4]):
                    _render_hit(hit, hit_index, key_prefix=f"msg_{idx}")

    prompt = st.chat_input(
        "PDFについて質問してください",
        disabled=st.session_state.is_ingesting,
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        _run_query(prompt)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="PDFに質問", page_icon="📄", layout="wide")
    _inject_ui_styles()
    _init_state()

    if settings.qdrant_mode == "local":
        st.warning(
            "現在は Qdrant が local モードです。Streamlit Cloud では server モード推奨です。"
        )

    left_col, right_col = st.columns([1.0, 1.35], gap="large")

    with left_col:
        _render_header()
        _render_upload_panel()

    with right_col:
        _render_chat_panel()


if __name__ == "__main__":
    main()
