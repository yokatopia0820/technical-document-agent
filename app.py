def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.2rem !important;
            padding-bottom: 1rem !important;
            max-width: 1400px;
        }

        h1, h2, h3 {
            letter-spacing: -0.01em;
            line-height: 1.18 !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        [data-testid="column"] > div {
            padding-top: 0.35rem;
        }

        .app-hero {
            padding: 1.1rem 1.15rem 1rem 1.15rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.30), rgba(15, 23, 42, 0.10));
            margin-top: 0.2rem;
            margin-bottom: 1rem;
            min-height: 158px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .helper-note {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.7;
        }

        .status-card {
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.2rem 0 0.8rem 0;
            background: rgba(15, 23, 42, 0.30);
        }

        .status-card.is-muted {
            opacity: 0.72;
            filter: grayscale(0.18);
        }

        .status-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .status-sub {
            font-size: 0.86rem;
            color: #cbd5e1;
            word-break: break-word;
        }

        .status-icon {
            font-size: 1.15rem;
            margin-right: 0.35rem;
        }

        .settings-note {
            color: #cbd5e1;
            font-size: 0.84rem;
            margin-top: 0.1rem;
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }

        div.stButton > button {
            border-radius: 12px !important;
            min-height: 2.75rem;
            font-weight: 700;
            transition: all 0.15s ease;
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
            border: 1px solid #ff474d !important;
            color: #ffffff !important;
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
            border: 1px solid #ff5a5f !important;
            outline: none !important;
            box-shadow: none !important;
        }

        div.stButton > button:disabled {
            opacity: 0.55;
            cursor: not-allowed;
            transform: none !important;
            border-color: rgba(148, 163, 184, 0.18) !important;
            box-shadow: none !important;
            outline: none !important;
        }

        div.stButton > button:disabled:hover {
            transform: none !important;
            border-color: rgba(148, 163, 184, 0.18) !important;
            box-shadow: none !important;
            outline: none !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )
