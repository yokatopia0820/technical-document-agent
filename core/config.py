from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    qdrant_mode: str = os.getenv("QDRANT_MODE", "server")  # local | server
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_local_path: str = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_local")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "technical_docs")

    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    llm_backend: str = os.getenv("LLM_BACKEND", "none")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    parser_backend: str = os.getenv("PARSER_BACKEND", "pymupdf")
    ocr_enabled: bool = _as_bool("OCR_ENABLED", True)
    ocr_engine: str = os.getenv("OCR_ENGINE", "tesseract")
    ocr_langs: str = os.getenv("OCR_LANGS", "jpn+eng")
    min_chars_for_native_text: int = int(os.getenv("MIN_CHARS_FOR_NATIVE_TEXT", "80"))

    local_pdf_dir: str = os.getenv("LOCAL_PDF_DIR", "./data")
    temp_dir: str = os.getenv("TEMP_DIR", "./tmp")
    audit_log_path: str = os.getenv("AUDIT_LOG_PATH", "./logs/audit.jsonl")
    tag_rules_path: str = os.getenv("TAG_RULES_PATH", "./config/tag_rules.json")
    document_profile_path: str = os.getenv(
        "DOCUMENT_PROFILE_PATH",
        "./config/document_profile_hitachi_202602.json",
    )
    max_hits: int = int(os.getenv("MAX_HITS", "8"))

    def ensure_dirs(self) -> None:
        paths = [
            self.local_pdf_dir,
            self.temp_dir,
            str(Path(self.audit_log_path).parent),
        ]
        if self.qdrant_mode == "local":
            paths.append(self.qdrant_local_path)

        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
