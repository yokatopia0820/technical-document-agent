from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import settings
from core.pdf_utils import normalize_text


@lru_cache(maxsize=1)
def load_document_profile() -> dict[str, Any]:
    path = Path(settings.document_profile_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "doc_name_patterns": [],
        "document_title": "",
        "doc_aliases": {},
        "section_page_ranges": {},
        "major_to_page_ranges": {},
        "demo_queries": [],
    }


class DocumentProfile:
    def __init__(self) -> None:
        self.profile = load_document_profile()

    def matches_doc_name(self, doc_name: str | None) -> bool:
        if not doc_name:
            return False
        normalized = normalize_text(doc_name).lower()
        return any(p.lower() in normalized for p in self.profile.get("doc_name_patterns", []))

    def resolve_page_range(self, major_tags: list[str], medium_tags: list[str]) -> list[int]:
        if any(tag == "商品別保証内容・サービス処理区分" for tag in major_tags) or any(tag == "保証" for tag in medium_tags):
            rng = self.profile.get("section_page_ranges", {}).get("商品別保証内容・サービス処理区分")
            if rng:
                return rng
        for tag in major_tags:
            rng = self.profile.get("major_to_page_ranges", {}).get(tag)
            if rng:
                return rng
        return []

    def canonicalize_query(self, text: str) -> str:
        normalized = normalize_text(text)
        lowered = normalized.lower()
        for canonical, aliases in self.profile.get("doc_aliases", {}).items():
            for alias in aliases:
                if alias.lower() in lowered:
                    normalized = normalized.replace(alias, canonical)
        return normalized

    def demo_queries(self) -> list[str]:
        return list(self.profile.get("demo_queries", []))
