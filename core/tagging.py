from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import settings
from core.document_profile import DocumentProfile
from core.pdf_utils import normalize_text


@lru_cache(maxsize=1)
def load_tag_rules() -> dict[str, Any]:
    path = Path(settings.tag_rules_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"major_tags": {}, "medium_tags": {}, "minor_patterns": [], "query_aliases": {}}


class TaggingEngine:
    def __init__(self) -> None:
        self.rules = load_tag_rules()
        self.profile = DocumentProfile()

    def derive_tags(self, *parts: str) -> tuple[str, str, str]:
        text = normalize_text(" ".join(parts)).lower()
        major = self._best_tag(text, self.rules.get("major_tags", {}), default="技術資料")
        medium = self._best_tag(text, self.rules.get("medium_tags", {}), default="本文")
        minor = self._minor_tag(text)
        return major, medium, minor

    def infer_filters(self, query: str, doc_name: str | None = None) -> dict[str, list[str]]:
        normalized = self.profile.canonicalize_query(query)
        for alias, canonical in self.rules.get("query_aliases", {}).items():
            normalized = normalized.replace(alias, canonical)
        text = normalized.lower()

        filters: dict[str, list[str]] = {"major_tag": [], "medium_tag": [], "minor_tag": []}
        if doc_name:
            filters["doc_name"] = [doc_name]

        for tag_name, spec in self.rules.get("major_tags", {}).items():
            if self._matches(text, spec):
                filters["major_tag"].append(tag_name)
        for tag_name, spec in self.rules.get("medium_tags", {}).items():
            if self._matches(text, spec):
                filters["medium_tag"].append(tag_name)

        minor_hits: list[str] = []
        for pattern in self.rules.get("minor_patterns", []):
            minor_hits.extend(re.findall(pattern, normalized, flags=re.IGNORECASE))
        filters["minor_tag"] = list(dict.fromkeys(normalize_text(v) for v in minor_hits if normalize_text(v)))

        if self.profile.matches_doc_name(doc_name):
            page_range = self.profile.resolve_page_range(filters["major_tag"], filters["medium_tag"])
            if page_range:
                filters["page_range"] = page_range
        return filters

    def demo_queries(self) -> list[str]:
        return self.profile.demo_queries()

    def _best_tag(self, text: str, definitions: dict[str, Any], default: str) -> str:
        best_tag = default
        best_score = 0
        for tag_name, spec in definitions.items():
            score = self._score(text, spec)
            if score > best_score:
                best_tag = tag_name
                best_score = score
        return best_tag

    @staticmethod
    def _matches(text: str, spec: dict[str, Any]) -> bool:
        return TaggingEngine._score(text, spec) > 0

    @staticmethod
    def _score(text: str, spec: dict[str, Any]) -> int:
        score = 0
        for keyword in spec.get("keywords", []):
            if keyword.lower() in text:
                score += 1
        for pattern in spec.get("regex", []):
            if re.search(pattern, text, flags=re.IGNORECASE):
                score += 2
        return score

    def _minor_tag(self, text: str) -> str:
        for pattern in self.rules.get("minor_patterns", []):
            hits = re.findall(pattern, text, flags=re.IGNORECASE)
            if hits:
                return normalize_text(hits[0])
        return normalize_text(text[:48])
