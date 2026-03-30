from __future__ import annotations

from typing import Any

from core.audit import audit_logger
from core.config import settings
from core.llm import AnswerEngine
from core.pdf_utils import verify_payload_against_pdf
from core.qdrant_store import QdrantStore, SearchHit
from core.tagging import TaggingEngine


class TechnicalRetriever:
    def __init__(self, store: QdrantStore, llm: AnswerEngine) -> None:
        self.store = store
        self.llm = llm
        self.tagger = TaggingEngine()

    def infer_key_filters(self, query: str, doc_name: str | None = None) -> dict[str, list[str]]:
        return self.tagger.infer_filters(query, doc_name=doc_name)

    def answer(self, query: str, doc_name: str | None = None) -> dict[str, Any]:
        filters = self.infer_key_filters(query, doc_name=doc_name)
        raw_hits = self.store.search(query=query, tag_filters=filters, limit=settings.max_hits)
        reranked = self._rerank(query, raw_hits, filters)

        evidence: list[dict[str, Any]] = []
        for hit in reranked[:4]:
            payload = dict(hit.payload)
            source_pdf_path = payload.get("source_pdf_path")
            if source_pdf_path:
                verification = verify_payload_against_pdf(source_pdf_path, payload)
                payload["verified"] = verification["verified"]
                payload["verification_summary"] = verification["verification_summary"]
                if verification.get("corrected_value"):
                    payload["value"] = verification["corrected_value"]
                    if isinstance(payload.get("shorthand_json"), dict) and payload.get("row_label") and payload.get("col_label"):
                        payload["shorthand_json"].setdefault(payload["row_label"], {})
                        payload["shorthand_json"][payload["row_label"]][payload["col_label"]] = verification["corrected_value"]
            evidence.append(payload)

        answer = self.llm.generate_answer(query, evidence)
        audit_logger.log(
            "query_answered",
            {
                "query": query,
                "doc_name": doc_name,
                "filters": filters,
                "hit_count": len(raw_hits),
                "top_hit": {
                    "point_id": evidence[0].get("point_id") if evidence else None,
                    "page_no": evidence[0].get("page_no") if evidence else None,
                    "verified": evidence[0].get("verified") if evidence else None,
                },
            },
        )
        return {
            "answer": answer,
            "filters": filters,
            "hits": evidence,
            "top_hit": evidence[0] if evidence else None,
        }

    def _rerank(self, query: str, hits: list[SearchHit], filters: dict[str, list[str]]) -> list[SearchHit]:
        q = query.lower()
        scored: list[tuple[float, SearchHit]] = []
        for hit in hits:
            payload = hit.payload
            score = hit.score

            if payload.get("verified"):
                score += 0.18
            if payload.get("favorite"):
                score += 0.10
            score += min(int(payload.get("favorite_weight", 0)) * 0.05, 0.30)

            if payload.get("major_tag") in filters.get("major_tag", []):
                score += 0.25
            if payload.get("medium_tag") in filters.get("medium_tag", []):
                score += 0.20
            if payload.get("minor_tag") in filters.get("minor_tag", []):
                score += 0.14

            for field_name in ["row_label", "col_label", "value", "doc_name"]:
                field_value = str(payload.get(field_name, "")).lower()
                if field_value and field_value in q:
                    score += 0.08

            if payload.get("parser_backend") == "docling":
                score += 0.02
            if payload.get("chunk_type") == "table_cell":
                score += 0.05

            scored.append((score, hit))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in scored]
