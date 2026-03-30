from __future__ import annotations

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
                # 既に存在する場合などは無視
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

        # 現行の Qdrant Query API を優先
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            raw_points = getattr(response, "points", response)

        # 旧来APIへのフォールバック
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
