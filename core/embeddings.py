from __future__ import annotations

import requests
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

from core.config import settings


class Embedder:
    def __init__(self) -> None:
        self.backend = settings.embedding_backend
        self.model_name = settings.embedding_model
        self._model = None
        self._dimension = None

        if self.backend == "sentence_transformers":
            self._model = SentenceTransformer(self.model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        elif self.backend == "ollama":
            probe = self.embed(["dimension probe"])[0]
            self._dimension = len(probe)
        else:
            raise ValueError(f"Unsupported embedding backend: {self.backend}")

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embedding dimension is not initialized.")
        return self._dimension

    def embed(self, texts: Iterable[str]) -> List[list[float]]:
        texts = list(texts)
        if not texts:
            return []

        if self.backend == "sentence_transformers":
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vectors]

        if self.backend == "ollama":
            results: list[list[float]] = []
            for text in texts:
                response = requests.post(
                    f"{settings.ollama_host}/api/embeddings",
                    json={"model": settings.ollama_embed_model, "prompt": text},
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                results.append(payload["embedding"])
            return results

        raise ValueError(f"Unsupported embedding backend: {self.backend}")
