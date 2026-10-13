"""In-process vector store. Swap for pgvector/FAISS in production."""

from __future__ import annotations

from dataclasses import dataclass

from app.embed import Embedder, HashingEmbedder, cosine


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._rows: list[tuple[str, str, list[float]]] = []

    def add(self, doc_id: str, text: str) -> None:
        self.add_many([(doc_id, text)])

    def add_many(self, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        vecs = self.embedder.embed([t for _, t in items])
        for (doc_id, text), vec in zip(items, vecs):
            self._rows.append((doc_id, text, vec))

    def __len__(self) -> int:
        return len(self._rows)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        ranked = sorted(
            (Hit(doc_id, text, cosine(q, vec)) for doc_id, text, vec in self._rows),
            key=lambda h: h.score,
            reverse=True,
        )
        return ranked[:k]
