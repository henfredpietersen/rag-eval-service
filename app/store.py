"""In-process vector store. Swap for pgvector/FAISS in production."""

from __future__ import annotations

from dataclasses import dataclass

from app.embed import cosine, embed


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._rows: list[tuple[str, str, list[float]]] = []

    def add(self, doc_id: str, text: str) -> None:
        self._rows.append((doc_id, text, embed(text)))

    def __len__(self) -> int:
        return len(self._rows)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = embed(query)
        ranked = sorted(
            (Hit(doc_id, text, cosine(q, vec)) for doc_id, text, vec in self._rows),
            key=lambda h: h.score,
            reverse=True,
        )
        return ranked[:k]
