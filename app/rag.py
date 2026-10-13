"""RAG pipeline: retrieve, abstain on weak evidence, answer from context only."""

from __future__ import annotations

from dataclasses import dataclass

from app.store import VectorStore

ABSTAIN = "I don't have enough evidence in the indexed documents to answer that."


@dataclass
class Answer:
    text: str
    citations: list[str]
    retrieved: list[dict]
    abstained: bool
    latency_ms: float
    backend: str


def _extractive_answer(query: str, contexts: list[str]) -> str:
    """Offline backend: return the highest-overlap sentence. Not a generative model."""
    q_tokens = set(query.lower().split())
    best = ""
    best_score = -1
    for ctx in contexts:
        for sentence in ctx.replace("?", ".").split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            score = len(q_tokens & set(sentence.lower().split()))
            if score > best_score:
                best_score = score
                best = sentence
    return best


STOP = {
    "the", "and", "for", "with", "that", "this", "from", "have", "what", "how",
    "when", "where", "does", "your", "you", "are", "was", "not", "can", "per",
}


def _content(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 3 and t not in STOP}


def _overlap(query: str, text: str) -> float:
    q = _content(query)
    if not q:
        return 0.0
    return len(q & _content(text)) / len(q)


def answer_query(
    store: VectorStore,
    query: str,
    k: int = 3,
    min_score: float = 0.12,
    min_overlap: float = 0.34,
    backend: str = "extractive",
) -> Answer:
    import time

    t0 = time.perf_counter()
    hits = store.search(query, k=k)
    retrieved = [
        {"doc_id": h.doc_id, "score": round(h.score, 4), "text": h.text} for h in hits
    ]
    strong = [h for h in hits if _overlap(query, h.text) >= min_overlap]
    if not strong:
        return Answer(
            text=ABSTAIN,
            citations=[],
            retrieved=retrieved,
            abstained=True,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            backend=backend,
        )
    contexts = [h.text for h in strong]
    text = _extractive_answer(query, contexts)
    citations = sorted({h.doc_id for h in strong})
    return Answer(
        text=text,
        citations=citations,
        retrieved=retrieved,
        abstained=False,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        backend=backend,
    )
