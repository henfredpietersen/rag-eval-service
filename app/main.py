"""FastAPI surface for the RAG demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.chunk import chunk_text
from app.eval import load_golden, run_eval
from app.rag import answer_query
from app.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
store = VectorStore()
app = FastAPI(title="rag-eval-service", version="0.1.0")


def bootstrap() -> None:
    if len(store):
        return
    corpus = ROOT / "data" / "corpus"
    for path in sorted(corpus.glob("*.md")):
        for i, chunk in enumerate(chunk_text(path.read_text(), size=500, overlap=60)):
            store.add(f"{path.stem}#{i}", chunk)


class Ask(BaseModel):
    question: str
    k: int = 3
    min_score: float = 0.12


@app.on_event("startup")
def _startup() -> None:
    bootstrap()


@app.get("/health")
def health() -> dict:
    bootstrap()
    return {"ok": True, "chunks": len(store)}


@app.post("/ask")
def ask(body: Ask) -> dict:
    bootstrap()
    result = answer_query(store, body.question, k=body.k, min_score=body.min_score)
    return {
        "answer": result.text,
        "citations": result.citations,
        "abstained": result.abstained,
        "latency_ms": result.latency_ms,
        "backend": result.backend,
        "retrieved": result.retrieved,
    }


@app.post("/eval")
def evaluate(k: int = 3, min_score: float = 0.12) -> dict:
    bootstrap()
    golden = load_golden(ROOT / "data" / "golden.json")
    return run_eval(store, golden, k=k, min_score=min_score)
