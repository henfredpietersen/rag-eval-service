"""Evaluation: retrieval hit-rate, faithfulness proxy, abstention accuracy."""

from __future__ import annotations

import json
from pathlib import Path

from app.rag import answer_query
from app.store import VectorStore


def load_golden(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def faithfulness(answer: str, contexts: list[str], abstained: bool) -> float:
    if abstained:
        return 1.0
    tokens = [t for t in answer.lower().split() if len(t) > 3]
    if not tokens:
        return 0.0
    blob = " ".join(contexts).lower()
    supported = sum(1 for t in tokens if t in blob)
    return supported / len(tokens)


def run_eval(store: VectorStore, golden: list[dict], k: int = 3, min_score: float = 0.12) -> dict:
    rows = []
    for item in golden:
        result = answer_query(store, item["question"], k=k, min_score=min_score)
        expected_docs = set(item.get("expected_docs") or [])
        got_docs = {r["doc_id"] for r in result.retrieved[:k]}
        hit = bool(expected_docs & got_docs) if expected_docs else None
        top1 = result.retrieved[0]["doc_id"] if result.retrieved else None
        hit1 = (top1 in expected_docs) if expected_docs else None
        ctx = [r["text"] for r in result.retrieved if r["score"] >= min_score]
        faith = faithfulness(result.text, ctx, result.abstained)
        should_abstain = bool(item.get("should_abstain"))
        abstain_ok = result.abstained == should_abstain
        rows.append(
            {
                "id": item["id"],
                "hit": hit,
                "hit_at_1": hit1,
                "top1": top1,
                "faithfulness": round(faith, 3),
                "abstained": result.abstained,
                "abstain_ok": abstain_ok,
                "latency_ms": result.latency_ms,
                "answer": result.text,
            }
        )
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    hits1 = [r["hit_at_1"] for r in rows if r["hit_at_1"] is not None]
    return {
        "n": len(rows),
        "k": k,
        "min_score": min_score,
        "embedder": getattr(getattr(store, "embedder", None), "name", "unknown"),
        "retrieval_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        "hit_rate_at_1": round(sum(hits1) / len(hits1), 3) if hits1 else None,
        "mean_faithfulness": round(sum(r["faithfulness"] for r in rows) / len(rows), 3),
        "abstain_accuracy": round(sum(r["abstain_ok"] for r in rows) / len(rows), 3),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 2),
        "rows": rows,
    }
