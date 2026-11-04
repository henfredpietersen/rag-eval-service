"""Glue between the demo page and the app package.

The page runs this file plus app/chunk.py, embed.py, store.py, rag.py and eval.py,
unchanged, in the visitor's browser (Pyodide), with the hashing embedder. app/main.py
is not used because it needs FastAPI. tests/test_web_demo.py runs the same calls under
CPython.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.chunk import chunk_text
from app.embed import HashingEmbedder
from app.eval import load_golden, run_eval
from app.rag import _content, answer_query
from app.store import VectorStore

# In the browser the data sits next to this file; in the repo it's one level up.
_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data" if (_HERE / "data").is_dir() else _HERE.parent / "data"
MIN_OVERLAP = 0.34  # app.rag.answer_query default

_store: VectorStore | None = None


def store() -> VectorStore:
    """Same corpus loading as app.main.load_corpus: one chunk set per markdown file."""
    global _store
    if _store is None:
        s = VectorStore(HashingEmbedder())
        items = []
        for path in sorted((DATA / "corpus").glob("*.md")):
            for i, chunk in enumerate(chunk_text(path.read_text(), size=500, overlap=60)):
                items.append((f"{path.stem}#{i}", chunk))
        s.add_many(items)
        _store = s
    return _store


def ask(question: str) -> str:
    s = store()
    result = answer_query(s, question)
    q_words = sorted(_content(question))
    chunks = []
    for r in result.retrieved:
        shared = sorted(_content(question) & _content(r["text"]))
        share = len(shared) / len(q_words) if q_words else 0.0
        chunks.append(
            {
                "doc_id": r["doc_id"],
                "score": r["score"],
                "text": r["text"],
                "shared_words": shared,
                "overlap": round(share, 3),
                "passes_gate": share >= MIN_OVERLAP,
            }
        )
    return json.dumps(
        {
            "question": question,
            "answer": result.text,
            "abstained": result.abstained,
            "citations": result.citations,
            "question_words": q_words,
            "min_overlap": MIN_OVERLAP,
            "chunks": chunks,
            "total_chunks": len(s),
        }
    )


def evaluate(name: str) -> str:
    """Run app.eval.run_eval on one golden set: 'golden' or 'golden_hard'."""
    golden = load_golden(DATA / f"{name}.json")
    report = run_eval(store(), golden)
    questions = {g["id"]: g for g in golden}
    for row in report["rows"]:
        g = questions[row["id"]]
        row["question"] = g["question"]
        row["expected_docs"] = g.get("expected_docs") or []
        row["should_abstain"] = bool(g.get("should_abstain"))
    return json.dumps(report)
