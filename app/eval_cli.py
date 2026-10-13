"""CLI.

python -m app.eval_cli                     # default embedder (EMBEDDER env, else hashing)
python -m app.eval_cli --embedder minilm   # one embedder, full rows
python -m app.eval_cli --compare           # hashing vs minilm, markdown table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.embed import EmbedderUnavailable, get_embedder
from app.eval import load_golden, run_eval
from app.main import load_corpus
from app.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden.json"
GOLDEN_HARD = ROOT / "data" / "golden_hard.json"
METRICS = ["retrieval_hit_rate", "hit_rate_at_1", "mean_faithfulness", "abstain_accuracy", "mean_latency_ms"]


def evaluate(name: str | None, golden: Path = GOLDEN) -> dict:
    store = load_corpus(VectorStore(get_embedder(name)))
    return run_eval(store, load_golden(golden))


def compare(names: list[str], golden: Path = GOLDEN) -> str:
    results: dict[str, dict | str] = {}
    for name in names:
        try:
            results[name] = evaluate(name, golden)
        except EmbedderUnavailable as exc:
            results[name] = f"not measured ({exc})"
    lines = ["| metric | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    n = next((r["n"] for r in results.values() if isinstance(r, dict)), "?")
    for m in METRICS:
        cells = [str(r[m]) if isinstance(r, dict) else "not measured" for r in results.values()]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    abst = []
    for name, r in results.items():
        if isinstance(r, dict):
            fired = [row["id"] for row in r["rows"] if row["abstained"]]
            abst.append(f"{name}: abstained on {fired}")
        else:
            abst.append(f"{name}: {r}")
    return f"### {golden.name} (n={n}, k=3)\n\n" + "\n".join(lines) + "\n\n" + "\n".join(abst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--golden", type=Path, default=GOLDEN)
    args = p.parse_args()
    if args.compare:
        for golden in (GOLDEN, GOLDEN_HARD):
            print(compare(["hashing", "minilm"], golden))
            print()
        return
    report = evaluate(args.embedder, args.golden)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print("--- rows ---")
    print(json.dumps(report["rows"], indent=2))


if __name__ == "__main__":
    main()
