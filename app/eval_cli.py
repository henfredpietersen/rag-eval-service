"""CLI: python -m app.eval_cli"""

import json
from pathlib import Path

from app.eval import load_golden, run_eval
from app.main import bootstrap, store

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bootstrap()
    report = run_eval(store, load_golden(ROOT / "data" / "golden.json"))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print("--- rows ---")
    print(json.dumps(report["rows"], indent=2))


if __name__ == "__main__":
    main()
