"""Build docs/walkthrough.js (and a local preview page) from web/page.js.

    python web/build.py

The page embeds the app modules it needs, the corpus, the golden sets and web/driver.py
byte for byte, and runs them in the browser with Pyodide. tests/test_web_demo.py fails
if the built page is out of date with those files.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BROWSER_HOME = "/home/pyodide"

# app/main.py is left out on purpose: it needs FastAPI, which the page doesn't use.
SOURCES = [
    "app/__init__.py",
    "app/chunk.py",
    "app/embed.py",
    "app/store.py",
    "app/rag.py",
    "app/eval.py",
    "data/golden.json",
    "data/golden_hard.json",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data" / "corpus").glob("*.md")),
]

PREVIEW = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG eval service: live demo</title>
<style>body{margin:0;background:#05070d}</style>
</head><body>
<div id="re-walkthrough" style="min-height:600px;background:#05070d"></div>
<script src="walkthrough.js"></script>
</body></html>
"""


def embedded_files() -> dict:
    files = {f"{BROWSER_HOME}/{src}": (ROOT / src).read_text() for src in SOURCES}
    files[f"{BROWSER_HOME}/demo_driver.py"] = (ROOT / "web" / "driver.py").read_text()
    return files


def build() -> str:
    template = (ROOT / "web" / "page.js").read_text()
    assert template.count("__FILES__") == 1
    return template.replace("__FILES__", json.dumps(embedded_files()))


def main() -> None:
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "walkthrough.js").write_text(build())
    (out / "walkthrough.html").write_text(PREVIEW)
    print("Wrote docs/walkthrough.js and docs/walkthrough.html")


if __name__ == "__main__":
    main()
