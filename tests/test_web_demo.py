"""The demo page runs app/ and web/driver.py in the browser with the hashing embedder.

These tests run the same driver under CPython and check that the page's examples and
eval numbers match what the page says, and that the built page embeds the current code.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import build  # noqa: E402
import driver  # noqa: E402


def ask(q):
    return json.loads(driver.ask(q))


def test_built_page_embeds_current_code():
    page = (ROOT / "docs" / "walkthrough.js").read_text()
    assert page == build.build(), "docs/walkthrough.js is stale: run python web/build.py"


@pytest.mark.parametrize(
    "question, abstained, cites",
    [
        ("What does the Growth plan cost per month?", False, "pricing#0"),
        ("Is there weekend phone support on Starter?", False, "support#0"),
        ("Is there a once-off setup fee?", False, "pricing#0"),
        ("What is the office wifi password?", True, None),
        ("Who is the CEO of the company?", True, None),
        # Known failures the page labels as such:
        ("When will my refund reach my card?", True, None),
        ("What time does the help desk open on weekdays?", True, None),
        ("Can I send back a gift card?", True, None),
        ("How long do I have to return an unused item?", False, "returns#0"),
    ],
)
def test_page_examples_behave_as_described(question, abstained, cites):
    a = ask(question)
    assert a["abstained"] is abstained
    if cites:
        assert cites in a["citations"]


def test_specific_claims_in_the_page_notes():
    # "Right answer, wrong top hit"
    a = ask("How long do I have to return an unused item?")
    assert a["chunks"][0]["doc_id"] == "support#0" and a["citations"] == ["returns#0"]
    # "weekdays." in the document vs "weekdays?" in the question: no shared words at all
    a = ask("What time does the help desk open on weekdays?")
    assert all(c["shared_words"] == [] for c in a["chunks"])
    # Gift card: only "gift" matches, 1 of 4 words
    a = ask("Can I send back a gift card?")
    best = max(a["chunks"], key=lambda c: c["overlap"])
    assert best["shared_words"] == ["gift"] and best["overlap"] == 0.25
    # The corpus is 3 chunks and k=3, so every chunk is always retrieved
    assert a["total_chunks"] == 3 and len(a["chunks"]) == 3


def test_live_eval_matches_readme_numbers():
    easy = json.loads(driver.evaluate("golden"))
    assert (easy["hit_rate_at_1"], easy["retrieval_hit_rate"], easy["abstain_accuracy"]) == (0.667, 1.0, 1.0)
    hard = json.loads(driver.evaluate("golden_hard"))
    assert (hard["hit_rate_at_1"], hard["retrieval_hit_rate"], hard["abstain_accuracy"]) == (0.75, 1.0, 0.4)
    assert [r["id"] for r in hard["rows"] if r["abstained"]] == ["q5", "q7", "q8", "q9"]


def test_recorded_minilm_figures_match_the_workflow_output():
    """The page shows MiniLM scores it can't compute in a browser; they must equal results/compare.md."""
    import re

    md = (ROOT / "results" / "compare.md").read_text(encoding="utf-8")
    js = (ROOT / "web" / "page.js").read_text(encoding="utf-8")
    for name in ("golden", "golden_hard"):
        section = md.split(f"### {name}.json")[1].split("###")[0]
        recorded = {m: float(v) for m, _, v in re.findall(r"\| (\w+) \| ([\d.]+) \| ([\d.]+) \|", section)}
        page = re.search(name + r": \{([^}]*)\}", js).group(1)
        shown = {k: float(v) for k, v in re.findall(r"(\w+): ([\d.]+)", page)}
        assert shown and all(recorded[k] == v for k, v in shown.items()), (name, shown, recorded)
