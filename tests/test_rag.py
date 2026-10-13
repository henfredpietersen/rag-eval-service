from pathlib import Path

from fastapi.testclient import TestClient

from app.eval import run_eval, load_golden
from app.main import app, bootstrap, store

ROOT = Path(__file__).resolve().parents[1]


def test_eval_hits_known_docs_and_abstains():
    bootstrap()
    report = run_eval(store, load_golden(ROOT / "data" / "golden.json"))
    assert report["retrieval_hit_rate"] == 1.0
    assert report["abstain_accuracy"] == 1.0
    assert report["mean_faithfulness"] >= 0.5


def test_ask_endpoint_cites_source():
    client = TestClient(app)
    res = client.post("/ask", json={"question": "What does the Growth plan cost per month?"})
    assert res.status_code == 200
    body = res.json()
    assert body["abstained"] is False
    assert any(c.startswith("pricing") for c in body["citations"])
