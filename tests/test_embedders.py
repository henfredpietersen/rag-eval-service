"""Second embedder behind the same interface.

The stub tests prove the wiring, not retrieval quality. Retrieval quality for
MiniLM only comes from `make compare` on a machine that can download the weights.
"""

import sys
import types
from pathlib import Path

import pytest

from app.embed import EmbedderUnavailable, HashingEmbedder, MiniLMEmbedder, get_embedder
from app.eval import load_golden, run_eval
from app.main import load_corpus
from app.rag import answer_query
from app.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
WIFI = "What is the office wifi password?"


class _FakeST:
    """Stands in for SentenceTransformer: normalised hashing vectors, numpy-free."""

    def __init__(self, name):
        self.name = name

    def encode(self, texts, normalize_embeddings=True):
        return HashingEmbedder(dim=64).embed(list(texts))


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return mod


def test_minilm_import_is_lazy():
    emb = get_embedder("minilm")
    assert isinstance(emb, MiniLMEmbedder)
    assert emb._model is None  # nothing loaded until first embed()


def test_unknown_embedder_rejected():
    with pytest.raises(ValueError):
        get_embedder("word2vec")


def test_minilm_missing_package_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # forces ImportError
    with pytest.raises(EmbedderUnavailable, match="not installed"):
        MiniLMEmbedder().embed(["hello"])


def test_minilm_interface_with_stub_model(fake_sentence_transformers):
    store = load_corpus(VectorStore(MiniLMEmbedder()))
    assert len(store) == 3
    hits = store.search("Growth plan cost", k=3)
    assert len(hits) == 3
    assert all(isinstance(h.score, float) for h in hits)


@pytest.mark.parametrize("name", ["hashing", "minilm"])
def test_wifi_question_abstains_for_each_embedder(name, fake_sentence_transformers):
    store = load_corpus(VectorStore(get_embedder(name)))
    result = answer_query(store, WIFI)
    assert result.abstained is True
    assert result.citations == []


def test_compare_marks_unavailable_embedder_not_measured(monkeypatch):
    from app import eval_cli

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    out = eval_cli.compare(["hashing", "minilm"])
    assert "not measured" in out
    assert "| abstain_accuracy | 1.0 | not measured |" in out


def test_real_minilm_if_available():
    pytest.importorskip("sentence_transformers")
    emb = MiniLMEmbedder()
    try:
        emb.embed(["warm up"])
    except EmbedderUnavailable as exc:
        pytest.skip(f"MiniLM weights not loadable here: {exc}")
    store = load_corpus(VectorStore(emb))
    report = run_eval(store, load_golden(ROOT / "data" / "golden.json"))
    assert report["abstain_accuracy"] == 1.0
    assert answer_query(store, WIFI).abstained is True
