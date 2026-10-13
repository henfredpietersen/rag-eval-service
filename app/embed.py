"""Embedders behind one interface.

- HashingEmbedder: deterministic, no download, no GPU. For tests and demos only.
- MiniLMEmbedder: sentence-transformers all-MiniLM-L6-v2. Imported lazily so the
  test suite runs without sentence-transformers or torch installed.

Both return L2-normalised vectors, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Iterable, Protocol

DIM = 256
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str, dim: int = DIM) -> list[float]:
    """Hashing-trick bag of words. Kept as a module function for backward compatibility."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_many(texts: Iterable[str]) -> list[list[float]]:
    return [embed(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [embed(t, self.dim) for t in texts]


class EmbedderUnavailable(RuntimeError):
    """Raised when an optional embedder cannot be loaded (package or weights missing)."""


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = MINILM_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderUnavailable(
                    "sentence-transformers is not installed. "
                    "pip install -r requirements-minilm.txt"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # network blocked, no cache, bad name
                raise EmbedderUnavailable(
                    f"could not load {self.model_name}: {type(exc).__name__}: {exc}"
                ) from exc
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


def get_embedder(name: str | None = None) -> Embedder:
    name = (name or os.environ.get("EMBEDDER") or "hashing").lower()
    if name == "hashing":
        return HashingEmbedder()
    if name == "minilm":
        return MiniLMEmbedder()
    raise ValueError(f"unknown embedder {name!r}; use 'hashing' or 'minilm'")
