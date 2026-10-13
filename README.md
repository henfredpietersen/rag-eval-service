# rag-eval-service

## Problem

A retrieval-augmented answering service needs to (a) find the right document, (b) refuse when the documents do not contain the answer, and (c) let you change one component and re-score it on the same questions. This repo is a small version of that loop: chunk, embed, retrieve, abstain, answer, score.

Built AI-assisted under my direction. I specified the behaviour, reviewed the output, and ran the commands shown here.

## How it works

- `app/chunk.py`: fixed-window chunker with overlap.
- `app/embed.py`: two embedders behind one interface (`embed(texts) -> vectors`):
  - `hashing`: hashing-trick bag of words. Deterministic, no download. Demo and test use only.
  - `minilm`: `sentence-transformers/all-MiniLM-L6-v2`. Imported lazily, so the tests run without it installed.
- `app/store.py`: in-process vector store, cosine over normalised vectors.
- `app/rag.py`: retrieve top-k, then the **abstain gate**: answer only if at least one retrieved chunk shares ≥34% of the question's content words. Otherwise return a fixed refusal. The reader is extractive (it returns the best-matching sentence). It is not a generative model.
- `app/eval.py`: golden-set scoring.

Choose the embedder with `EMBEDDER=hashing|minilm` or `--embedder`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test        # pytest
make eval        # one embedder, per-question rows
make compare     # hashing vs MiniLM on both golden sets
make serve       # POST /ask {"question": "..."}, POST /eval, GET /health

# MiniLM (downloads ~90 MB of weights from HuggingFace on first run)
pip install -r requirements-minilm.txt
make compare

# Docker, one service, one command
docker compose up --build
WITH_MINILM=1 EMBEDDER=minilm docker compose up --build
```

## Measured

Output of `make compare` in this repo, pasted unedited. Run on a cloud workspace with no GPU. HuggingFace was blocked from that workspace, so MiniLM could not load and shows "not measured". Run `make compare` on a machine with internet access to fill that column.

```
### golden.json (n=4, k=3)

| metric | hashing | minilm |
| --- | --- | --- |
| retrieval_hit_rate | 1.0 | not measured |
| hit_rate_at_1 | 0.667 | not measured |
| mean_faithfulness | 1.0 | not measured |
| abstain_accuracy | 1.0 | not measured |
| mean_latency_ms | 0.15 | not measured |

hashing: abstained on ['q4']
minilm: not measured (sentence-transformers is not installed. pip install -r requirements-minilm.txt)

### golden_hard.json (n=5, k=3)

| metric | hashing | minilm |
| --- | --- | --- |
| retrieval_hit_rate | 1.0 | not measured |
| hit_rate_at_1 | 0.75 | not measured |
| mean_faithfulness | 1.0 | not measured |
| abstain_accuracy | 0.4 | not measured |
| mean_latency_ms | 0.12 | not measured |

hashing: abstained on ['q5', 'q7', 'q8', 'q9']
minilm: not measured (sentence-transformers is not installed. pip install -r requirements-minilm.txt)
```

How to read it:

- **`retrieval_hit_rate` (@3) is meaningless here.** The corpus is 3 chunks and k=3, so every chunk is always retrieved. `hit_rate_at_1` is the retrieval number that means something.
- **`mean_faithfulness` = 1.0 is by construction.** The reader copies a sentence from the context, so its tokens are always in the context. The metric only becomes informative once a generative reader is swapped in.
- **The wifi question (q4) abstains**, as does "Who is the CEO" (q9).
- **`golden_hard.json` shows the real weakness.** These are paraphrased questions ("When will my refund reach my card?"). The lexical abstain gate refused 3 answerable questions (q5, q7, q8). The gate does not depend on the embedder, so a better embedder can improve `hit_rate_at_1` but **will not** fix these false refusals. The fix is a cosine threshold calibrated per embedder on a labelled set. That work has not been done yet.
- Latency is sub-millisecond, noisy run to run, and excludes any LLM call. Do not quote it.
- n=4 and n=5. These are smoke tests, not a benchmark.

## Tests

`tests/test_rag.py` (original) and `tests/test_embedders.py`:

- MiniLM import stays lazy. A missing package raises a clear `EmbedderUnavailable`.
- A stub `SentenceTransformer` proves MiniLM plugs into the same store and eval. It tests the wiring, not the quality.
- The wifi question abstains under both embedders.
- `test_real_minilm_if_available` runs the real model when it can load, and skips otherwise. It was **skipped** in the run recorded here.

## What I would not ship

- The hashing embedder. It is for deterministic tests only.
- The lexical abstain gate as the only refusal check. It refuses paraphrases (see the hard set).
- An extractive reader in place of a generative model, or a faithfulness proxy that can't fail for it.
- An in-process store with no persistence. Use pgvector or FAISS behind `VectorStore.search`.
- Any claim from a 9-question golden set.
- The Docker image as-is. `compose.yaml` and the `Dockerfile` were **not built or run** in this session (no Docker daemon in the workspace).
