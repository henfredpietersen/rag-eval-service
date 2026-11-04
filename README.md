# rag-eval-service

**Try it live:** [outview.co.za/case-study-rag-eval](https://outview.co.za/case-study-rag-eval/). Ask questions, see what it retrieves and why it answers or refuses, and run the golden-set eval in your browser.

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

Output of `make compare`, run on GitHub's servers by `.github/workflows/minilm-compare.yml` on 4 November 2026 (sentence-transformers 6.2.0, CPU only, `all-MiniLM-L6-v2` at revision `1110a243`). The full output, with versions, is in [`results/compare.md`](results/compare.md). The cloud workspace this was built in can't download model weights, which is why MiniLM runs there instead. Re-run it from the Actions tab at any time.

```
### golden.json (n=4, k=3)

| metric | hashing | minilm |
| --- | --- | --- |
| retrieval_hit_rate | 1.0 | 1.0 |
| hit_rate_at_1 | 0.667 | 1.0 |
| mean_faithfulness | 1.0 | 1.0 |
| abstain_accuracy | 1.0 | 1.0 |
| mean_latency_ms | 0.16 | 10.09 |

hashing: abstained on ['q4']
minilm: abstained on ['q4']

### golden_hard.json (n=5, k=3)

| metric | hashing | minilm |
| --- | --- | --- |
| retrieval_hit_rate | 1.0 | 1.0 |
| hit_rate_at_1 | 0.75 | 1.0 |
| mean_faithfulness | 1.0 | 1.0 |
| abstain_accuracy | 0.4 | 0.4 |
| mean_latency_ms | 0.13 | 9.77 |

hashing: abstained on ['q5', 'q7', 'q8', 'q9']
minilm: abstained on ['q5', 'q7', 'q8', 'q9']
```

How to read it:

- **`retrieval_hit_rate` (@3) is meaningless here.** The corpus is 3 chunks and k=3, so every chunk is always retrieved. `hit_rate_at_1` is the retrieval number that means something.
- **`mean_faithfulness` = 1.0 is by construction.** The reader copies a sentence from the context, so its tokens are always in the context. The metric only becomes informative once a generative reader is swapped in.
- **The wifi question (q4) abstains**, as does "Who is the CEO" (q9).
- **MiniLM ranks better.** `hit_rate_at_1` goes from 0.667 to 1.0 on the easy set and from 0.75 to 1.0 on the paraphrase set. On 3 and 4 answerable questions, that's one question each.
- **`golden_hard.json` shows the real weakness, and MiniLM doesn't fix it.** These are paraphrased questions ("When will my refund reach my card?"). The lexical abstain gate refused 3 answerable questions (q5, q7, q8) under both embedders, so `abstain_accuracy` stays at 0.4. That was the prediction before MiniLM was run: the gate checks word overlap, not the embedding score, so a better embedder can't change what it refuses. The fix is a cosine threshold calibrated per embedder on a labelled set. That work has not been done yet.
- Latency depends on the machine and excludes any LLM call. MiniLM's ~10 ms per question against the hashing embedder's ~0.1 ms is the expected order of difference, nothing more. Do not quote it.
- n=4 and n=5. These are smoke tests, not a benchmark.

## Web demo

`docs/walkthrough.js` ([live](https://outview.co.za/case-study-rag-eval/)) is an interactive page that runs `app/chunk.py`, `embed.py`, `store.py`, `rag.py`, `eval.py` and `web/driver.py`, unchanged, in the visitor's browser with [Pyodide](https://pyodide.org), using the hashing embedder. It shows each question's content words, the retrieved chunks with their scores and gate overlap, and runs `run_eval` on both golden sets live. MiniLM can't run in a browser, so its column shows the recorded figures from `results/compare.md`, labelled as recorded.

```bash
python web/build.py        # rebuilds docs/walkthrough.js from web/page.js, app/ and data/
```

`tests/test_web_demo.py` fails if the built page is out of date, checks every example on the page behaves as its note says, and checks the live eval reproduces the numbers above, and checks the page's recorded MiniLM figures match `results/compare.md`. Every page question and both evals were also run in Pyodide 0.26.4 under Node and matched CPython exactly.

## Tests

`tests/test_rag.py` (original), `tests/test_embedders.py` and `tests/test_web_demo.py` (22 pass, 1 skipped):

- MiniLM import stays lazy. A missing package raises a clear `EmbedderUnavailable`.
- A stub `SentenceTransformer` proves MiniLM plugs into the same store and eval. It tests the wiring, not the quality.
- The wifi question abstains under both embedders.
- `test_real_minilm_if_available` runs the real model when it can load, and skips otherwise. It is skipped in the cloud workspace, which can't download the weights. The real model's scores come from the comparison workflow instead.

## What I would not ship

- The hashing embedder. It is for deterministic tests only.
- The lexical abstain gate as the only refusal check. It refuses paraphrases (see the hard set).
- An extractive reader in place of a generative model, or a faithfulness proxy that can't fail for it.
- An in-process store with no persistence. Use pgvector or FAISS behind `VectorStore.search`.
- Any claim from a 9-question golden set.
- The Docker image as-is. `compose.yaml` and the `Dockerfile` were **not built or run** in this session (no Docker daemon in the workspace).
