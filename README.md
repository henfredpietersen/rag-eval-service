# rag-eval-service

Retrieval-augmented answering with a repeatable eval harness. Offline by default: hashing embeddings, an in-process vector store, and an extractive reader so `make test` needs no API key and no GPU.

This is the portfolio slice for “own the lifecycle”: ingest, retrieve, abstain, score, compare a config change.

## What it proves

- Chunking, embeddings, top-k retrieval, citations
- Abstain when the best score is below a threshold (hallucination guard, not a vibe check)
- Golden-set metrics: retrieval hit rate, faithfulness proxy, abstain accuracy, latency
- A knob you can turn and re-score (`k`, `min_score`)

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test
make eval
make serve
# POST /ask {"question": "What does the Growth plan cost per month?"}
# POST /eval
```

## Metrics on the checked-in golden set

| metric | value |
| --- | --- |
| retrieval hit rate @3 | 1.0 |
| abstain accuracy | 1.0 |
| faithfulness proxy | token overlap with retrieved context |

Faithfulness here is a proxy (answer tokens supported by context). Swap the extractive reader for an LLM and keep the same harness.

## Swap-in production pieces

- Embeddings: replace `app/embed.py` with OpenAI or a local sentence-transformer. Keep the store interface.
- Store: pgvector or FAISS behind `VectorStore.search`.
- Reader: hosted model or vLLM. Log tokens and rand cost next to latency.
- Do not ship the hashing embedder. It is for deterministic demos only.

## Layout

```
app/chunk.py app/embed.py app/store.py app/rag.py app/eval.py app/main.py
data/corpus/   data/golden.json
tests/test_rag.py
```
