# RAGAS Evaluation for DlightRAG

DlightRAG reuses LightRAG's built-in [RAGAS](https://docs.ragas.io/) evaluation
framework. The adapter in `scripts/ragas_eval.py` inherits from LightRAG's
`RAGEvaluator` and translates DlightRAG's `/api/answer` response format so the
rest of the evaluation pipeline — metrics, concurrency, progress bars, CSV/JSON
export — works unchanged.

## Quick Start

```bash
# 1. Install eval dependencies (not in DlightRAG's runtime deps)
uv pip install ragas datasets langchain-openai

# 2. Point at a running DlightRAG
uv run python scripts/ragas_eval.py --api http://localhost:8100
```

If the DlightRAG API has auth enabled:

```bash
DLIGHTRAG_API_TOKEN="your-bearer-token" \
  uv run python scripts/ragas_eval.py --api https://your-server.example.com
```

## Metrics

RAGAS computes four scores per test case (each 0–1):

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer factually grounded in the retrieved context? |
| **AnswerRelevancy** | Does the answer actually address the question? |
| **ContextRecall** | How much of the ground-truth information was retrieved? |
| **ContextPrecision** | Is the retrieved context clean, or full of noise? |

The **RAGAS Score** is the unweighted average of the four metrics.

## Test Dataset Format

A JSON file with a `test_cases` array:

```json
{
  "test_cases": [
    {
      "question": "What is the cancellation policy?",
      "ground_truth": "Reservations cancelled within 24 hours receive a full refund. After 24 hours only the deposit is refunded.",
      "project": "customer-faq"
    },
    {
      "question": "Who is the CTO of the company?",
      "ground_truth": "The CTO is Dr. Sarah Chen, appointed in March 2024."
    }
  ]
}
```

| Field | Required | Purpose |
|---|---|---|
| `question` | yes | The query sent to DlightRAG |
| `ground_truth` | yes | The expected answer; used for recall/relevance scoring |
| `project` | no | Optional grouping label shown in results |

Place the dataset anywhere and pass `--dataset`:

```bash
uv run python scripts/ragas_eval.py --api http://localhost:8100 --dataset my_tests.json
```

The stub dataset bundled with LightRAG (`lightrag/evaluation/sample_dataset.json`)
is used when no `--dataset` is given. It expects documents that were bundled
with LightRAG's sample corpus — swap in your own dataset after ingesting your
actual documents.

## How the Adapter Works

LightRAG's `RAGEvaluator` calls `POST /query` on a LightRAG API server.
DlightRAG uses `POST /api/answer` with a different request and response shape.

`DlightRAGAdapterEvaluator` inherits the entire evaluator and overrides **one
method** — `generate_rag_response()`. It translates:

```
DlightRAG /api/answer response          →  LightRAG RAGEvaluator format
─────────────────────────────────────       ─────────────────────────────
{                                           {
  "answer": "...",                            "answer": "...",
  "contexts": {                               "contexts": [
    "chunks": [                                 "chunk text 1",
      {"content": "chunk text 1"},              "chunk text 2",
      {"content": "chunk text 2"},            ]
    ]                                       }
  }
}
```

Everything else — the RAGAS `evaluate()` call, the two-stage concurrency
pipeline (RAG semaphore → RAGAS semaphore), tqdm progress bars, CSV/JSON
export, benchmark statistics, and console summary table — runs unmodified
from LightRAG.

## Configuration

### Zero-config default

When ``EVAL_LLM_BINDING_API_KEY`` is **not** set, the adapter auto-resolves
eval credentials from DlightRAG's own config:

| Eval setting | Auto-resolved from |
|---|---|
| ``EVAL_LLM_BINDING_API_KEY`` | ``config.llm.roles.query.api_key`` → ``config.llm.default.api_key`` |
| ``EVAL_LLM_MODEL`` | ``config.llm.roles.query.model`` → ``config.llm.default.model`` |
| ``EVAL_LLM_BINDING_HOST`` | ``config.llm.roles.query.base_url`` → ``config.llm.default.base_url`` |
| ``EVAL_EMBEDDING_BINDING_API_KEY`` | ``EVAL_LLM_BINDING_API_KEY`` → DlightRAG embedding key (if OpenAI-compatible provider) |
| ``EVAL_EMBEDDING_BINDING_HOST`` | ``EVAL_LLM_BINDING_HOST`` → DlightRAG embedding base_url (if OpenAI-compatible) |

This means **no extra ``.env`` entries are needed** when DlightRAG already
has a working LLM config (OpenAI, OpenRouter, or any OpenAI-compatible endpoint).
Native-SDK-only providers (Anthropic, Gemini) need an explicit
``EVAL_LLM_BINDING_API_KEY`` because RAGAS requires an OpenAI-compatible API.

### Explicit overrides

All auto-resolved values can be overridden:

### Concurrency and Tuning

```bash
# How many test cases retrieve in parallel (DlightRAG calls)
# Default 10. Set higher if your DlightRAG can handle it.
export EVAL_QUERY_TOP_K=10

# How many RAGAS evaluations run in parallel (RAGAS is LLM-heavy)
# Default 2. Increase for faster runs if you have high rate limits.
export EVAL_MAX_CONCURRENT=2
```

### DlightRAG Connection

```bash
# Base URL — set once via env or CLI flag
export DLIGHTRAG_API_URL="http://localhost:8100"

# Bearer token — only when auth_mode is 'simple' or 'jwt'
export DLIGHTRAG_API_TOKEN="..."
```

## Output

Results are written to `./ragas_eval_results/` (override with `--output-dir`):

```
ragas_eval_results/
├── results_20260610_143052.csv   # Per-question scores
└── results_20260610_143052.json  # Full results with details
```

Console output shows a per-question table and a summary:

```
===================================================================================================================
📊 EVALUATION RESULTS SUMMARY
===================================================================================================================
#    | Question                                             | Faith  | AnswRel | CtxRec | CtxPrec | RAGAS  | Status
-------------------------------------------------------------------------------------------------------------------
1    | What is the cancellation policy?                     | 0.8523 | 0.9100  | 0.7800 | 0.9200  | 0.8656 | ✓
2    | Who is the CTO of the company?                       | 0.9400 | 0.8700  | 0.6500 | 0.8800  | 0.8350 | ✓
===================================================================================================================

======================================================================
📈 BENCHMARK RESULTS (Average)
======================================================================
Average Faithfulness:      0.8962
Average Answer Relevance:  0.8900
Average Context Recall:    0.7150
Average Context Precision: 0.9000
Average RAGAS Score:       0.8503
----------------------------------------------------------------------
Min RAGAS Score:           0.8350
Max RAGAS Score:           0.8656
```

## CI Integration

Add an evaluation gate to your CI pipeline:

```yaml
# .github/workflows/eval.yml
evaluation:
  runs-on: ubuntu-latest
  services:
    postgres:
      image: dlightrag-postgres:pg18
      # ... PG config ...
  steps:
    - uses: actions/checkout@v6
    - run: docker compose up -d
    - run: uv pip install ragas datasets langchain-openai
    - run: uv run python scripts/ragas_eval.py --api http://localhost:8100 --dataset tests/eval/regression.json
      env:
        DLIGHTRAG_API_URL: http://localhost:8100
        EVAL_LLM_BINDING_API_KEY: ${{ secrets.EVAL_LLM_API_KEY }}
```

## Architecture Note

```
┌────────────────────────────────────────────────────┐
│  scripts/ragas_eval.py                             │
│                                                    │
│  DlightRAGAdapterEvaluator(RAGEvaluator)           │
│    └─ generate_rag_response()  ← OVERRIDE          │
│         POST /api/answer → translate format        │
│                                                    │
│  Everything else inherited from RAGEvaluator:      │
│    • RAGAS evaluate() with 4 metrics               │
│    • Two-stage semaphore pipeline                  │
│    • tqdm progress bars                            │
│    • CSV/JSON export                               │
│    • Benchmark stats + console table               │
└────────────────────────────────────────────────────┘
         │                              ▲
         │ POST /api/answer             │ RAGAS calls eval LLM
         ▼                              │
   DlightRAG API                  OpenAI / custom endpoint
```

This is a thin format-adapter — all evaluation logic lives in LightRAG's
`lightrag.evaluation` module. When LightRAG updates `RAGEvaluator` (new
metrics, better concurrency, bug fixes), DlightRAG gets the improvements
without code changes.

## Troubleshooting

**"Cannot connect to DlightRAG API"**
: Make sure `dlightrag-api` is running. Check `docker compose ps` or
`curl http://localhost:8100/health`.

**All contexts are empty**
: The test questions may not match ingested documents. Verify documents
are ingested (`GET /files`) and that `top_k` is reasonable.

**RAGAS scores are all low or NaN**
: Check that the eval LLM API key is set (`EVAL_LLM_BINDING_API_KEY` or
`OPENAI_API_KEY`). NaN scores often mean the eval LLM call failed silently.

**"ImportError: ragas datasets not installed"**
: Run `uv pip install ragas datasets langchain-openai`. These are eval-only
dependencies, intentionally separate from DlightRAG's runtime.
