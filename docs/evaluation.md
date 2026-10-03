# Evaluation

This page is for teams measuring answer quality with RAGAS. It owns the
evaluation workflow, dataset format, metrics, outputs, and release use.
Runtime retrieval behavior lives in [retrieval-answer.md](retrieval-answer.md);
REST answer contracts live in [interfaces.md](interfaces.md).

DlightRAG reuses LightRAG's built-in [RAGAS](https://docs.ragas.io/) evaluation
framework. The adapter in `scripts/ragas_eval.py` inherits from LightRAG's
`RAGEvaluator` and replaces only how an answer is obtained, so the rest of the
evaluation pipeline — metrics, concurrency, progress bars, CSV/JSON export —
works unchanged.

## Quick Start

```bash
# 1. Install eval dependencies (not in DlightRAG's runtime deps)
uv sync --group eval

# 2. Run with your test dataset — API URL and creds auto-resolve
uv run python scripts/ragas_eval.py --dataset my_questions.json
```

When running from outside the repo or against a remote instance, pass `--api`:

```bash
uv run python scripts/ragas_eval.py --api https://dlightrag.example.com --dataset my_questions.json
```

No-auth and simple-auth API connection settings auto-resolve from DlightRAG's
config. JWT setups must pass an externally issued bearer token with
`--api-key` or `$DLIGHTRAG_API_TOKEN`.

## Metrics

RAGAS computes four scores per test case (each 0–1):

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer factually grounded in the retrieved context? |
| **AnswerRelevancy** | Does the answer actually address the question? |
| **ContextRecall** | How much of the ground-truth information was retrieved? |
| **ContextPrecision** | Is the retrieved context clean, or full of noise? |

The **RAGAS Score** is the unweighted average of the four metrics, ignoring any
that are NaN.

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

Place the dataset anywhere and pass `--dataset`; it is always required, with no
built-in default.

## How the Adapter Works

LightRAG's `RAGEvaluator` reads each answer from a LightRAG server's
`POST /query` response. A DlightRAG answer is a durable Answer Run instead, so
`DlightRAGAdapterEvaluator` overrides **one method**,
`generate_rag_response()`. It submits
`{"query": <question>, "top_k": $EVAL_QUERY_TOP_K}` to `POST /answer`, which
accepts the Run and returns its `run_id`, then follows
`GET /runs/{run_id}/events` to the Run's `done` event, reading
`GET /runs/{run_id}` instead when the stream ends early or has expired. It
translates the succeeded Run's result:

```
Answer Run result                       →  LightRAG RAGEvaluator format
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

Chunks without text are dropped, and a failed or cancelled Run fails its test
case. The result's presentation and provenance fields, such as `parts`,
`evidence_images`, `references`, `sources`, `artifacts`, `artifact_outcome`, and
`trace`, are ignored: RAGAS scores the answer text against textual chunk
contexts.

Everything else — the RAGAS `evaluate()` call, the two-stage concurrency
pipeline (RAG semaphore → RAGAS semaphore), tqdm progress bars, CSV/JSON
export, benchmark statistics, and console summary table — runs unmodified
from LightRAG.

## Configuration

### Zero-config default

When `EVAL_LLM_BINDING_API_KEY` is **not** set and DlightRAG's query role uses
the `openai` provider, the adapter auto-resolves eval credentials from
DlightRAG's own config:

| Eval setting | Auto-resolved from |
|---|---|
| `EVAL_LLM_BINDING_API_KEY` | `models.chat.roles.query.api_key` → `models.chat.default.api_key` |
| `EVAL_LLM_MODEL` | `models.chat.roles.query.model` → `models.chat.default.model` |
| `EVAL_LLM_BINDING_HOST` | `models.chat.roles.query.base_url` → `models.chat.default.base_url` |
| `EVAL_EMBEDDING_BINDING_API_KEY` | `EVAL_LLM_BINDING_API_KEY` → DlightRAG embedding key (`openai` or `openai_compatible` provider) |
| `EVAL_EMBEDDING_BINDING_HOST` | `EVAL_LLM_BINDING_HOST` → DlightRAG embedding `base_url` (`openai` or `openai_compatible` provider) |
| `DLIGHTRAG_API_URL` | `http://<interfaces.api.host>:<interfaces.api.port>` |
| `DLIGHTRAG_API_TOKEN` | `access.api_token` under `simple` auth |

JWT deployments must provide an externally issued bearer token via
`DLIGHTRAG_API_TOKEN`. A query role on the `anthropic` or `gemini` provider
resolves no eval LLM, because LightRAG's RAGAS evaluator uses an
OpenAI-compatible client: set `EVAL_LLM_BINDING_API_KEY` (or `OPENAI_API_KEY`)
and `EVAL_LLM_MODEL`, which otherwise defaults to `gpt-4o-mini`.

### Explicit overrides

All auto-resolved values can be overridden:

```bash
export EVAL_LLM_MODEL=gpt-4o
export EVAL_LLM_BINDING_API_KEY="..."
uv run python scripts/ragas_eval.py --api https://dlightrag.example.com --api-key "..." --dataset my_tests.json
```

### Concurrency and Tuning

```bash
# Retrieval breadth sent to DlightRAG /answer as top_k.
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

Console output shows one row per question plus aggregate, minimum, and maximum
scores.

## Manual release evaluation

RAGAS evaluation runs outside pull-request CI: it needs an operator-selected
dataset, a running DlightRAG, and evaluator model credentials, and its scores
depend on the corpus and evaluator models rather than being deterministic. The
release operator runs `scripts/ragas_eval.py` as above and reviews the results
before a release.

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

**"ImportError: RAGAS dependencies not installed"**
: Run `uv sync --group eval`. Ragas is an eval-only dependency, separate from
DlightRAG's runtime and locked through LightRAG's `evaluation` extra.
