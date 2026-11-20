# Retention Schedule

Every clock a DlightRAG deployment runs, what it bounds, who enforces it, and
the evidence. Numbers are the shipped defaults; a deployment that changes one
changes a promise, so the DPA states the configured value and not this file.

Two clocks are configuration and the rest are constants: the Run clock
(`runtime.run_retention_days`, default 365) and the telemetry TTL
(`HYPERDX_OTEL_EXPORTER_TABLES_TTL`, shipped 168h). Everything else is hygiene
the code owns, and a deployment that wants a different span changes one number.

## Schedule

| Data | Clock | Enforced by | Evidence |
|---|---|---|---|
| Terminal Answer Runs: row, events, transcript, Agentspace Inventory | `runtime.run_retention_days` (365 default, ≥1) from `finished_at` | Hourly retention pass, bounded `SKIP LOCKED` batches; deletes the rows and cascades | `docs/run-runtime.md` (Retention) |
| Terminal Retrieval and Corpus Mutation Runs | 7 days | Same pass | `docs/run-runtime.md` |
| Agentspace tree (its files, `notes/`, `artifacts/`, `tmp/`, spills) | 7 days after `finished_at`, independently of the Run clock | Agentspace reclaimer sweep, guarded by the Run's row | `_TREE_RETENTION` in `engine/answer/agentspace.py` |
| Agent Session and its entries | While a routed Run references it; the lane history follows the Run clock | Retention pass and the row cascade | `docs/domain-language.md` (Agent Session) |
| Conversation | A turn lives with its Run; a conversation with no turns left is deleted | Hourly sweep | `docs/run-runtime.md` |
| Blobs (Artifacts, attachments, fetched bytes) | No clock of their own: removed when the last referencing row goes | Row cascade | ADR 0010 |
| Active Memory records | No clock: they stand until the owner changes, forgets or clears them | Owner actions, Memory tools, Settings | `packages/memory` |
| Superseded and forgotten Memory records (tombstones) and the operation journal | `superseded_retention_days` = `runtime.run_retention_days` (365), from `updated_at` for records and `created_at` for the journal | Memory janitor: once at startup, then hourly | `_PURGE_RECORDS` in `_storage/pg.py` |
| Corpus Mutation stages | 7 days, or when their Run ends | Retention pass | `docs/run-runtime.md` |
| Telemetry: traces, logs, metrics, session replay | **7 days** (`168h`, ClickHouse TTL on every `otel_*` table and `hyperdx_sessions`) | ClickHouse TTL | `packages/observability/stack/.env`, live table DDL |
| Container logs | The deployment's log driver; not set by this repository | Deployment | — |
| Handover packages (`dlightrag-owner-data export`) | 7 days from export (`_EXPORT_TTL_DAYS`), stated in the package's own manifest and in the request ledger | An operator's timer runs `dlightrag-owner-data sweep-expired`; a package whose deadline cannot be read is left in place and reported | `_EXPORT_TTL_DAYS` in `owner_data.py`; `owner_data_sweep.py` reads the deadline the package states; the request ledger keeps one `sweep` receipt per package judged, saying whether the removal happened |
| Backups | The deployment's policy, and it may not outlast the erasure promise: **30 days** at the shipped promise | Deployment, verified against the promise | — |

## The two numbers a deployment must pin

1. **Telemetry TTL.** The stack's fallback is `168h` (7 days) when
   `HYPERDX_OTEL_EXPORTER_TABLES_TTL` is unset, in `compose.yaml` and in the
   standalone collector alike, and `bootstrap` writes the same `168h` into the
   generated stack environment. (The upstream collector image's own fallback is
   `720h`; the stack states its own and never reaches it.) A DPA that says seven
   days is only true if the environment says so: pin the variable per deployment
   and check the table DDL, which is what an auditor will read.
2. **Backup retention.** An erasure is only complete once the last backup
   holding the data has aged out, and a backup kept for R days keeps a copy
   alive until the request plus R. The promise is therefore one number, not two:
   **the backup horizon may not outlast the erasure promise** — 30 days for
   both. A deployment that wants a longer restore window raises the promise
   with it; it does not keep copies past what the DPA states.

## What each clock does not cover

- **Memory the owner keeps**: retention is not a licence to delete memories that
  are still active. Only the owner, their tools or their Settings change them.
- **Blobs referenced by a surviving Run**: a conversation delete releases only
  what nothing else references; an attachment another turn still uses stays.
- **Corpus documents**: they belong to a Corpus Workspace, not to one person's
  transcript, and they leave when the workspace is deleted — a decision the
  workspace's creator makes, not the retention sweep.
- **Telemetry under a shorter clock than the product**: a Run may live a year
  while its spans live a week. That is deliberate: the trace is for operating
  the service, and the durable record of the Run is PostgreSQL.

## Where this becomes a promise

The DPA and the privacy policy state: the configured Run clock, the seven-day
Agentspace tree, the seven-day telemetry TTL, the deletion of tombstones at the
Run clock, and the 30-day erasure promise the backup horizon sits inside. Each
of those sentences has a row above and a line of code or a policy behind it; a
deployment that changes the configuration owes its customers a notice, and this
file is where the mapping is kept.
