# Data Map

Engineering facts about the data a DlightRAG deployment holds, written so that
counsel can draft a privacy policy, a DPA and a records-of-processing entry from
it. It is not itself a policy, and it states today's behaviour, not an intention.

**Two service lines, two deployments.** The B2C line is a deployment we operate
for consumers, where we are the controller. The B2B line is a deployment we
operate for one customer, where that customer is the controller of its people's
data and we are its processor — the two are separate deployments, separate
databases and separate subprocessor sets, and a commitment made for one is not
inherited by the other. Regions are EU, APAC and China; the deployment holding
the data decides which law applies, and no data moves between them.

## Where the data lives

| Plane | Store | What it holds |
|---|---|---|
| Operational state | PostgreSQL (`dlightrag-postgres:pg18`) | Runs, conversations, agent transcripts, memory, connections, corpus metadata |
| Products | PostgreSQL blobs (`dlightrag_blobs`, `dlightrag_blob_chunks`) | Content-addressed Artifact and attachment bytes, released when the last reference goes (ADR 0010) |
| Corpus and knowledge base | PostgreSQL + pgvector (`lightrag_*`, `dlightrag_doc_metadata`) | Uploaded documents, chunks, vectors, graph, document status |
| Agentspace trees | `answer.agent.workspace_root` volume | One directory per Run: the model-visible Agentspace, its `notes/`, `artifacts/`, `tmp/` |
| Operator files | `deployment.working_dir` volume | Corpus source files, inputs, and the key ring that seals credentials |
| Skills | `skills_root`, `owner_skills_root` volumes | Operator-global Skills and each owner's own published Skills |
| Telemetry | ClickHouse (ClickStack) | Spans, log records, metrics, and session replay |

## PostgreSQL tables by data class

The tables below are the public schema of one deployment.

| Class | Tables | Notes |
|---|---|---|
| Conversation and Run content | `dlightrag_runs`, `dlightrag_run_events`, `dlightrag_answer_run_stages`, `dlightrag_answer_run_routing`, `dlightrag_answer_child_sessions`, `dlightrag_answer_child_operations`, `dlightrag_answer_child_guidance`, `web_conversations`, `web_conversation_turns`, `web_conversation_commands`, `dlightrag_answer_feedback` | The question, the answer, per-step events, child-agent lineage, the reader's feedback, and server-owned command events |
| Agent transcript | `dlightrag_agent_sessions`, `dlightrag_agent_session_entries`, `dlightrag_agent_session_registers`, `dlightrag_answer_session_notes`, `dlightrag_agent_controls` | Every model message, tool call and tool result the Run's agent saw — including Skill documents it loaded, and the Session's own notes |
| Agentspace accounting | `dlightrag_answer_workspace_inventory`, `dlightrag_answer_committed_spills`, `dlightrag_answer_resources`, `dlightrag_answer_evidence` | What the Agentspace held, spilled tool output a `read` resolves, admitted resources, and evidence |
| Products | `dlightrag_answer_run_artifacts`, `dlightrag_answer_artifact_attachments`, `dlightrag_blobs`, `dlightrag_blob_chunks` | Published Artifacts and the bytes behind them |
| Memory | `dlightrag_memory_records`, `dlightrag_memory_operations`, `dlightrag_answer_memory_settings`, `dlightrag_session_memory_progress` | The owner's preferences, facts and experiences, their operation journal, the owner's Memory switch, and the pass's watermark and failure count |
| Corpus and knowledge base | `dlightrag_workspace_meta`, `dlightrag_doc_metadata`, `dlightrag_metadata_field_stats`, `lightrag_doc_full`, `lightrag_doc_chunks`, `lightrag_doc_status`, `lightrag_entity_chunks`, `lightrag_relation_chunks`, `lightrag_full_entities`, `lightrag_full_relations`, `lightrag_graph_nodes`, `lightrag_graph_edges`, `lightrag_llm_cache`, `lightrag_vdb_*`, `dlightrag_promotion_jobs`, `dlightrag_corpus_mutation_windows` | Documents uploaded for retrieval, their parsed text, vectors and graph; `p_*_w_default` are LightRAG's per-workspace tables |
| Connections and credentials | `dlightrag_connection_heads`, `dlightrag_connection_grants`, `dlightrag_connection_generations`, `dlightrag_connection_oauth_flows`, `dlightrag_answer_connection_pins` | An owner's personal MCP/OAuth connections; credentials are sealed with the key ring, never stored in clear |
| Agent accounts and browser | `dlightrag_agent_accounts`, `dlightrag_agent_account_settings`, `dlightrag_agent_browser_leases` | Accounts the Agent registered on the owner's behalf, their sealed credentials, and browser lease rows |
| Schedules | `dlightrag_schedules`, `dlightrag_schedule_fires` | What an owner asked to run on the clock: the prompt, the cadence, and the identity (user id, auth mode and token claims) the Schedule acts as, plus what each slot did |
| Privacy and policy state | `dlightrag_privacy_settings`, `dlightrag_runtime_policies`, `dlightrag_model_catalogue` | The contribution switch (ADR 0041), deployment policy, the model catalogue |
| Operational | `dlightrag_schema_migrations` | Migration bookkeeping; no personal data |
| Requests | `dlightrag_owner_data_requests` | The ledger of exports and erasures: requester, line, region, per-class counts, what stayed in place, outcome and the promise deadline. It names the owner and is the one table an erase does not delete — it is the evidence that the request was served, and it holds no content |

## Filesystem planes

- **Agentspace** (`answer.agent.workspace_root`, default
  `~/.dlightrag/agent_workspaces`): one root per Run, holding the Run's
  Agentspace tree and the spilled tool results beside it. The tree is the Run's
  whole mutable world; its `notes/` carries the Session's memory across Runs.
- **Working directory** (`deployment.working_dir`): corpus source files, operator
  inputs, and the key ring `personal-mcp-connections.md` describes. Reading it
  is refused to Agent processes by the confinement policy (ADR 0024).
- **Skills**: the packaged built-ins inside the image, the operator's global root,
  and each owner's own shard. A Skill document reaches a Run only through
  `load_skill`, and a protected one is masked on the surfaces a user reads.

## Telemetry

`otel_traces`, `otel_logs`, `otel_metrics_*` and `hyperdx_sessions` in the
deployment's own ClickStack. A span carries a pseudonymous `user.id` (the
owner id, hashed) and the bounded review content the observability document
allows: the Run's question (≤2000 characters), its answer, and tool summaries;
credentials, image bytes, DOM text and input values are masked at the seam
before they leave the process. Session replay masks all text and inputs. This
plane **is** personal data, and an erasure has to reach it.

## Special categories

The product does not ask for them, and memory kinds are `preference`, `fact` and
`experience` — but people write what they write: a memory body may name a health
condition or a political view, and a corpus document may be anything. Two
consequences for the policy text: do not claim special-category data is never
processed, and treat memory bodies and corpus documents as content that can hold
it.

## Egress

Every outbound call, and whether it is on by default, is in
[subprocessors.md](subprocessors.md). Agents cannot reach the network except
through the paths that document lists: the browser pool's egress proxy is the
only way out of a browser member, and the working directory is not readable from
an Agent process.

## Rights, and which product path serves them today

| Right | Today | Gap |
|---|---|---|
| Access / portability | `dlightrag-owner-data export`: one package per owner — conversations and turns, Runs and events, the agent transcript, memory and its journal, blobs reassembled by digest, the owner's own Skills, and connection/Agent-account metadata with the sealed columns withheld and named ([ADR 0049](../adr/0049-owner-export-and-erasure.md)) | The package states its own deadline (7 days) and an operator's timer sweeps it; a signed, time-limited download is the next step |
| Erasure | `dlightrag-owner-data erase`: every table that names the owner (discovered from the schema, plus the two conversation tables that use `principal_id`), unreferenced blobs, the owner's Agentspace trees and Skills shard, telemetry rows carrying the owner's `user.id`, and the Corpus Workspaces the requester created (each delivered as one `delete_workspace` Corpus Mutation Run) | Refuses while a Run is queued or running; a covered Workspace is deleted asynchronously by that Run, so until it executes the Workspace row still names its creator and the receipt's `corpora_deletion_runs` is where the promise is checked; `--keep-corpora` keeps one instead and names it, and `--line b2b` requires `--customer-approval`; Workspaces the requester did not create stay, since membership is not enumerable from this table; an orphan Agentspace tree cannot be attributed to one owner and is named rather than deleted |
| Rectification | The owner can edit or forget a memory record; conversations are not editable | No correction path for a stored transcript |
| Restriction / objection | Memory off, contribution off, execution off per deployment | No per-purpose restriction switch for telemetry |
| Consent withdrawal | Contribution switch ends the permission at once (ADR 0041) | None |

## What this document is not

It is not a retention schedule ([retention.md](retention.md)), not a
subprocessor list ([subprocessors.md](subprocessors.md)), and not a policy.
The owner-erasure and owner-export design is
[ADR 0049](../adr/0049-owner-export-and-erasure.md).
