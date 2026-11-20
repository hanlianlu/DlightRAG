# Subprocessors and Egress

Every path by which a deployment's data leaves the process, and every third
party that may process it. This is the annex a DPA attaches: for the B2B line we
are a processor and each row is a subprocessor the customer must be told about;
for the B2C line we are the controller and each row is a recipient in the
privacy policy.

Rows marked **confirm** are not decided in this repository — hosting, billing,
support and email are deployment decisions the business owns. They are listed
because a DPA that omits them is wrong, and the list is only useful once each
has a value.

## Third parties by purpose

| Vendor | Purpose | Data it receives | Region | Default | Config |
|---|---|---|---|---|---|
| OpenRouter | Chat model (default role) | The Run's prompts and tool results, bounded by the model's context | US | On | `models.chat.default.base_url` |
| DeepSeek | `extract` / `keyword` / `query` / `vlm` roles | Conversation text for extraction, planning, keyword and image work | **China** (`api.deepseek.com`) | On in the shipped config | `models.chat.roles.*` |
| DeepSeek | Web search and fetch | The search query, or the URL to fetch | **China** (`api.deepseek.com`) | Only when a key is configured | `answer.web_sources.deepseek` |
| Voyage AI | Embeddings and rerank | Document chunks, memory queries, retrieval passages | US | On | `models.embedding`, `models.rerank` |
| Exa | Web search | The search query | US | Only when a key is configured | `answer.web_sources.exa` |
| Tavily | Web search | The search query | US | Only when a key is configured | `answer.web_sources.tavily` |
| MinerU | Document parsing | Uploaded documents | Self-hosted sidecar by default; the vendor's cloud only if configured | Self-hosted | `corpus.sidecars.mineru` |
| Docling | Document parsing (alternative) | Uploaded documents | Self-hosted container | Off | `corpus.sidecars.docling` |
| Object storage (S3-compatible) | Agent Mailbox objects | Mail bodies the Agent sends and receives | Deployment's bucket region | Off unless a bucket is named | `answer.agent.mailbox.bucket` |
| Cloud hosting and managed PostgreSQL | Compute, storage, backups | Everything above, at rest | Deployment region | **confirm** | deployment |
| Billing | Subscription and invoices | Name, contact, payment method — not product content | **confirm** | **confirm** | — |
| Support desk and email delivery | Support tickets, transactional mail | Whatever the customer writes to support | **confirm** | **confirm** | — |
| Error and uptime monitoring | Service health | Must stay free of product content; if added, it becomes a processor | **confirm** | **confirm** | — |

## Ours, not a subprocessor

- **ClickStack (ClickHouse, HyperDX, collector)** — the deployment's own
  telemetry store, operated by the same deployment. It receives no data from
  outside the deployment's own processes.
- **The Agent Browser pool and its egress proxy** — our containers, on internal
  networks, with the proxy as the only way out.
- **Pre-baked artefacts**: the tiktoken encoding, the Public Suffix List snapshot
  and the model catalogue ship inside the image and are never fetched at runtime,
  so they are not transfers.

## Egress the owner or customer directs

Two paths carry data to third parties **we** do not choose, and the policy must
say so rather than list them as subprocessors:

- **Personal Connections** (`personal-mcp-connections.md`): the owner's own
  MCP/OAuth servers. Their operator is the owner's or customer's counterparty.
- **Agent Browser navigation and Agent Accounts** (ADR 0032, 0034): the sites an
  Agent visits or registers on, including mail the Agent sends through the
  configured mailbox.

A customer on the B2B line can also point the model, search and parser settings
at providers of its own; those become the customer's subprocessors, and the DPA
should say which side configures what.

## Regions

The three deployments are separate: EU, APAC and China, each with its own
database, telemetry store, keys and provider set. Data does not move between
them.

- **EU**: the DeepSeek rows are transfers to China and need an adequacy decision
  or SCCs plus a transfer impact assessment — or, better, a per-region provider
  set that does not route EU personal data to it. The same question applies to
  any US provider. A DPIA is likely needed for the memory and personalisation
  features.
- **APAC**: per-country rules (Singapore PDPA, Japan APPI, Korea PIPA, Australia
  Privacy Act) differ on consent basis and cross-border transfer; each
  deployment states its own.
- **China**: PIPL wants a separate consent for processing and a lawful
  cross-border mechanism (CAC standard contract or security assessment) before
  any data leaves; a China deployment that must not transfer is a provider-set
  decision, not an afterthought.
- **UK**: none today. If a UK service line opens, the EU SCC set needs the UK
  addendum; nothing else changes.

## Keeping this list true

A new provider becomes a DPA annex change: the B2B line owes customers notice
before it starts processing (30 days is the common commitment), and the B2C line
owes a policy update. The list is generated from the same configuration the
deployment runs, so the audit is: read this file, read the deployment's
configuration, and compare the two.
