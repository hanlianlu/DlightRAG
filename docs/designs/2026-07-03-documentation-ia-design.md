# Documentation System Design

Status: approved for implementation planning.
Date: 2026-07-04.

## Context

DlightRAG's documentation has grown through feature work. The current README is
accurate in many places, but it is not serving as a clear product entry point.
It mixes product overview, local setup, cloud setup, API examples, SDK usage,
MCP, architecture, configuration, auth, Langfuse, operations, development, and
RAGAS evaluation.

The reference docs have the same problem at a smaller scale. Several files are
valuable but named or structured around their history instead of their reader:
`response-schema.md` is really an interface contract, `module-layers.md` is a
code-architecture reference rather than a product architecture overview,
`PG.md` uses an inconsistent name, and `config-reference.md` carries behavior
explanations that belong to retrieval, security, operations, or architecture.

The redesign should treat the docs as one product system, not as independent
markdown files.

## Documentation Positioning

The documentation has four primary reader paths:

| Reader | Need | Primary path |
|---|---|---|
| New evaluator | Understand what DlightRAG is and run it locally | README |
| SDK/API/MCP consumer | Call the product correctly | README -> `docs/interfaces.md` |
| Enterprise operator | Deploy, secure, configure, and maintain it | README -> security/configuration/PostgreSQL/operations docs |
| Maintainer | Understand boundaries and internals | architecture/retrieval docs plus code |

README is the front door. It should help a reader choose the right next page,
not duplicate those pages.

Reference docs are owner pages. Each durable fact should have one owner. Other
pages can summarize that fact in one or two sentences only when it helps
navigation, then link to the owner.

## README Design

README should be a product entry point with this order:

1. Summary
2. Architecture At A Glance
3. Choose Your Deployment Path
4. Quick Start
5. Use DlightRAG
6. Core Concepts
7. Security Model
8. Operations And Development
9. Documentation Map

README should keep:

- product positioning and runtime ownership
- the architecture diagram plus a short explanation
- local and cloud topology choices
- the shortest successful local setup path
- concise examples for Web, REST, Python SDK, and MCP
- short explanations of workspaces, ingestion sources, metadata, retrieval,
  answers, citations, and security
- links to owner docs for details

README should not keep:

- exhaustive API payload tables
- long configuration field tables
- PostgreSQL tuning detail
- detailed RAGAS workflow
- retrieval internals
- long JWT/JWKS/access-control explanations
- internal code-layer reference detail

## Target Public Docs

Use a small, topic-owned docs set. File names should be lower-kebab case and
reader-oriented.

| Target file | Source of truth for |
|---|---|
| `docs/architecture.md` | Product architecture, runtime ownership, LightRAG/DlightRAG boundary, architecture diagram, and links to code layering |
| `docs/interfaces.md` | SDK, REST, MCP, and Web-facing contracts for ingest, retrieve, answer, jobs, contexts, sources, citations, and multimodal payloads |
| `docs/security.md` | Auth modes, static JWT, JWKS/OIDC issuers, OAuth/IdP boundary, access control, claim mapping, and deployment security posture |
| `docs/configuration.md` | Config precedence, public config boundary, field groups, defaults, and when to change advanced settings |
| `docs/retrieval-answer.md` | Retrieval pipeline, metadata filtering, BM25, query images, rerank, answer packing, citations, and semantic highlights behavior |
| `docs/postgresql.md` | PostgreSQL 18 requirements, extensions, pooling, HNSW, AGE patches, schema migrations, and deployment tuning |
| `docs/operations.md` | Maintenance commands, rebuilds, operational safety, and recovery workflows |
| `docs/evaluation.md` | RAGAS evaluation workflow, dataset format, metrics, output, and CI gate |

Architecture assets such as `docs/architecture.svg` and
`docs/architecture.drawio` remain assets, not standalone docs.
`docs/designs/` remains planning history, not public product documentation.

Do not add a public `docs/README.md`; the root README's Documentation Map is
the index.

## Rename And Merge Plan

The target docs should replace historical names rather than preserve stale
aliases:

| Current file | Target |
|---|---|
| `docs/response-schema.md` | Rename and reshape to `docs/interfaces.md` |
| `docs/config-reference.md` | Rename and reshape to `docs/configuration.md` |
| `docs/retrieval_answer_mechanism.md` | Rename and reshape to `docs/retrieval-answer.md` |
| `docs/PG.md` | Rename and reshape to `docs/postgresql.md` |
| `docs/ragas-evaluation.md` | Rename and reshape to `docs/evaluation.md` |
| `docs/module-layers.md` | Fold the product-level parts into `docs/architecture.md`; keep code-layer detail there only if it remains useful and compact |
| `docs/operations.md` | Keep name; tighten scope if needed |

All links should be updated to target names. Do not leave compatibility shim
docs.

## Page Structure

Each public doc should start with:

1. who the page is for
2. what topic it owns
3. what related topics live elsewhere

Then use this default body order:

1. mental model or decision guide
2. minimal working example
3. complete reference
4. edge cases and operational notes

This keeps pages useful for both skimming and exact reference lookup.

## Content Ownership Rules

Use these source-of-truth boundaries:

- README owns orientation and short examples only.
- `docs/architecture.md` owns runtime boundaries and architecture narrative.
- `docs/interfaces.md` owns request/response shapes and interface behavior.
- `docs/security.md` owns authentication, authorization, and identity-provider
  boundaries.
- `docs/configuration.md` owns config fields, precedence, and defaults.
- `docs/retrieval-answer.md` owns retrieval and answer internals.
- `docs/postgresql.md` owns database requirements and tuning.
- `docs/operations.md` owns maintenance and recovery commands.
- `docs/evaluation.md` owns evaluation workflow.

Do not create separate docs for metadata, BM25, Langfuse, source retention,
semantic highlights, parser routing, or MCP transport unless they grow into a
distinct reader path. For now:

- metadata and source retention belong to interfaces for call behavior and
  configuration for fields/defaults
- BM25 and semantic highlights belong to retrieval-answer for behavior and
  configuration for settings
- Langfuse belongs to configuration for settings and README for the short
  optional/no-op summary
- parser routing belongs to architecture for ownership and configuration for
  settings
- MCP transport belongs to interfaces for contracts and security for exposure
  posture

## Style And Layout

Docs should be concise but not shallow:

- Prefer one clear example over several near-duplicates.
- Prefer tables for exact contracts and defaults.
- Prefer short diagrams or text flows for architecture and pipelines.
- Avoid long curl blocks in README; put detailed REST examples in
  `docs/interfaces.md`.
- Keep SDK examples idiomatic Python, not pseudo-code.
- Keep headings reader-facing, not implementation-facing.
- Keep filenames and headings in title case or lower-kebab consistently.
- Use cross-links for adjacent topics instead of repeating their content.

## New Doc Test

Add a new public doc only when all are true:

- it has a distinct reader path or operating mode
- it owns facts that do not naturally belong to an existing owner doc
- placing the material in an existing doc would make that doc harder to scan

Otherwise, place the content in the existing owner doc and link to it.

## Non-Goals

- Do not build a documentation site.
- Do not expand into a large docs taxonomy.
- Do not duplicate full reference tables in README.
- Do not preserve stale file names as shim docs.
- Do not rewrite working technical content just for different prose if moving
  and tightening it is enough.

## Acceptance Checks

- README architecture appears immediately after the summary and before quick
  start.
- README reads as a coherent product entry point.
- A reader can choose local, SDK/API/MCP, enterprise deployment, or maintainer
  paths from the README without scanning unrelated sections.
- Public docs use the target file names above.
- Each public doc states what it owns and what lives elsewhere.
- JWT/JWKS/access control is introduced under a security model.
- Interface contracts are documented in `docs/interfaces.md`.
- Product architecture is documented in `docs/architecture.md`.
- Historical links to renamed docs are updated.
- No ignored `docs/superpowers/` artifacts are committed.
