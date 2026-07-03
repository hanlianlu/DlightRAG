# Documentation Information Architecture Design

Status: approved for implementation planning.
Date: 2026-07-03.

## Context

The current README is doing too many jobs at once. It acts as product overview,
local setup guide, cloud deployment guide, configuration reference, API index,
SDK example, operations guide, and development guide. That makes important
concepts appear out of order: architecture arrives after setup and usage, while
JWT/JWKS/access control appears abruptly inside configuration.

The documentation set should keep README as the front door and move deep
reference material into focused docs. Reference docs should be materially
exclusive to each other: each durable fact should have one owner, and other
pages should link to that owner instead of restating the same substance.

## Approved Direction

README becomes a product entry point. `docs/` becomes the reference library.

This is intentionally not a full documentation-site redesign. The goal is to
fix misplaced material, reduce README weight, and give security and interface
contracts clear homes without adding unnecessary structure.

## README Structure

Target README order:

1. Summary
2. Architecture At A Glance
3. Deployment Topologies
4. Quick Start
5. Use DlightRAG
6. Core Concepts
7. Security
8. Operations And Development
9. Documentation Map

README should keep:

- the product positioning and runtime ownership model
- the architecture diagram plus a short explanation
- the minimum local setup path
- concise usage examples for Web, REST, Python SDK, and MCP
- short explanations of workspaces, ingestion sources, metadata, retrieval,
  answers, citations, and security
- links to deeper references

README should not keep:

- long configuration tables
- exhaustive API payload fields
- PostgreSQL tuning detail
- detailed RAGAS instructions
- detailed retrieval internals
- long auth/JWT/JWKS/access-control explanations

## Docs Ownership

Target public reading surface:

- root `README.md`
- eight public reference markdown docs under `docs/`

`docs/architecture.svg` and `docs/architecture.drawio` are architecture
assets, not standalone docs. `docs/designs/` contains planning records, not
public product documentation.

Keep these reference responsibilities:

| File | Responsibility |
|---|---|
| `docs/interface-contracts.md` | SDK, REST, MCP, and Web-facing request/response contracts |
| `docs/security.md` | Authentication, JWT/JWKS, IdP boundary, and access control |
| `docs/config-reference.md` | Typed configuration and advanced overrides |
| `docs/PG.md` | PostgreSQL requirements, tuning, and LightRAG storage notes |
| `docs/operations.md` | Maintenance commands and operational safety notes |
| `docs/ragas-evaluation.md` | Evaluation setup and workflow |
| `docs/retrieval_answer_mechanism.md` | Retrieval, fusion, filtering, multimodal query, and answer internals |
| `docs/module-layers.md` | Code organization and import boundary reference |

Rename `docs/response-schema.md` to `docs/interface-contracts.md` because it
covers SDK, REST, MCP, and Web-facing contracts, not only response schemas.

Add `docs/security.md` because auth and authorization are no longer a small
configuration footnote. It owns:

- `auth_mode: none`, `simple`, and `jwt`
- static JWT keys versus JWKS/OIDC issuers
- OAuth/IdP boundary: DlightRAG verifies tokens but does not issue them
- access-control modes
- JWT claim to workspace/action mapping
- deployment recommendations for local, internal, and enterprise setups

Do not add separate docs for metadata, BM25, Langfuse, source retention,
semantic highlights, or parser routing now. They fit inside the existing
owners:

- metadata/source retention: `docs/interface-contracts.md` for call behavior,
  `docs/config-reference.md` for config fields
- BM25 and semantic highlights: `docs/retrieval_answer_mechanism.md` for
  runtime behavior, `docs/config-reference.md` for config fields
- Langfuse: `docs/config-reference.md` for settings, README for the short
  no-op/enablement summary
- parser routing: README for topology, `docs/config-reference.md` for settings

## New Doc Test

Add a new public doc only when all are true:

- it has a distinct audience or operating mode
- it owns facts that do not naturally belong to an existing owner doc
- keeping it inside an existing doc would make that doc harder to scan

Otherwise, place the content in the existing owner doc and link to it.

## De-Duplication Rule

Each topic gets one source of truth:

- README owns orientation and short examples only.
- `docs/security.md` owns auth, JWT/JWKS, IdP boundaries, and access control.
- `docs/interface-contracts.md` owns SDK/REST/MCP/Web request and response
  contracts.
- `docs/config-reference.md` owns config fields and defaults.
- `docs/retrieval_answer_mechanism.md` owns retrieval and answer internals.
- `docs/PG.md` owns PostgreSQL deployment and tuning details.
- `docs/operations.md` owns maintenance commands.
- `docs/module-layers.md` owns code layering.

Other pages may mention a topic only to orient the reader and link to its
owner. They should not copy full tables, payload definitions, or long setup
procedures from another owner page.

## Non-Goals

- Do not build a documentation site.
- Do not introduce a larger docs taxonomy unless the existing files no longer
  fit.
- Do not add `docs/README.md`; the root README documentation map is enough.
- Do not duplicate full reference tables in README.
- Do not duplicate the same substantive reference content across docs.
- Do not preserve stale names just for compatibility; update links instead.

## Acceptance Checks

- README architecture appears immediately after the summary and before quick
  start.
- README can be read as a coherent product entry point without needing to skim
  unrelated reference detail.
- JWT/JWKS/access control is introduced under a security model, not as a random
  configuration subsection.
- Public product documentation is limited to README plus the eight owner docs
  listed above.
- Interface contracts are documented in `docs/interface-contracts.md`.
- Each major topic has one source-of-truth page, with cross-links instead of
  duplicated substance.
- All links to the renamed contract doc are updated.
- Existing architecture diagram references remain accurate.
- No ignored `docs/superpowers/` artifacts are committed.
