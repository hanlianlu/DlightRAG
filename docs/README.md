# Documentation

Where to read, by what you are doing. The product's front page is the
[root README](../README.md); this page is the map for the `docs/` tree.

## Reading Paths

| You are | Read, in order |
|---|---|
| Deploying for the first time | [README Quick Start](../README.md#quick-start) → [Configuration](configuration.md) → [Deployment and storage](architecture.md#deployment-and-storage) |
| Operating a deployment | [Operations](operations.md) → [PostgreSQL](postgresql.md) → [Observability](observability.md) → [Security](security.md) |
| Integrating over REST, MCP, or the in-process Python API | [Interfaces](interfaces.md) → [Retrieval and Answer](retrieval-answer.md) → [Run runtime](run-runtime.md) |
| Understanding the design | [Architecture](architecture.md) → [Domain language](domain-language.md) → [Decision records](adr/README.md) |
| Evaluating the built-in Skills and the Session Memory Pass | [Skills routing eval](skills-routing-eval.md) · [Memory pass eval](memory-pass-eval.md) |
| Contributing | [README Development](../README.md#development) → [Decision records](adr/README.md) |

## Ownership

One topic, one owner: the [root README's table](../README.md#documentation) is the
authority on which document owns what, and most documents name their own scope in the
opening lines.

## Decision Records

[Decision records](adr/README.md) indexes every decision with the status its own file
states. The index is generated: `uv run python scripts/adr_index.py` after adding or
editing one, and `make platform-check` fails when it is stale. Decision records explain
why the product is shaped this way; they are not required reading to operate it.

## Trust and privacy

[Data map](trust/data-map.md), [retention](trust/retention.md),
[subprocessors](trust/subprocessors.md), and the
[vulnerability disclosure](trust/vulnerability-disclosure.md) served at
`/.well-known/security.txt`.

## Diagrams

[System](diagrams/system.svg) shows DlightRAG's service processes, the shared
state they use, and the neighbours a deployment reaches;
[architecture](diagrams/architecture.svg) shows the Python layers, the composition
root, and the separately distributed packages. The Web frontend's own layers are
in [Architecture](architecture.md#web-frontend-ownership).

