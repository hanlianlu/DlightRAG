# Application, Engine, and Adapters

DlightRAG's first-look source tree mixes capability owners, technical layers, transports, and concrete implementations. The accepted target is three visible code zones plus one private composition factory: inbound adapters call Application; Application owns product use cases; Engine owns execution; concrete adapters implement Application or Engine interfaces. Implementation is pending.

## Status

Accepted. Implementation pending. Canonical architecture documents and diagrams continue to describe the current tree until each reorganization milestone lands.

## Context

The current root mixes four taxonomies at once: product capabilities (`answer`, `rag`, `access`), architectural roles (`application`, `services`, `runtime`, `adapters`), transports (`api`, `web`, `mcp`, `sdk`), and a peer `services` package that already belongs to Application. Two folders named `agent` exist for legitimate reasons — a product-neutral kernel and an Answer host — but the names hide that distinction. Inbound HTTP and MCP currently reach through Application, services, and deep Engine types. The documented layering needs nine levels to explain one request.

Pi coding-agent looks clearer because it exposes one product facade (`AgentSession`), one creation function (`createAgentSession`), and modes that call that facade directly. Pi is not smaller internally, and it is not a durable multi-capability service. The lesson worth copying is the short visible path, not a single `core` dump or local JSONL persistence.

DlightRAG still has essential complexity that Pi does not: concurrent durable runs, PostgreSQL leases and fencing, corpus ingestion and retrieval, Fast and Research, REST plus browser plus MCP, Access, Memory, conversations, artifacts, and crash recovery. The reorganization must keep those seams. It must remove accidental hops, duplicate public Python surfaces, and folders that pretend to be owners.

## Decision

Organize the installable product as three visible zones:

1. **Application** — what the product does: configuration, access, lifecycle, health, Answer Runs, Corpus Administration, Retrieval, Memory integration, and Web Conversations.
2. **Engine** — how the product executes: AI, Agent, Runtime, RAG, and Answer as sibling owners under Engine. Ownership is parallel; the dependency DAG is documented and enforced, not nested as fake parent directories.
3. **Adapters** — concrete edge mechanisms: HTTP, MCP, PostgreSQL, Observability, and Maintenance.

The request path is:

```text
HTTP / MCP / Maintenance  ->  Application  ->  Engine
```

HTTP and MCP are protocol adapters: they translate external calls into Application use cases. Maintenance is a command-line adapter: rebuild BM25 and rebuild VDB are Corpus Administration operations, so those commands call Application rather than Engine or PostgreSQL directly.

Outbound adapters implement interfaces owned by Application or Engine. They do not sit on the inbound call path. PostgreSQL may implement Application conversation and corpus ports as well as Engine session and run ports.

Composition exists, but it is not a public zone. A private root composition module wires adapters into Application. The root facade exports `Application`, `DlightragConfig`, `create_application`, and `__version__`. `create_application` asynchronously returns a started Application. Callers close it. Tests may inject in-memory implementations through Application's constructor.

Inbound adapters may import only Application facades and Application-owned request, result, and error contracts. They must not deep-import Engine. Application owns the caller-facing Answer, Retrieval, Corpus Administration, Memory, and Web Conversation contracts, including errors transports must handle. Engine identities that leak today are wrapped or re-homed at the Application facade; they are not imported by HTTP, MCP, or Maintenance. Engine must not import Application or Adapters. Engine children keep today's allowed direction:

- AI depends on no other product modules.
- Agent may depend on AI.
- Runtime may depend on Agent, not Answer or RAG.
- RAG may depend on AI.
- Answer may depend on AI, Agent, RAG, and Runtime.

Profile Memory remains the independent `dlightrag-memory` distribution. Application Memory is only the product capability gate over that package.

The public Python HTTP SDK is retired. Application is the only in-process Python interface. REST is the only remote public interface. The existing async HTTP client moves inside the HTTP adapter for CLI and evaluation reuse. The sync client is deleted.

## Why this shape

A short visible path is the design goal. Extra directories that only classify modules add hops without hiding complexity. Extra public clients that only wrap REST add a third product surface without a third capability.

Application cannot both be the use-case facade and import PostgreSQL. Adapters cannot each assemble the process graph. A private composition function is the remaining role: it runs once at process start, then drops out of the request path.

Engine children stay siblings because their dependencies form a DAG, not a tree. Nesting Runtime under Agent, RAG under Answer, or everything under Answer would lie about ownership. Adding Foundation/Capability/Coordination/Product folders would lengthen paths while leaving each layer with one or two real modules.

HTTP and the browser share one server lifetime and one SSE/cursor implementation. They belong together as one HTTP adapter, not as two first-class packages that import each other. MCP stays beside HTTP because it is a different protocol with its own process entry. The Python remote client is not a third inbound protocol; it is HTTP-adapter machinery.

## Rejected alternatives

- **Keep many first-class owners.** Filesystem peers would continue to look like equal architecture layers.
- **Foundation / Core / Product / Adapters / Interfaces directories.** Classification layers without behavior.
- **A public Bootstrap package.** Makes startup wiring look like a product capability.
- **Let Application compose concrete adapters.** Reverses the facade: Application would depend on PostgreSQL and HTTP.
- **Let each inbound adapter create PostgreSQL and engines.** Duplicates composition and couples transports to storage.
- **Merge the generic Agent kernel into Answer.** Runtime, PostgreSQL, and services already consume Agent types; that merge inverts an enforced seam.
- **Split RAG into top-level Corpus and Retrieval.** Both share one LightRAG workspace runtime and store bundle.
- **A Pi-shaped `core` bucket.** Pi's own core is tens of thousands of lines; DlightRAG would hide owners, not reveal them.
- **Vertical feature slices with per-feature Postgres.** Answer already consumes Knowledge and Memory; feature-local storage would either duplicate schema or create hidden cross-feature adapter imports.
- **Keep a public Python SDK.** It is a REST wrapper, not a distinct capability. Application already is the in-process API.
- **Keep or expand a sync HTTP client.** No production caller requires it.
- **Compatibility shims for old Python paths.** Clean break is accepted; dual trees are the failure mode this work exists to remove.
- **Rename operator config keys, REST paths, or PostgreSQL tables to match Python directories.** Those are different interfaces from source layout.
- **Line-count gates.** Large files can be deep implementations; small files can be pass-throughs.

## Consequences

Implementers must move code by zone and milestone, not by file size. Canonical architecture, interfaces, and SVG diagrams stay current-state until the matching milestone. After the last milestone, the installable tree must contain only Application, Engine, and Adapters as visible product zones; `dlightrag-memory` remains a separate distribution.

The first-look story becomes one sentence: adapters call Application, Application calls Engine, composition is private. Readers should not need a nine-layer legend to find the request path.

## Invariants this decision does not change

Product behavior, REST paths, configuration keys, environment variables, PostgreSQL object names, and persistence semantics stay the same. Fast and Research remain one Answer product with one Agent Session tree. The generic Agent kernel stays product-neutral. RAG remains one workspace runtime with internal corpus and retrieval owners. Local filesystem tools and in-memory session repositories stay inside Agent as internal adapters, not top-level product adapters.
