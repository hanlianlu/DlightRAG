# Browser Wire Validation and the api-Layer Boundary

The browser consumes server JSON through one validated edge: api modules own valibot schemas that are simultaneously the wire type, the runtime check, and the single place Wire Format is translated.

## Status

Accepted and implemented; the wire contract documented here is the current one.

## Context

Before this change REST responses were trusted by assertion: six api modules ended with `return await response.json() as T`, and only the answer-submission adapter — the one module that must survive hostile responses — narrowed payloads at runtime. The streaming side told a different story: SSE events pass through a hand-rolled parser into a whitelist discriminated union with cursor bookkeeping, so one data path was validated and the other was not. The asymmetry also showed in naming: the workspace store translated snake_case wire fields into camelCase domain objects, while chat features consumed raw `conversation_id`/`byte_size` fields straight from the wire. The wire was typed twice by hand — as interface declarations (~130 lines in api/conversations.ts) and as whatever the projection layer happened to read — and validated zero times.

## Decision

Adopt valibot as the only schema library in the browser. Every REST api module declares its payload schemas colocated with its client functions. A schema is the single source of truth: the TypeScript domain type is inferred from it, `parse` validates every response at the edge, and a transform step renames wire fields to their domain spelling exactly once. Wire Format stops at the api layer; nothing in stores/ or ui/ reads a snake_case server field.

The SSE path keeps its hand-rolled interpretation. Its events already cross a whitelist union, the transport owns resume semantics and cursor bookkeeping, and folding streaming events through a schema validator would duplicate the interpretation layer for no added safety.

## Why this shape

A schema is one declaration serving three consumers — types, runtime validation, and documentation — where the status quo paid twice (interfaces by hand, trust by cast) and validated nothing. Translating field names inside the schema, rather than ad hoc in stores, gives the boundary one owner and deletes the second naming convention instead of documenting it. Valibot over zod for the smallest tree-shaken cost in a bundle that budgets its dependencies carefully; over hand-written guards to avoid twelve endpoints of boilerplate that would drift.

## Rejected alternatives

- **Hand-written narrowing guards.** Zero dependencies, but per-endpoint boilerplate with no type inference and guaranteed drift.
- **Parse without renaming.** Smallest diff, but keeps two naming conventions alive and leaks wire spelling into ui/.
- **Status-quo casts.** The reason this decision exists.
- **zod.** Same role, larger bundle cost.
- **Validate SSE events through schemas too.** Duplicates the interpretation layer; the whitelist union already rejects unknown event types.
- **Ask the server for camelCase.** Changes the public REST contract for a client-side convenience; wire spelling is server-owned.

## Consequences

valibot is the browser schema library (alongside lit, lit-localize, dompurify, mermaid, and xstate). ui call sites that consumed raw wire fields change once to domain names. New endpoints must declare a schema before their client function exists — the api-layer review question becomes "where is the schema". Schemas are hand-authored, not generated; if the server later publishes a machine-readable contract, they can be derived instead of written.

Failures cross the same edge. A refusal answering the general `{detail, error_type, error_kind?}` envelope, or a response that fails its schema, becomes one `ApiError` in `api/wire.ts`: status, the server's public reason, its type, and its kind — never copy of its own. ui code chooses localized text from those fields; a type or reason the body does not name stays null rather than being inferred from the status. One status is read the same everywhere: every reader of a refused response notes its status in `api/wire.ts`, and a 401 from any route signs the whole page out, so the shell asks the reader to sign in again and no Feature handles a 401 of its own. The browser Answer commands (`{kind, message, error_kind?}`) and the Connections module (`{kind, message}`) answer different envelopes and keep their own typed errors; the Connections routes' other refusals, such as authorization and request validation, answer the general envelope and become `ApiError`.

## Scope

This decision does not freeze REST paths or response shapes. When the server contract changes, the owning api schema, inferred domain type, transform, and callers change together; server-owned snake_case still stops at that boundary. SSE keeps its separate whitelist interpretation and durable cursor/reconnect behavior unless a later decision replaces that boundary explicitly.
