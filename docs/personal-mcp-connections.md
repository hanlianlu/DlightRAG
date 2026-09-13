# Personal MCP Connections

**Status: IMPLEMENTED IN THE CURRENT WORKTREE / FINAL VALIDATION AND REVIEW PENDING**

**Implementation baseline:** `main@be16218297601e2c965d596b762d49358a389e3b` plus the owned, uncommitted implementation. Owner management, all three authentication modes, automatic discovery/refresh, atomic Research binding and effect fencing, retention/GC, and keyring maintenance are implemented. This document records the current contract, not a released-version or full-CI acceptance claim.

This plan is the implementation authority for the accepted target. It is governed by [ADR 0012](adr/0012-personal-connections-and-hot-plug.md) and uses the canonical [Domain Language](domain-language.md). It supersedes the earlier local research proposals, not their historical baseline evidence. The optional gitignored research is not required to understand or implement this tracked design.

## Product contract

- Settings owns the path **Settings → Connections → MCP**.
- Each eligible authenticated user owns, manages, authorizes, enables, and disables only their own Connections.
- JWT `(iss, sub)` owners and the stable local single-user `none` owner are eligible. Shared `simple` authentication has no personal Connections capability.
- Every enabled Connection with a published catalogue is included automatically when that owner accepts a future Research-capable Answer Run, regardless of whether the Answer entered through Web, REST, inbound MCP, or the in-process Application.
- Fast has no MCP tools. The composer has no Tools/MCP control, and a conversation has no Connection selector.
- The first release supports MCP Streamable HTTP only. Arbitrary Web-managed stdio is not accepted.
- Authentication choices are unauthenticated, a write-only personal static bearer, and OAuth through the locked `mcp==2.2.0` SDK.
- The Connection is the authorization unit. Enabling it authorizes all current and future remote tools; the UI never projects the catalogue and offers no per-tool checkbox, allowlist, or later drift approval. The Capability Catalogue is agent-facing: it is what a Run pins, not what a person approves item by item.
- OAuth scope expansion still requires provider consent. It is not implied by Connection enablement.
- There are no per-call write confirmations. Up-front enable consent warns that tools can modify, send, or delete data allowed by the external grant, including in authorized shared external workspaces.
- DlightRAG isolates its private Connection records, credentials, Runs, Conversations, files, and caches by owner. It does not claim that an arbitrary external MCP server enforces the same local user boundary.
- A Connection fault does not immediately fail the whole Research Run. Other tools remain usable; the failed result instructs Research to state which requested part could not be completed.
- A possibly dispatched side-effect call is never retried automatically.

## Scope and non-goals

The implementation includes CRUD, enable/disable, all three authentication choices, automatic catalogue refresh, atomic Run binding, recovery, effect-time revocation, retention/GC, and Settings status together. Independent full validation and review remain release gates.

It does not add a marketplace, public Connection-management REST API, inbound-MCP management tools, arbitrary request headers, MCP resources/prompts/apps, a protocol registry, a universal `PluginManager`, a second `RunRuntime`, or a durable invocation-permit ledger. It does not merge Connections with models, Skills, Profile Memory, or Web resources.

The reusable hot-plug semantics are only **immutable publication → atomic Run pin → effect-time revoke → retention/GC**. Keep those mechanics private inside Connections until a second real consumer proves an extraction seam.

## Ownership mapping

| Domain term | Authority and owner |
|---|---|
| Connection | Owner-scoped logical remote MCP relationship, including label, endpoint, activation state, and current generation; owned by Application Connections |
| Credential Grant | Owner authorization for one remote account/resource audience; canonical lifecycle and safe projection owned by Connections |
| Capability Catalogue | Validated all-or-nothing observation of remote tool definitions; remote data is untrusted, Connections publishes it |
| Connection Generation | Immutable endpoint digest, Grant identity, and catalogue digest used by future acceptance |
| Run Connection Binding | Secret-free owner/Run reference to one generation plus its activation epoch; accepted atomically with the Run |
| Effect Intent / Effect Settlement | Existing Agent Session Runtime recovery authority; Connections does not duplicate it |
| Dispatch gate | Ephemeral result of the final database authorization/fence check; not a durable domain record |
| MCP session | Process-local SDK/HTTP resources for one discovery or foreground call; owned by the MCP adapter |
| Product policy | Endpoint, network, quota, timeout, and secret-handling ceilings; may deny user authority but never create it |
| Settings projection | Redacted owner view and commands; owned by the browser Feature and Web adapter |

## Deep module and neutral host seam

`dlightrag.application.connections` is one deep module. Callers learn owner eligibility, optimistic revision conflicts, safe projections, and the following small interface; discovery, grants, publication, scheduling, gate ordering, and error classification remain inside.

```python
class Connections:
    async def read(self, *, owner_id: str, auth_mode: str) -> ConnectionsView: ...
    async def change(self, *, owner_id: str, auth_mode: str,
                     expected_revision: str, command: ConnectionCommand) -> ConnectionsView: ...
    async def replace_bearer(self, *, owner_id: str, auth_mode: str,
                             connection_id: str, bearer: SecretStr) -> ConnectionsView: ...
    async def begin_authorization(self, *, owner_id: str, auth_mode: str,
                                  connection_id: str) -> AuthorizationStart: ...
    async def bind_research(self, *, owner_id: str,
                            auth_mode: str) -> BoundResearchConnections: ...
    async def restore_research(self, *, bindings: tuple[RunConnectionBinding, ...],
                               claim: ResearchToolClaim) -> tuple[AgentTool, ...]: ...
```

Management results never contain a token, authorization code, client secret, decrypted envelope, or credential reference usable outside the module. `BoundResearchConnections` contains schema-only `AgentTool` definitions and secret-free bindings.

Engine Answer owns neutral `RunConnectionBinding`, `ResearchToolClaim`, and `ResearchConnectionToolResolver` contracts. A claim carries trusted `owner_id`, `run_id`, `worker_id`, the parent Run fence/cancellation callback, and no Web request object. Generic `ToolRuntime` additionally carries its current Agent Session fencing epoch alongside the existing `execution_scope` and `intent_id`, so parent and Child Session dispatch can be fenced without adding owner or MCP identity to Agent Core.

The private composition root injects `Connections.restore_research` behind the Engine-owned resolver interface. Engine imports neither Application nor MCP; Application imports Engine contracts in the existing direction. No request or model argument supplies an owner.

## Exact destination and dependency direction

### New modules

| Path | Responsibility |
|---|---|
| `src/dlightrag/application/connections/__init__.py` | Export the small interface, commands, views, bindings, errors, store port, and `McpClientPort` |
| `src/dlightrag/application/connections/service.py` | Implement eligibility, grant lifecycle, publication, scheduler, binding, restore, dispatch gate, health transitions, and GC eligibility |
| `src/dlightrag/application/connections/credentials.py` | Validate the injected private key ring and implement versioned authenticated encryption/decryption with an established crypto library; no environment reads or custom cryptography |
| `src/dlightrag/adapters/postgres/connections.py` | Implement owner-scoped schema/migrations, encrypted envelopes, OAuth inbox, refresh/discovery leases, CAS/locks, normalized pins, and NOTIFY wakeups |
| `src/dlightrag/adapters/mcp/oauth.py` | Adapt SDK `OAuthClientProvider`, SDK `TokenStorage`, redirect handler, callback handler, and the fenced store operations |
| `src/dlightrag/engine/answer/execution/connection_binding.py` | Define secret-free binding and host-injected resolver/claim contracts only; no MCP or Application import |
| `src/dlightrag/engine/network_admission.py` | Hold DNS/IP/redirect/address-pinning primitives shared by public GET and MCP HTTP clients |
| `src/dlightrag/adapters/http/browser/routes/connections.py` | Project authenticated same-origin management and OAuth callback routes only |
| `frontend/api/connections.ts` | Validate the redacted browser wire and issue same-origin commands |
| `frontend/ui/settings-connections.ts` | Own Settings → Connections → MCP state, intent, async work, focus, warnings, and safe credential forms |

### Existing modules changed by the implementation

| Path | Target change |
|---|---|
| `src/dlightrag/application/answer_runs/service.py` | Resolve owner Connections only for a Research-capable acceptance; put their definitions in `AgentRunPlan`; pass bindings to every acceptor |
| `src/dlightrag/engine/answer/execution/input.py` | Serialize and bound `run_connection_bindings`; reject secret-like fields; never require a binding to remain the current head |
| `src/dlightrag/engine/answer/execution/executor.py` | For resolved Research, restore pinned tools through the injected resolver and trusted `RunSession`; Fast resolves none; remove the static tuple |
| `src/dlightrag/engine/agent/tools/contracts.py` | Add only a provider-neutral Agent Session fencing epoch to `ToolRuntime`; do not add owner, Web, credential, or MCP facts |
| `src/dlightrag/engine/answer/research/runtime.py` | Populate that fencing epoch for parent and Child Session tool effects |
| `src/dlightrag/adapters/postgres/runtime/run_store.py` | Forward bindings through `create_run` into `accept_run`; direct validation/FK-pin insertion belongs inside the transaction owned by `accept_run`. `create_run_in` performs the equivalent work inside its caller-owned Web transaction |
| `src/dlightrag/application/web_conversations/service.py` | Forward bindings through `_WebAnswerAcceptor` rather than creating a Web-only binding path |
| `src/dlightrag/application/web_conversations/models.py` | Extend the atomic Web-turn store contract with bindings |
| `src/dlightrag/adapters/postgres/web/web_conversations.py` | Forward bindings to `create_run_in` inside the existing Conversation/turn/Run transaction |
| `src/dlightrag/adapters/mcp/personal_http.py` | Bounded Streamable-HTTP discover/call behavior over a supplied SDK HTTP client; obsolete `outbound.py` deployment/re-export path removed |
| `src/dlightrag/engine/public_http.py` | Consume `network_admission` without turning its GET helper into MCP transport |
| `src/dlightrag/application/application.py` | Expose/start/stop Connections in dependency order; stop refresh before shutdown and close it after Run workers drain |
| `src/dlightrag/_compose.py` | Construct the private credential cipher from the resolved secret-only settings field; inject it, Postgres, MCP/OAuth, acceptance binder, and execution resolver; remove the config-built global tuple |
| `src/dlightrag/application/config/sections.py` | Replace endpoint declarations with non-secret Connection policy, finite quotas/timeouts, OAuth public callback metadata, and a secret-only excluded `credential_secret_keyring` field in nested Connections settings |
| `src/dlightrag/application/config/loading.py` | Preserve explicit `--env-file` loading and source precedence for the new field; propagate validated secret settings without a second dotenv/environment loader |
| `src/dlightrag/application/config/yaml_source.py` | Reject the credential key-ring field in YAML before source merging, even when a higher-priority secret source would override it |
| `.env.example` | Document the secret-only key-ring variable and safe generation/rotation instructions with placeholders, never working keys |
| `config.yaml` | Remove deployment server/tool declarations and document only non-secret Connection policy defaults |
| `src/dlightrag/adapters/http/browser/routes/__init__.py` | Mount the Web-only routes |
| `frontend/ui/settings.ts` | Add Settings navigation and compose the Connections Feature; do not touch the composer |
| `src/dlightrag/adapters/http/browser/routes/bootstrap.py`, `frontend/api/bootstrap.ts` | Add a required owner-specific `personal_mcp_connections` capability in the server projection and validated frontend wire; bump `contract_version` from 1 to 2 atomically and update bootstrap fixtures/consumers |
| `docs/architecture.md`, `docs/configuration.md`, `docs/security.md` | At implementation time, change current-state text only after the capability ships |

Direction remains `browser → application.connections`, `application → engine contracts`, and concrete `postgres/mcp → application-owned ports`. `run_store.py` receives a purpose-built Postgres-side pin writer; it does not expose a universal transaction or import HTTP/MCP. This follows [ADR 0001](adr/0001-application-engine-adapters-architecture.md) and [ADR 0011](adr/0011-owner-specific-operational-state-adapters.md).

## Durable state

All primary keys, unique keys, lookups, and foreign keys below include `owner_id` where identity crosses a table.

- `dlightrag_connection_heads(owner_id, connection_id, revision, label, enabled, activation_epoch, head_generation, consent_version, tombstoned_at, refresh_due_at, refresh_owner, refresh_epoch, refresh_expires_at, observed_status, last_attempt_at, last_error_kind)`.
- `dlightrag_connection_generations(owner_id, connection_id, generation, endpoint_json, endpoint_digest, grant_id NULL, catalogue_json, catalogue_digest, created_at)`; endpoint JSON rejects userinfo, fragments, credential-bearing query data, and arbitrary headers.
- `dlightrag_connection_grants(owner_id, grant_id, kind, audience_digest, consented_scopes, status, secret_version, encrypted_envelope, key_id, refresh_owner, refresh_epoch, refresh_expires_at, updated_at)`; `kind` is `bearer` or `oauth`. Unauthenticated generations have no Grant.
- `dlightrag_answer_connection_pins(owner_id, run_id, connection_id, generation, activation_epoch, catalogue_digest)` with composite foreign keys to the owner Run and immutable generation and `ON DELETE CASCADE` from Run.
- `dlightrag_connection_oauth_flows(flow_id, owner_id, connection_id, flow_owner, flow_lease_expires_at, state_hash, encrypted_result, expires_at, consumed_at)`; this is a short-lived callback inbox, not a workflow runtime.

Catalogue JSON is allowed because quotas make one generation bounded and immutable. The normalized pin is mandatory because JSON alone cannot stop concurrent GC.

### Executable secret-source seam

The proposed input is `DLIGHTRAG_ANSWER__AGENT__CONNECTIONS__CREDENTIAL_SECRET_KEYRING`, loaded by the existing `load_config(env_file=...)` / `DlightragConfig` source pipeline: process environment or local `.env` supplied by the operator, with orchestrator Secrets injected as environment. It targets `answer.agent.connections.credential_secret_keyring: SecretStr | None` (`exclude=True`, `repr=False`); trusted in-process callers may explicitly supply the same typed secret field. YAML rejects that path before merging. This is a secret carried by typed settings, not non-secret Application Configuration or a Deployment Binding, consistent with ADR 0006; no second loader reads a different `.env`.

The secret value is JSON shaped as `{"active":"<key-id>","keys":{"<key-id>":"<base64url-encoded-32-byte-key>"}}`. Validate nonempty unique IDs, strict key encoding/length, and that `active` exists; validation errors never echo values. `_compose.py` unwraps it only to construct `CredentialCipher` in `application/connections/credentials.py`. Use an established AES-256-GCM implementation with fresh nonces and authenticated owner/Connection/Grant identity; persist only a versioned ciphertext envelope, nonce, and key ID. Do not derive these keys from JWTs, database passwords, or the cursor signing key.

All workers receive the same ring. New encryption uses `active`; reads can use retained IDs. Rotation adds a key, switches `active`, and re-encrypts envelopes with secret-version CAS before an operator removes old keys. Missing/invalid keys must fail closed for credential storage/use, never fall back to plaintext or an ephemeral generated key; an unreadable existing envelope reports deployment misconfiguration rather than prompting the user to overwrite it. No key enters PostgreSQL, settings dumps, UI, logs, or Run input. Deleting grant ciphertext removes live access but does not promise cryptographic erasure of historical backups while retained deployment keys can decrypt them; backup/key retention is an explicit operator responsibility.

## Publication and automatic discovery

Create and credential commands first persist an owner-scoped disabled draft/grant in a short transaction. Network discovery happens after that transaction. A successful complete candidate is published by expected-head-revision CAS; a stale result is discarded and rescheduled.

Discovery uses `tools/list` through `mcp==2.2.0`, follows bounded pagination, validates every tool name/description/input schema, creates deterministic local names from stable Connection identity plus the remote name, and rejects collisions. The entire candidate publishes or none of it does.

Enabled Connections are refreshed automatically by a bounded PostgreSQL-claimed scheduler. `refresh_due_at` provides a finite maximum poll interval; startup/reconnect scans recover missed NOTIFY wakes. A claim transaction records an expiring epoch and ends before network I/O. Publication requires the same claim/head/grant epoch, so a late worker cannot overwrite a newer edit, disable, or credential replacement.

A discovery error leaves the last-good generation published, updates only redacted observed status, and schedules bounded backoff with jitter. Duplicate cursors, malformed schemas, oversized descriptions, too many tools/pages, or an over-budget whole catalogue fail the candidate. Defaults must bound per-owner Connections and enabled tools, per-Connection tools, pages, per-tool schema/description bytes, total catalogue bytes, and discovery concurrency; operators may lower those ceilings.

New remote tools and schema/description changes are automatically admitted for future Runs by publishing a whole new generation. There is no manual probe/restart dependency and no per-tool consent. Existing bindings never change.

## Atomic acceptance and retention

`bind_research` performs no remote I/O. For eligible Research-capable acceptance it reads enabled heads with complete last-good catalogues, constructs schema-only tools, and returns bindings. Simple auth, Fast-only valid mode sets, disabled heads, and incomplete new Connections return no tools.

Direct `PGRunStore.create_run` only forwards to `accept_run` in the baseline (`run_store.py:2481-2534`); it owns no transaction. Put direct validation and pin writes inside the existing `accept_run` transaction. Web `create_run_in` receives the Conversation-owned transaction. Both paths use the same order inside those actual accepting transactions:

1. Resolve idempotent replay as today; an existing Run keeps its original pins.
2. Lock referenced head rows in ascending `(owner_id, connection_id)` order.
3. Lock referenced generation and Grant rows in the same Connection order.
4. Verify owner, exact generation/head snapshot, catalogue digest, enabled state, activation epoch, Grant identity/status, and Connection consent version.
5. Insert the Run, accepted input, routing/Conversation projection, and all normalized pins before commit.

A stale binding aborts without a Run. Answer acceptance performs one bounded, network-free rebind/rebuild retry; repeated churn returns a conflict rather than accepting a mismatched `AgentRunPlan`.

Ordinary endpoint or catalogue publication changes the head only for future Runs. A disable or tombstone increments `activation_epoch`; re-enable uses the newer epoch and cannot revive an old binding. A bearer replacement, different external account, or consented OAuth scope expansion creates a new Grant and generation and retires the old Grant. Ordinary same-scope OAuth refresh stays within the same Grant.

Pins live as long as their retained Run row. GC locks a non-head generation, verifies no normalized pin and no active publication/refresh claim, then deletes it. A concurrent acceptance either pins before GC or observes the missing/stale generation and retries; it never commits a dangling JSON reference. Small secret-free generation metadata follows Run retention. Retired Grant secret ciphertext is removed from the live store independently of metadata retention; backup erasure has the separate limits described above.

## Restore, dispatch, revoke, and cancellation

Recovery decodes the Run bindings and asks the injected resolver for the exact local tool definitions. It never substitutes the current Connection head. `AgentRunPlan` equality remains the final definition check before provider/tool effects. A pin freezes local name, schema, description, and routing metadata only; it cannot freeze remote code, data, availability, or side effects.

Every MCP tool remains `replay_policy="never"`. Existing Agent Session Runtime commits `ToolEffectPending` before calling the closure; recovery of an uncertain effect settles `outcome_unknown` and does not call MCP again.

Immediately before any MCP/OAuth HTTP I/O for a tool effect, the closure calls `check_cancelled` and runs one short gate transaction. The canonical lock order is Connection head → generation → Grant → parent Run → Child Session lease when applicable. The gate verifies:

- the exact owner/Run normalized pin and activation epoch;
- an active matching Grant/audience and allowed secret version;
- product endpoint/network/quota policy;
- live parent Run lease, worker, fencing epoch, and no cancellation;
- for a Child Session, the `execution_scope`, live child lease, and `ToolRuntime` fencing epoch;
- the already committed `(execution_scope, intent_id)` pending effect.

The transaction commits before session creation and returns a one-shot in-memory dispatch token. Disable/revoke takes the same Connection/Grant locks:

- revoke commits first: the gate rejects and the MCP adapter receives zero calls;
- gate commits first: the operation is considered in flight, even if socket output follows the revoke.

No database transaction is held over network I/O. The in-flight watch tracks the exact owner/Connection activation, active Grant, and parent/Child execution authority, not routine secret-version changes from same-Grant refresh or ciphertext re-encryption. Credential replacement and revocation still retire the old Grant. NOTIFY and a process-local task index provide best-effort cancellation of in-flight sessions, but no rollback promise. Lease loss, cancellation, timeout, disconnect, or crash after possible dispatch settles as failed/unknown and never causes an automatic retry.

Connection, session, DNS, and credential caches include at least `(owner_id, endpoint_digest, grant_id, secret_version)`; unauthenticated keys still include owner and endpoint. First release uses one foreground SDK session per call and does not build a universal connection-sharing platform.

## OAuth and credential lifecycle

The locked SDK source `mcp/client/auth/oauth2.py` has local SHA-256 `e709fa1676352a417afb6d653599141072c562d4f0dd8ca1d67f1cee8d85b54f` and matches the upstream `v2.2.0` file. Use its `OAuthClientProvider`, `TokenStorage`, PKCE/state, resource validation, redirect/callback hooks, token exchange, and refresh behavior. Do not implement a parallel OAuth protocol.

DlightRAG supplies only product integration:

1. Settings starts authorization for one authenticated owner/Connection.
2. The SDK redirect handler records a hash of SDK state with an expiring flow and returns the authorization URL to that Settings session.
3. `GET /web/oauth/connections/mcp/callback` requires the authenticated owner, locates the flow by state hash, encrypts the callback result once, strips the query by redirecting back to Settings, and never logs it.
4. The callback may land on any worker; PostgreSQL/NOTIFY wakes only the live `flow_owner`, whose SDK callback handler atomically consumes the inbox.
5. SDK TokenStorage persists tokens and client information in the Grant envelope scoped by owner, Connection, Grant, and audience.

The SDK's pending PKCE verifier/state remains process memory. If the flow owner dies or its lease expires, no other worker resumes the exchange: the flow expires, any callback becomes unusable, and Settings asks the user to authorize again.

OAuth uses an expiring Grant-scoped refresh lease without a database transaction over remote I/O. As an approved safety refinement, refresh preflight is separate from the effect session: the locked SDK constructs refresh requests and safe same-origin token redirects, and its generator is closed before its original MCP request can be sent. Failed refresh never enters discovery, registration or consent. Foreground preflight first checks the trusted pending effect, owner, Grant, Run/Child fence and cancellation; after a successful TokenStorage CAS, the complete effect gate runs again before one token-only MCP call. TokenStorage CAS requires `(grant_id, active status, lease owner, live refresh epoch, expected secret version)`. Only one refresh sequence runs per Grant; expiry permits takeover but stale saves/releases cannot overwrite expired-lease re-encryption, replacement or revocation. Cosmetic re-encryption skips live refresh leases both at candidate selection and at its actual CAS, preserving externally rotated refresh tokens. CAS failure discards returned tokens. No 401/403 after an effect causes refresh-and-replay.

A provider-requested scope outside `consented_scopes` never expands authority in a worker. It marks `needs_auth`; Settings performs a new consent flow and publishes a new Grant/generation. Expired credentials refresh automatically only within already consented scope. Missing refresh capability, rejected refresh, or nonstandard provider requirements become `needs_auth`, not a spontaneous worker redirect. SDK support does not imply universal provider compatibility.

## Streamable HTTP security and limits

Only the Connection endpoint and SDK-discovered OAuth URLs pass through the shared network-admission primitives. Validate every connect/reconnect/redirect and OAuth metadata/token target, classify every resolved address, pin the admitted address while preserving Host/SNI, reject HTTPS downgrade, and disable transport-level automatic retries.

Default policy requires HTTPS and denies loopback, private, link-local, multicast, and metadata addresses; an operator can explicitly allow narrower self-hosted destinations. Such policy permits network reach only—it never authorizes a user's remote account. Reject userinfo, fragments, credential-bearing query parameters, arbitrary browser headers, cookies, proxy headers, and inbound DlightRAG bearer forwarding.

Finite connect, request, idle, and total timeouts; redirect count; concurrent calls; response/SSE buffer; tool-result bytes; and structured-content/media counts are mandatory. Remote definitions, instructions, errors, and results are untrusted and length-bounded. Server instructions do not enter the system prompt. Supported tool text/structured content passes through existing fit/spill policy; secrets and raw sensitive errors do not enter logs/events.

## Settings routes and UX

Management is a Web projection only:

| Method and path | Meaning |
|---|---|
| `GET /web/api/connections/mcp` | Redacted owner list, revisions, and observed status; never catalogue content |
| `POST /web/api/connections/mcp` | Create a disabled Streamable-HTTP Connection |
| `PATCH /web/api/connections/mcp/{connection_id}` | Revision-CAS label/endpoint/enable/disable commands |
| `DELETE /web/api/connections/mcp/{connection_id}` | Revision-CAS tombstone and immediate future-dispatch revocation |
| `POST /web/api/connections/mcp/{connection_id}/probe` | Queue/perform a bounded owner probe without enabling |
| `PUT /web/api/connections/mcp/{connection_id}/bearer` | Replace a write-only bearer with a new Grant |
| `POST /web/api/connections/mcp/{connection_id}/oauth` | Begin SDK OAuth from Settings |
| `POST /web/api/connections/mcp/{connection_id}/revoke` | Revoke/erase the active Grant and block future dispatch |
| `GET /web/oauth/connections/mcp/callback` | State-bound OAuth callback inbox deposit; not a public management interface |

Mutations use current same-origin auth/CSRF and expected revisions; the OAuth callback uses authenticated owner plus SDK state because a provider redirect cannot supply same-origin CSRF. Cross-owner identifiers return not-found. `simple` returns 403 and bootstrap hides the Feature; eligibility is decided by that bootstrap capability, so the projection carries no eligibility flag of its own.

The list distinguishes authoritative `disabled/enabled/revoked` from observations `ready/degraded/needs-auth/refreshing` by reporting the redacted observed status, and never claims a permanent global “connected” state. It shows no tool names, schemas, catalogue age, or raw error kinds: Settings answers whether a server is reachable and authorized, and the Agent is the only consumer of what that server offers. The projection carries exactly what Settings renders — catalogue facts and error classification stay server-side — while activation epoch and generation remain because the integration suite reads them here as the authoritative read model.

Enable requires one whole-Connection warning acknowledgement per owner session — the first enable in a drawer session asks once and every later switch is a single tap; an owner who already has an enabled Connection is never asked again. The operator-recorded `consent_version=1` attests to that standing authorization rather than to a per-Connection dialog. New catalogue publication does not ask again; OAuth scope growth does. The switch also owns discovery: a Connection with no confirmed catalogue is probed before it is enabled, and a check that reports an authentication failure leaves the Connection off instead of enabling a server that cannot answer.

There is no composer affordance, per-conversation selection, or per-tool checkbox.

Settings offers no **revoke** command even though the route exists: delete already retires the Grant and erases the encrypted envelope, so revoke would only preserve the endpoint string while costing a second destructive action next to the one that removes the Connection. A future surface that needs to keep the configuration while destroying only the credential may expose it again.

## Fault behavior

- Authentication failure marks only that Connection `needs-auth`; its call returns a bounded failed ToolResult and Research continues.
- Transport, protocol, timeout, malformed result, or remote tool disappearance marks only that Connection `degraded`; last-good definitions remain pinned and other Connection/built-in tools remain callable.
- Tool failure text names the Connection/tool, says whether outcome may be unknown, forbids automatic retry, and instructs the final Answer to identify the requested part not completed.
- Catalogue refresh failure publishes nothing and preserves last-good; a new Connection without a complete catalogue cannot be enabled.
- Store/gate/fence corruption is an internal authority failure and may fail the Run; an ordinary remote MCP fault must not.
- A remote rejection of an old pinned schema is reported; the Run never silently rediscovers or switches generation.

## Vertical implementation slices and acceptance tests

### Slice 1 — Owner module, state, and Settings CRUD

Land neutral contracts, Connection migrations/store, eligible-owner policy, bootstrap contract v2, redacted Web routes, Settings navigation, disabled drafts, unauthenticated/static bearer storage, secret-source/cipher integration, network admission, and bounded probe. No Answer tools yet.

Tests: unit commands/revisions/eligibility; Postgres two-owner isolation and envelope no-echo; browser JWT A/B and local-none paths; `simple` 403 and hidden bootstrap capability; v1/v2 mismatch rejected rather than silently granting access; CSRF; malicious endpoint/DNS/redirect cases; environment/explicit-env-file key-ring loading, YAML rejection, invalid/missing/rotated key handling; snapshots/logs/errors contain no bearer or encryption key.

### Slice 2 — Publication, automatic refresh, and whole-Connection consent

Land all-or-nothing discovery, stable tool identity, enable warning/version, due-work claims, backoff, last-good health, and automatic future generation publication.

Tests: duplicate/colliding/malformed/oversized candidates publish zero; missed NOTIFY/startup scan converges; two workers cannot publish a stale claim; new tool and schema drift automatically create a future generation; no tool checkbox exists.

### Slice 3 — Cross-interface atomic Run binding

Land acceptance binding, input encoding, normalized pins, both `create_run` paths, Web forwarding, executor resolver, and Fast exclusion. Remove static deployment tuple only when this slice passes.

Tests: the same JWT owner receives the same automatic tool plan through Application, REST, inbound MCP, and Web; owner B and `simple` do not; Fast does not; Web Conversation/turn/Run/pins commit or roll back together; snapshot→GC→accept cannot dangle; R1 pins generation 1 while R2 pins 2 after publication; process restart reconstructs R1 exactly.

### Slice 4 — Effect gate, recovery, and degraded continuation

Land trusted claims, ToolRuntime session fence, gate locks, cancellation task index, result bounds, status updates, and never-retry outcomes.

Tests: deterministic barriers prove revoke-first causes zero fake HTTP calls and gate-first is in-flight/unknown; disable→enable never revives an old binding; stale parent and Child Session fences fail before I/O; cancellation fails before I/O; crash after pending/gate never redispatches; one failed Connection leaves remaining tools usable and final output reports the unavailable part.

### Slice 5 — SDK OAuth and fenced refresh

Land SDK provider/storage adapter, Settings start/callback, expiring single-use inbox, encrypted client/token state, grant replacement, needs-auth, and refresh lease/CAS.

Tests use local in-process fake AS/MCP adapters only: state/PKCE/resource/audience behavior, callback on worker B waking worker A, duplicate/expired callback rejection, worker-A death requiring reauthorization, concurrent refresh serialization, stale refresh losing CAS to revoke/replacement/expired-lease rotation, live refresh surviving cosmetic rotation attempts, gate-first calls surviving routine secret-version changes, and no transaction held during remote refresh.

### Slice 6 — Retention, hardening, and close-out

Land GC, key-rotation operation support, quotas/metrics, and docs/current-state cutover. Deployment `OutboundMcpServerConfig`/stdio paths and their re-export shim are removed. OAuth preflight, GC and writer keyring maintenance are implemented; final independent full validation/review remain pending.

Tests: pinned generations survive until retained Run deletion; cascade releases pins; retired secrets are removed from the live store; cache keys cannot cross owner/grant/version; fault matrix covers auth, DNS, protocol, timeout, unknown side effect, catalogue drift, shutdown, and listener reconnect; `uv run lint-imports` proves no reverse import.

Each slice adds unit and Postgres integration coverage at the deep module interface, plus browser tests only where browser behavior exists. The final gate includes the repository CI, multi-worker fake-transport races, migration verification for writer/reader roles, and a bounded local relative-link check.

## Known truthful limits

- A pinned definition cannot freeze remote implementation or data.
- A local Credential Grant is not proof of external tenant/user isolation; it may intentionally authorize a shared external workspace.
- Gate commit to socket I/O has an acknowledged interval classified as in flight.
- Closing a request or MCP session does not roll back a remote side effect.
- OAuth works only where the provider interoperates with the locked SDK flow; unsupported extensions remain unsupported. Providers omitting `expires_in` have no known expiry to trigger automatic refresh. A later authentication rejection requires explicit reauthorization; no invented TTL, forced refresh on 401, or effect replay compensates for missing expiry metadata.
- Automatic polling has bounded, nonzero catalogue staleness; current Runs intentionally remain on their accepted generation.

## Current operational lifecycle

- Writer startup applies the single current Connections migrations; readers verify the schema without creating or adapting old tables. Reader processes may perform their normal owner operational commands/discovery but do not run writer keyring/GC maintenance. Incompatible schemas produce configuration/storage errors, never a destructive reset.
- Writer maintenance performs bounded batches (100 grants/generations, 100 expired inboxes), at startup and at most every 60 seconds. `Application.connections.maintain()` permits one trusted in-process writer pass. It returns only re-encryption/collection counts. No extra CLI, service, environment loader or general scheduler exists.
- Grant ciphertext re-encryption uses the injected active key and secret-version CAS. Live refresh leases are skipped at selection and CAS; a later bounded maintenance pass revisits them after release/expiry. A zero re-encryption count during a live lease is not completed key inventory and does not authorize old-key removal. Concurrent refresh and rotation cannot overwrite each other. Retired credentials lose live ciphertext immediately; secret-free metadata follows retained Run pins. Run deletion cascades pins; only unpinned non-head generations are collected. Tombstones are removed only after all pins and short-lived flows end. Live discovery/Grant leases prevent collection.
- OAuth flows are once-only and expire within the configured 30–600 seconds. Each worker admits at most four authorization tasks; PostgreSQL bounds each owner's active flows to four and recent retained flows to 128. Catalogue, endpoint, transport, tool and call limits remain in `ConnectionPolicy`.
- Notifications are hints with polling/reconnect startup scans. Shutdown cancels discovery/maintenance and pending authorizations before Run drain, then closes outstanding MCP tasks. A possibly sent effect is never replayed.
- See [credential rotation operations](configuration.md#personal-connection-credential-rotation) for deployment order, verification and backup limits.

Focused fake-SDK, actual PostgreSQL multi-worker, browser, accessibility and i18n regressions cover these paths. Dedicated full CI and independent reviewers must still validate the combined owned worktree. Tests use no real model, MCP or OAuth provider; dead-worker refresh takeover is modeled by durable lease expiry, not an OS process kill. Unknown in-flight external effects cannot be rolled back.
