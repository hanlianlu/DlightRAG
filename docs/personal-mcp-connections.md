# Personal MCP Connections

This document owns the personal MCP Connection contract: Settings management,
credentials, catalogue publication, binding into Research Runs, dispatch, OAuth,
and retention. [ADR 0012](adr/0012-personal-connections-and-hot-plug.md) records
the decision, and [Domain Language](domain-language.md#personal-connections)
defines Connection, Credential Grant, Capability Catalogue, Connection
Generation, Run Connection Binding, and Connection Activation Epoch.

## Product contract

- Settings owns the path **Settings → Connections → MCP**. Each eligible owner
  creates, authorizes, enables, disables, and deletes only their own
  Connections.
- JWT owners (issuer and subject, including Web identities verified at a trusted
  edge) and the local single-user `none` owner are eligible. Shared `simple`
  authentication has no personal Connections.
- Every enabled Connection with a published catalogue joins each
  Research-capable Answer Run its owner accepts, whether the Answer arrives
  through Web, REST, inbound MCP, or the in-process Application. Fast has no MCP
  tools. The composer has no tool control, and a conversation has no Connection
  selector.
- Transport is MCP Streamable HTTP only. Stdio and deployment-declared servers
  are not accepted; `answer.agent.outbound_mcp` is a configuration error.
- Authentication is none, a write-only personal static bearer, or OAuth through
  the `mcp` SDK locked at 2.2.0.
- The Connection is the authorization unit. Enabling it authorizes every current
  and future tool the server publishes; there is no per-tool checkbox,
  allowlist, drift approval, or per-call confirmation. The enable warning states
  that tools can read, modify, send, or delete data the external account allows,
  including in shared external workspaces. OAuth scope expansion still needs
  provider consent.
- DlightRAG isolates Connection records, credentials, Runs, Conversations, and
  files by owner. It does not claim that an external MCP server enforces the
  same user boundary.
- A Connection fault does not fail the Research Run: the call returns a failed
  tool result, other tools stay usable, and the Answer names the part it could
  not complete. A possibly dispatched call is never retried automatically.

## Scope and non-goals

There is no marketplace or registry lookup, no public Connection-management REST
API, no inbound-MCP management tool, no arbitrary request header, and no MCP
resources, prompts, or apps: only `initialize`, `tools/list`, and `tools/call`
are used. There is no universal plugin manager, second `RunRuntime`, or durable
invocation-permit ledger; the Agent Session effect runtime remains the only
effect authority. Connections stay separate from models, Skills, Profile Memory,
and Web resources. The mechanics of immutable publication, atomic Run pin,
effect-time revoke, and retention stay private to
`dlightrag.application.connections`.

## Starter presets

`PRESETS` in `application/connections/presets.py` is a short, static,
hand-reviewed list that the owner read projects as `presets`: Notion (`oauth`),
Hugging Face (`none`), and Wolfram (`none`). Each is a first-party public HTTPS
endpoint, and none asks for a pasted secret. Choosing a preset fills the create
form's label and endpoint; after the create command succeeds, Settings opens the
new Connection on the preset's authentication tab. A preset creates, enables,
authorizes, and stores nothing, and the create command still validates the
endpoint; `tests/unit/test_connection_presets.py` holds every preset to that
same validation.

## Where it lives

Paths are under `src/dlightrag/` unless they start with `frontend/`.

| Module | Responsibility |
|---|---|
| `engine/answer/owner.py` | `personal_owner`: the eligibility rule (JWT or `none`), shared with Profile Memory, the Run pin writer, and the bootstrap capability |
| `application/connections/service.py` | `Connections`: Settings commands, catalogue validation, refresh scheduling, Research binding and restore, dispatch, OAuth flows, maintenance |
| `application/connections/models.py`, `policy.py`, `presets.py`, `client_metadata.py` | Commands, redacted views, and the store, MCP, and OAuth ports; `ConnectionPolicy`; presets; the Client ID Metadata Document |
| `application/connections/credentials.py` | `CredentialCipher`: key-ring validation and AES-256-GCM envelopes; access-token checks |
| `adapters/postgres/connections.py` | `PGConnectionsStore`: schema, owner-scoped state, claims and leases, revision CAS, NOTIFY, dispatch gate, GC. `PGConnectionPinWriter`: validation and pin insertion inside an accepting transaction |
| `adapters/mcp/personal_http.py`, `adapters/mcp/oauth.py` | `PersonalMcpClient`: bounded Streamable HTTP over an admitted transport. `PersonalOAuthClient`: SDK authorization and refresh preflight |
| `engine/network_admission.py` | DNS and IP admission shared with public Web reads |
| `engine/answer/execution/connection_binding.py` | Secret-free `RunConnectionBinding`, `ResearchToolClaim`, `ResearchConnectionToolResolver`, `StaleConnectionBindingError` |
| `engine/agent/tools/contracts.py` | `ToolRuntime.fencing_epoch`, which lets the gate fence parent and Child Session calls without owner, credential, or MCP facts in Agent Core |
| `application/answer_runs/service.py`, `adapters/postgres/runtime/run_store.py`, `application/web_conversations/`, `adapters/postgres/web/web_conversations.py` | Binding at Answer acceptance; pins written by `accept_run` or by `create_run_in` inside the Web turn transaction |
| `engine/answer/execution/executor.py`, `engine/answer/tools/composition.py` | Restoring pinned tools for resolved Research, checking the accepted `AgentRunPlan`, and preview-or-spill of tool output |
| `_compose.py`, `application/application.py`, `application/config/` | Composition, lifecycle order, the `answer.agent.connections` settings, and YAML rejection of the key ring |
| `adapters/http/browser/routes/connections.py`, `adapters/http/browser/auth.py` | Web routes; callback query capture and the public metadata path in the Web middleware |
| `adapters/http/browser/routes/bootstrap.py`, `routes/chat.py`, `answer_events.py` | The `personal_mcp_connections` capability; Tool Activity labels on Run event streams |
| `frontend/api/connections.ts`, `frontend/ui/settings-connections.ts` | Browser wire validation and the Settings feature |

Engine owns the neutral binding contracts and imports neither Application nor
MCP. Application imports Engine contracts, and the PostgreSQL and MCP adapters
implement Application-owned ports. `_compose.py` injects
`Connections.bind_research` into Answer acceptance and
`Connections.restore_research` into the executor. `uv run lint-imports` enforces
these directions
([ADR 0001](adr/0001-application-engine-adapters-architecture.md),
[ADR 0011](adr/0011-owner-specific-operational-state-adapters.md)). Connections
owns Grants, catalogues, generations, and bindings; the Agent Session runtime
keeps Effect Intent and Effect Settlement; the MCP adapter owns process-local
SDK sessions; product policy can deny authority but never grant it.

```python
class Connections:
    # Settings; an ineligible auth mode raises ConnectionsError (status 403)
    async def read(self, *, owner_id, auth_mode) -> ConnectionsView
    async def change(self, *, owner_id, auth_mode, expected_revision,
                     command) -> ConnectionsView
    async def replace_bearer(self, *, owner_id, auth_mode, connection_id,
                             bearer, expected_revision=None,
                             endpoint=None) -> ConnectionsView
    async def begin_authorization(self, *, owner_id, auth_mode, connection_id,
                                  expected_revision,
                                  endpoint=None) -> AuthorizationStart
    async def authorization_callback(self, *, owner_id, auth_mode, state,
                                     code=None, issuer=None,
                                     error=None) -> None
    def published_client_metadata(self) -> dict | None
    # Answer
    async def bind_research(self, *, owner_id,
                            auth_mode) -> BoundResearchConnections
    async def restore_research(self, *, bindings,
                               claim) -> tuple[AgentTool, ...]
    async def pinned_tool_labels(self, *, owner_id, auth_mode,
                                 run_id) -> Mapping[str, str]
    # Lifecycle
    async def start(self, *, validate_only=False) -> None
    async def maintain(self) -> dict[str, int]
    async def stop_refresh(self) -> None
    async def aclose(self) -> None
```

Management results never contain a token, authorization code, client secret,
envelope, or catalogue. `BoundResearchConnections` holds schema-only tool
declarations and secret-free bindings. `ResearchToolClaim` carries the trusted
owner, Run, worker, Run fencing epoch, and cancellation check of the executing
Run; no request or model argument supplies an owner.

## Durable state

Every foreign key and owner lookup includes `owner_id`.

- `dlightrag_connection_heads(owner_id, connection_id, revision, label, enabled,
  activation_epoch, head_generation, consent_version, tombstoned_at,
  refresh_due_at, refresh_owner, refresh_epoch, refresh_expires_at,
  observed_status, last_attempt_at, refresh_failures, last_error_kind)`: the
  mutable head, with a deferrable foreign key to its current generation.
- `dlightrag_connection_generations(owner_id, connection_id, generation,
  endpoint_json, endpoint_digest, grant_id, catalogue_json, catalogue_digest,
  created_at)`: immutable. `endpoint_json` holds only the validated URL,
  `grant_id` is null for an unauthenticated generation, and `catalogue_json` is
  null until discovery succeeds.
- `dlightrag_connection_grants(owner_id, connection_id, grant_id, kind,
  audience_digest, consented_scopes, status, secret_version, encrypted_envelope,
  key_id, refresh_owner, refresh_epoch, refresh_expires_at, updated_at)`: `kind`
  is `bearer` or `oauth`, `audience_digest` is the SHA-256 of the endpoint the
  Grant was issued for, and `status` is `active` or `retired`.
- `dlightrag_answer_connection_pins(owner_id, run_id, connection_id, generation,
  activation_epoch, catalogue_digest)`: composite foreign keys to the Run
  (`ON DELETE CASCADE`) and to the pinned generation.
- `dlightrag_connection_oauth_flows(flow_id, owner_id, connection_id,
  flow_owner, flow_lease_expires_at, endpoint, expected_revision, state_hash,
  encrypted_result, encrypted_credentials, expires_at, deposited_at,
  consumed_at, finished_at, succeeded)`: a short-lived callback inbox, not a
  workflow runtime.

Policy quotas bound each generation's catalogue JSON. The normalized pin table,
not the bindings in a Run's accepted input, is what keeps GC from deleting a
pinned generation.

## Secret handling and key ring

The key ring is
`DLIGHTRAG_ANSWER__AGENT__CONNECTIONS__CREDENTIAL_SECRET_KEYRING`. The existing
`load_config(env_file=...)` pipeline reads it from the process environment or
the operator's `.env` (orchestrator Secrets arrive as environment) into
`answer.agent.connections.credential_secret_keyring: SecretStr | None` with
`exclude=True` and `repr=False`; a trusted in-process caller may pass the same
typed field. YAML that sets the field fails configuration even when a
higher-priority source would override it. It is a secret carried by typed
settings, not Application Configuration or a Deployment Binding
([ADR 0006](adr/0006-configuration-ownership-and-deployment-bindings.md)).

The value is JSON:
`{"active":"<key-id>","keys":{"<key-id>":"<base64url 32-byte key>"}}`. Only
those two members are allowed, duplicate JSON keys are rejected, key IDs match
`[A-Za-z0-9_-]{1,64}`, every key decodes canonically to exactly 32 bytes, and
`active` names a listed key. Validation errors never echo a value. Only
`CredentialCipher` unwraps the secret. `_compose.py` also passes the whole
`answer.agent.connections` settings object, as the policy, to `Connections`,
which hands it to the PostgreSQL store and the MCP and OAuth adapters; they hold
the excluded `SecretStr` without reading it.

Encryption is AES-256-GCM from `cryptography`, with a fresh 12-byte nonce and
associated data that binds the owner, Connection, and Grant or OAuth flow. The
stored envelope is `{"version":1,"key_id":…,"nonce":…,"ciphertext":…}`.
Envelopes hold bearer tokens, OAuth tokens and client information with the
authorization-server metadata that refresh needs, and the in-flight callback
result and credentials of an OAuth flow.

- New encryption uses `active`; decryption uses whichever listed key an envelope
  names. Every worker needs the same ring.
- A missing ring or an unreadable envelope fails closed with 503. A Connection
  whose existing envelope cannot be read refuses bearer replacement and OAuth
  instead of overwriting it. Nothing falls back to plaintext or a generated key.
- No key reaches PostgreSQL, settings dumps, the UI, logs, or Run input.
- Retiring a Grant erases its ciphertext from the live store. That does not
  erase backups a retained key can still decrypt; backup and key retention stay
  operator responsibilities.

Rotation adds a key, switches `active`, lets writer maintenance re-encrypt live
Grants, and removes the old key only after its counts reach zero; see
[credential rotation](configuration.md#personal-connection-credential-rotation).

## Publication and discovery

- **Create** stores a disabled draft without network I/O: generation 0 with the
  endpoint, no catalogue, and observed status `disabled`. An owner may hold
  `max_connections` Connections that are not deleted.
- **Discovery** runs outside any database transaction, for a probe, a background
  refresh, an endpoint edit of an unauthenticated Connection, a bearer save, and
  an OAuth authorization. Its result is published in one short transaction under
  the owner's advisory lock and revision CAS.
- Discovery opens one SDK session, calls `initialize`, and pages `tools/list` up
  to `max_pages`. Each cursor must be non-empty, at most 2048 characters, and
  unseen. Each tool needs a unique name matching `[A-Za-z0-9_.-]{1,128}`, a
  description within `max_description_bytes`, and an `object` input schema that
  is valid JSON Schema (Draft 2020-12) within `max_schema_bytes`; the list must
  fit `max_tools` and `max_catalogue_bytes`. Echoes of the Connection's own
  credential are redacted first. Any violation rejects the whole candidate.
- A tool's local name is
  `mcp_<connection_id>_<24 hex digits of SHA-256(remote name)>`; a collision
  rejects the candidate. The server's `initialize` instructions are discarded.
- **Refresh.** Each worker runs `discovery_concurrency` refresh loops. A loop
  claims one enabled, due head that is not deleted, not revoked, and not under a
  live claim, using `FOR UPDATE SKIP LOCKED`. The claim increments
  `refresh_epoch`, takes a lease of `2 × discovery_timeout + 5` seconds, sets
  status `refreshing`, and commits before any network I/O. Publication requires
  the same claim owner and epoch, the head revision seen at claim time, the same
  generation Grant, and that Grant's secret version and refresh epoch, so a late
  worker cannot overwrite a newer edit, disable, or credential change.
- A successful refresh publishes a new generation, even when the catalogue is
  unchanged, sets `ready`, and schedules the next refresh `refresh_seconds`
  later. A failed refresh publishes no generation and keeps the last-good one.
  Either outcome increments the head revision. A failed refresh records
  `needs-auth` for an authentication failure and `degraded` otherwise, then
  backs off `min(refresh_seconds, 2^n)` seconds times a random factor from 0.8
  to 1.0, where `n` counts consecutive failures up to 10. If a new catalogue
  would push an enabled Connection past `max_enabled_tools`, the refresh records
  a `quota` error instead of publishing.
- `NOTIFY dlightrag_connections_changed` wakes refresh loops and in-flight
  watchers. Loops also wake at least once a second, and every listener
  (re)connect triggers a scan, so a missed notification only delays work.
- New tools and schema or description changes reach future Runs through
  publication, with no restart and no per-tool consent. Existing pins never
  change.

## Atomic acceptance and retention

`bind_research` performs no remote I/O. Answer acceptance calls it when the
requested mode is not `fast` and Research is in the Valid Mode Set. It returns
schema-only tools and bindings for the owner's Connections that are enabled, not
deleted, recorded with `consent_version=1`, and published with a catalogue, and
whose generation Grant, if any, is active. The tools join the accepted
`AgentRunPlan`, and the bindings enter the prepared input as
`run_connection_bindings`: at most 100, exact fields, no duplicates, no secrets.

`PGRunStore.accept_run`, which `create_run` forwards to, and `create_run_in`,
which runs inside the Web Conversation transaction, take the same steps inside
the accepting transaction:

1. Return an idempotent replay unchanged; an existing Run keeps its original
   pins.
2. Require the prepared input's bindings to equal the bindings being pinned, and
   allow bindings only for an eligible, non-Fast acceptance.
3. Lock the referenced heads in ascending `(owner_id, connection_id)` order.
4. For each Connection in the same order, lock its generation and then its
   Grant.
5. Verify owner, enabled state, consent version, current head generation,
   activation epoch, catalogue digest, and active Grant.
6. Insert the Run, its routing, the Web Conversation and turn rows where
   applicable, and every pin before commit.

A stale binding raises `StaleConnectionBindingError`, and no Run is created.
Acceptance returns an idempotent replay if one appeared meanwhile and otherwise
rebinds once without network I/O. A second stale result fails with
`AnswerConnectionsChangedError`, a conflict that asks the caller to submit
again.

How later changes reach Runs:

- Publication moves the head for future Runs only.
- Disable, delete, and revoke increment `activation_epoch`, so a later enable
  cannot revive an older binding. Enable keeps the epoch it finds.
- Every bearer save and every completed OAuth authorization creates a new Grant
  and generation and retires the Connection's earlier Grants: status `retired`,
  ciphertext erased, secret version and refresh epoch incremented. A token
  refresh stays within its Grant.
- Changing the endpoint of an authenticated Connection requires a new bearer or
  OAuth authorization (409 with kind `requires_reauthorization`). The old head
  and Grant stay live until the candidate's discovery and revision CAS succeed.
- A bearer save that includes the endpoint, which is what Settings sends,
  discovers with the new bearer first and keeps the enabled state. A bearer save
  without an endpoint stores the Grant, disables the Connection with a new
  activation epoch, and then probes.

Retention:

- Pins live as long as their Run row; deleting the Run cascades them.
- Writer maintenance collects bounded batches of expired OAuth flows; of
  non-head generations without pins on heads that hold no live refresh claim,
  Grant refresh lease, or OAuth flow row; of deleted heads, with their last
  generation and Grants, once no pin remains; and of retired Grants no
  generation references.
- Acceptance pins only the current head generation, which GC never deletes, and
  the pin's foreign key arbitrates any race, so a dangling pin cannot commit.
  Generation metadata follows Run retention, while retired ciphertext is erased
  at once.

## Restore, dispatch, revoke, and cancellation

For a Run that resolves to Research with bindings, the executor builds a
`ResearchToolClaim` from its trusted `RunSession` and calls the injected
resolver. `restore_research` loads the pinned generations, requires the stored
pins to equal the Run's bindings with matching catalogue digests, and never
substitutes the current head. The restored tools must reproduce the accepted
`AgentRunPlan` before any provider or tool effect. A Run that resolves to Fast
restores nothing. A pin freezes local names, schemas, descriptions, and routing;
it cannot freeze remote code, data, availability, or side effects.

Connection tools are declared `replay_policy="never"`, and their arguments must
validate against the pinned schema. The Agent Session runtime commits
`ToolEffectPending` before calling the tool; recovery of an uncertain pending
effect settles `outcome_unknown` and never calls MCP again.

A call proceeds in this order:

1. Arguments larger than `max_call_argument_bytes` fail with "No call was sent".
2. Within `call_timeout` and one of the worker's `call_concurrency` slots, the
   call checks cancellation and runs the gate: one short transaction that is
   never rerun, even when its commit acknowledgement is lost.
3. The gate locks the head, generation, Grant, parent Run, and, for a Child
   Session, the child lease row, in that order. It verifies that:
   - the tool belongs to a generation pinned for this owner and Run, and the
     runtime tool name matches;
   - the head is enabled, not deleted, recorded with `consent_version=1`, and at
     the pinned activation epoch, and the generation's catalogue digest matches;
   - any Grant is active, of kind `bearer` or `oauth`, holds an envelope, and
     has an audience digest equal to the generation's endpoint digest, and the
     stored endpoint matches that digest;
   - the parent Run is `running` under a live lease held by this worker at the
     claim's fencing epoch, with no cancellation request; a parent call carries
     that fencing epoch, a Child Session call needs its own live, uncancelled
     child lease at the `ToolRuntime` fencing epoch, and the Agent Session lease
     matches;
   - exactly one committed `ToolEffectPending` exists for this
     `(execution_scope, intent_id)`, and its tool name, call id, `never` replay
     policy, contract version, schema digest, argument digest, and stored
     arguments match the call.
4. The endpoint is checked against network policy again. For a Connection with a
   Grant, the credential is decrypted, an expired OAuth access token goes
   through the refresh preflight, and the whole gate then runs a second time,
   whether or not a refresh happened.
5. The worker checks cancellation, refuses a second call for the same owner,
   Run, execution scope, and intent, and calls the tool on a fresh SDK session.

Revoke, disable, and delete lock the same head and Grant rows as the gate:

- If revoke commits first, the gate rejects and the MCP adapter receives zero
  calls.
- If the gate commits first, the call is in flight, even if its request leaves
  the process after the revoke.

No database transaction spans network I/O. While a call runs, a watcher checks
every 0.25 seconds, or on NOTIFY, that the head is still enabled at the same
activation epoch with consent, that the Grant is still active, and that the Run
and Child fences still hold. It ignores routine secret-version changes from a
same-Grant refresh or re-encryption. Losing any of these, a timeout,
cancellation, or shutdown cancels the local task; the result is a failed call
whose outcome may be unknown, with no rollback and no automatic retry.

There is no shared connection, session, DNS, or credential cache. Every
discovery and call opens a new SDK session on a new transport without
keep-alive, and admits its target per request.

## OAuth and credential lifecycle

Authorization uses the SDK's `OAuthClientProvider`, `TokenStorage`, PKCE and
state, resource validation, redirect and callback hooks, token exchange, and
refresh. DlightRAG supplies only the product integration:

1. Settings calls `POST .../oauth` for one owned Connection at the current
   revision. The request fails before contacting any provider unless
   `oauth_callback_url` is set to a URL whose path is exactly
   `/web/oauth/connections/mcp/callback` with no query, the key ring can
   encrypt, and any existing envelope can be read.
2. A flow row records the initiating worker as `flow_owner`, the endpoint, the
   expected revision, and a lifetime of `oauth_timeout`. The initiator renews a
   10-second lease every 2 seconds. Starting again supersedes the Connection's
   pending flow.
3. The SDK discovers the protected resource and authorization server, registers
   or identifies the client, and calls the redirect hook. The hook accepts one
   redirect per flow; a later step-up needs a new flow. It validates the
   authorization URL (no userinfo or fragment, HTTPS when required, exactly one
   `state`, an admitted host), records the SHA-256 of `state`, takes the
   requested `scope` as the consented scopes, and hands the URL to Settings. The
   request waits at most `discovery_timeout` for it.
4. `GET /web/oauth/connections/mcp/callback` requires the authenticated owner.
   Web middleware moves the query string into request state before any logging.
   The handler finds a live flow by owner and state hash, encrypts
   `{state, code, iss, error}` once, deposits it, and redirects to
   `/web/?settings=connections`, adding `&authorization=restart` on failure. The
   callback may reach any worker; NOTIFY wakes only the flow owner, whose SDK
   callback hook consumes the deposit exactly once.
5. After the token exchange, the SDK finishes discovery with the new token.
   DlightRAG then publishes the Grant, holding the encrypted tokens, client
   information, expiry, and authorization-server and protected-resource
   metadata, together with a generation for the new catalogue in one
   revision-CAS transaction, and marks the flow succeeded.

The SDK's pending PKCE verifier and state live only in the initiating process.
If that worker dies or its lease lapses, no other worker resumes the exchange;
the flow fails, and Settings asks the user to authorize again. A token whose
scope exceeds the consented scopes is rejected, so broader scope always takes a
new Settings authorization and a new Grant. Each worker runs at most four
authorizations, and PostgreSQL allows an owner at most four live flows and 128
unexpired flows; beyond these, the request fails with 429.

An authorization carries the owner revision read when it began, and completing
it is a CAS on that revision. Every published background refresh of any of the
owner's Connections changes the revision, and refresh claims do not wait for a
pending authorization. A refresh that publishes during the authorization window,
which can last up to `oauth_timeout`, therefore fails the authorization with
"Connections revision changed", and the user has to start again. The more
enabled Connections an owner has, the likelier this is: each refreshes
`refresh_seconds` after a success, and while it keeps failing, after a backoff
that starts at about two seconds.

### Client registration

The MCP authorization specification orders client registration: a client the
server already knows, then a Client ID Metadata Document, then Dynamic Client
Registration. DlightRAG supports the last two and lets the locked SDK choose:

- When the authorization server advertises
  `client_id_metadata_document_supported`, the SDK uses this deployment's
  published document URL as `client_id`. No registration call happens and no
  client secret exists.
- Otherwise the SDK registers dynamically. A client secret returned by
  registration is stored only inside the encrypted Grant envelope.

`GET /web/oauth/connections/mcp/client-metadata` serves the document:
`client_id`, `client_name`, `redirect_uris`, `grant_types`, `response_types`,
and `token_endpoint_auth_method="none"`, with
`Cache-Control: public, max-age=300`. It is public by protocol, because an
authorization server fetches it without a credential; it carries only facts an
authorization redirect already reveals and reads no owner, cookie, or Connection
state. Its `client_id` is derived from the configured callback: the same HTTPS
origin, the fixed path `/web/oauth/connections/mcp/client-metadata`, and no
userinfo, query, or fragment, and it equals the document URL exactly. A callback
that is not HTTPS publishes no document (the path returns 404), and its
Connections register dynamically. Pre-registered client credentials are not
supported.

### Token refresh

An access token is used only while it is unexpired. When a call or a catalogue
refresh finds it expired:

- The worker claims the Grant's refresh lease for `discovery_timeout + 5`
  seconds; one refresher runs per Grant, and others wait and then reuse its
  token. The claim needs a head that is not deleted and an active `oauth` Grant
  with an envelope for the same audience. A call runs the full gate before
  claiming and again before contacting the token endpoint, then requires the
  Grant id and secret version it started from.
- The SDK builds the refresh from the stored metadata. DlightRAG sends only POST
  requests to the stored token endpoint's origin, at most six, and closes the
  SDK's auth generator when it yields the original MCP request, which is never
  sent. A missing refresh token or token endpoint, a rejected refresh, a scope
  beyond the consented scopes, or any other failure of the token request,
  including its own connect or idle timeout, an admission refusal, a network
  error, or a server error, becomes an authentication failure (`needs-auth`).
  Only a refresh preflight that exceeds `discovery_timeout` as a whole, or a
  gate refusal, is recorded as `degraded` instead. With the default
  `idle_timeout` below `discovery_timeout`, a hung token endpoint therefore
  records `needs-auth`. Refresh never enters discovery, registration, or
  consent.
- Saving the new token is a CAS on Grant id, active status, lease owner, live
  lease, refresh epoch, and expected secret version; losing the CAS discards the
  token. Re-encryption skips live refresh leases at selection and again at its
  own CAS, so it never overwrites a rotated refresh token.
- After a refresh, a call runs the complete gate again and sends one request
  with the new token. No 401 or 403 after an effect triggers refresh-and-replay.

A provider that omits `expires_in` yields a token without a known expiry, which
is never refreshed automatically; a later authentication rejection marks the
Connection `needs-auth`.

## Streamable HTTP security and limits

- An endpoint is an absolute HTTP(S) URL with a host, and has no userinfo,
  fragment, control or whitespace characters, or credential-like query parameter
  (`token`, `key`, `secret`, `signature`, names ending in `_token` or `_secret`,
  and similar). `require_https` rejects plain HTTP.
- Every request, including OAuth metadata, registration, and token requests and
  followed redirects, resolves its host within `connect_timeout`. It is rejected
  when the host is `localhost`, `*.localhost`, or `*.local`, or when any
  resolved address is not public unicast: loopback, private, link-local (which
  covers cloud metadata addresses), multicast, reserved, or unspecified. A host
  matching an `allow_private_hosts` pattern is exempt. That policy grants
  network reach only, never remote-account authority.
- The request goes to the first admitted address with the original `Host` header
  and TLS SNI. Transport retries, HTTP/2, keep-alive, compression, and proxy
  settings from the environment are off; a compressed response is rejected, and
  a response body larger than `max_response_bytes` fails the request.
- A non-OAuth session accepts only requests to the endpoint's own scheme, host,
  and port. With an HTTPS endpoint or `require_https`, every hop must use HTTPS.
  An OAuth session may reach other admitted origins, but a bearer
  `Authorization` header may go only to the endpoint's origin.
- Only MCP protocol headers pass (`accept`, `content-type`, `content-length`,
  `mcp-protocol-version`, `mcp-method`, `mcp-name`, `mcp-session-id`,
  `last-event-id`), plus `Host`, `connection: close`,
  `accept-encoding: identity`, and the Connection's own `Authorization` header.
  Cookies, browser headers, and the caller's DlightRAG credentials never reach
  the server.
- The SDK follows a redirect only within the endpoint origin and with the same
  method. In a tool call the transport also refuses the SDK's GET stream, sends
  each JSON-RPC request id once, and sends at most one `tools/call`, so neither
  a redirect nor a stream resumption can resend an effect.
- A result may hold at most `max_result_parts` parts. Only text parts are
  accepted; image, audio, and resource parts fail the call. Structured content
  is appended as JSON and the bearer value is redacted; a result whose text is
  larger than `max_result_bytes` fails the call instead of being truncated. A
  remote error result becomes a failed call without its remote text. Accepted
  text goes through the preview-or-spill step shared by injected tools: text
  over the shared tool-result byte or line limit is previewed and spilled to a
  readable Resource. With Agent execution `disabled` there is no Agent Workspace
  to spill to, so such a call reports its full output as unavailable.
- SDK and HTTP-library log records are suppressed inside these sessions. Logs
  and Run events carry redacted categories, never remote text or secrets.

`answer.agent.connections` holds the non-secret policy. Operators tune it within
these hard limits. Exceeding a size or count limit fails the request, call, or
discovery candidate; nothing is truncated.

| Field | Default | Range | Bounds |
|---|---|---|---|
| `oauth_callback_url` | unset | ≤ 2048 characters | Public callback URL; OAuth is unavailable while unset |
| `oauth_timeout` | 300 s | 30–600 | Lifetime of one authorization flow, including its discovery |
| `max_connections` | 20 | 1–100 | Connections per owner, not counting deleted ones |
| `max_enabled_tools` | 256 | 1–1024 | Tools across one owner's enabled Connections |
| `max_tools` | 128 | 1–256 | Tools in one catalogue |
| `max_pages` | 16 | 1–32 | `tools/list` pages in one discovery |
| `max_schema_bytes` | 32768 | 1–65536 | One tool's input schema |
| `max_description_bytes` | 8192 | 1–16384 | One tool's description |
| `max_catalogue_bytes` | 524288 | 1–1048576 | One whole catalogue |
| `discovery_concurrency` | 4 | 1–16 | Refresh loops and concurrent discoveries or token refreshes per worker |
| `refresh_seconds` | 300 | 1–3600 | Refresh interval and backoff ceiling |
| `discovery_timeout` | 30 s | 1–120 | One discovery outside an authorization flow, one token refresh, or the wait for an authorization URL |
| `call_timeout` | 60 s | 1–120 | One tool call, gate included |
| `call_concurrency` | 8 | 1–32 | Concurrent tool calls per worker |
| `max_call_argument_bytes` | 65536 | 1–262144 | Encoded arguments of one call |
| `max_result_bytes` | 262144 | 1–1048576 | Text of one result |
| `max_result_parts` | 32 | 1–128 | Content parts of one result |
| `connect_timeout` | 10 s | 1–30 | Connect and DNS admission of one request |
| `idle_timeout` | 15 s | 1–60 | Read and write inactivity |
| `max_response_bytes` | 1048576 | 1–4194304 | One HTTP response body |
| `allow_private_hosts` | `[]` | host patterns | Hosts exempt from the private-address denial |
| `require_https` | `true` | boolean | Plain HTTP rejected |

## Settings routes and UX

Management is a Web projection only:

| Method and path | Meaning |
|---|---|
| `GET /web/api/connections/mcp` | Redacted owner list, revision, observed status, and presets; never catalogue content |
| `POST /web/api/connections/mcp` | Create a disabled Streamable HTTP Connection |
| `PATCH /web/api/connections/mcp/{connection_id}` | Revision-CAS `edit` (label or endpoint), `enable` (requires `consent_version: 1`), or `disable` |
| `DELETE /web/api/connections/mcp/{connection_id}` | Revision-CAS delete: tombstone, retire Grants, and revoke future dispatch at once |
| `POST /web/api/connections/mcp/{connection_id}/probe` | Discover now without enabling |
| `PUT /web/api/connections/mcp/{connection_id}/bearer` | Save a write-only bearer as a new Grant, optionally with the endpoint |
| `POST /web/api/connections/mcp/{connection_id}/oauth` | Begin SDK OAuth and return the authorization URL |
| `POST /web/api/connections/mcp/{connection_id}/revoke` | Retire and erase the Grants, disable, and mark the Connection `revoked` |
| `GET /web/oauth/connections/mcp/callback` | State-bound callback deposit; not a management interface |
| `GET /web/oauth/connections/mcp/client-metadata` | Public Client ID Metadata Document |

- Every command carries `expected_revision`; a stale revision returns 409. The
  revision is a digest of each head's `(connection_id, revision)`. Every command
  that writes a Connection, including a probe and a completed OAuth
  authorization, and every published background refresh, successful or failed,
  changes it; refresh claims and call observations do not. While a refresh keeps
  failing, its backoff starts at a few seconds, so a command can get 409 between
  the 5-second polls; Settings then asks for a reload. Commands that discover
  before they publish carry the revision they started with through the network
  work, so a refresh published in between fails them too: an endpoint edit or a
  bearer save that includes the endpoint within `discovery_timeout`, and an
  OAuth authorization within `oauth_timeout` (see
  [OAuth](#oauth-and-credential-lifecycle)). Mutations need the Web session plus
  the CSRF double-submit header and same-origin checks, in `none` mode as well.
  Validation errors return a generic 422 that echoes no input, and another
  owner's `connection_id` returns 404. An ineligible auth mode gets 403, and the
  bootstrap capability `personal_mcp_connections` is false, which hides the
  feature; the projection carries no eligibility flag of its own.
- `POST .../oauth` and the callback's 303 redirect send
  `Cache-Control: no-store` and `Referrer-Policy: no-referrer`. The callback
  relies on the authenticated owner and SDK state instead of CSRF, because a
  provider redirect cannot carry a same-origin token. The metadata document is
  the only public Web path besides login and logout.

Each Connection in the projection carries `connection_id`, `label`, `endpoint`,
`enabled`, `authentication` (`none`, `bearer`, or `oauth`), `status`
(`disabled`, `refreshing`, `ready`, `degraded`, `needs-auth`, or `revoked`),
`authorization_status` of its latest OAuth flow (`pending`, `succeeded`,
`failed`, or null), `activation_epoch`, and `generation`. `enabled` is
authoritative, and `status` is the last observation. The projection carries no
tool names, schemas, catalogue age, or raw error kinds, and Settings renders
neither the epoch nor the generation: Settings answers whether a server is
reachable and authorized, and the Agent is the only consumer of what the server
offers.

In Settings:

- The MCP group is collapsed by default and reads `N of M enabled`. The view
  polls every 5 seconds.
- Turning a Connection on shows the whole-Connection warning once per Settings
  session, and not at all when the owner already has an enabled Connection;
  every enable carries `consent_version=1`. A Connection whose status is neither
  `ready` nor `refreshing` is probed first, and a probe that does not end
  `ready` leaves it off. A new catalogue never asks again.
- The bearer field is write-only and saves with the current endpoint. OAuth
  shows a link to continue at the provider, and authorization must finish in the
  same session. Only an unauthenticated Connection edits its endpoint in place.
- Delete asks for confirmation and removes the endpoint, label, and credential.
  Settings offers no revoke action, because delete already retires the Grant and
  erases its ciphertext; the `revoke` route remains for a surface that must keep
  the configuration while destroying only the credential.
- There is no composer affordance, per-conversation selection, or per-tool
  checkbox.

The live Tool Activity is the one place outside Settings that names a remote
tool. When the browser subscribes to a Run's events,
`Connections.pinned_tool_labels` resolves each pinned tool to
`Connection label · remote tool name`, collapsed to one line of at most 96
characters, and the browser edge adds it to that tool's events. The label is
display only: it authorizes nothing, exposes no schema, endpoint, credential, or
Connection id, and is never stored, because the durable event keeps
transport-neutral identity. It survives a later disable or delete, since the
head row stays until its pins are gone. A Connection tool reports no Tool
Subject; like a Child Session or Memory Operation tool, its label is the whole
row.

## Fault behavior

- An oversized argument sends nothing, records no status, and returns a failed
  result saying that no call was sent. So does a denial by the first gate, and a
  `call_timeout` that expires while the call waits for a `call_concurrency` slot
  or inside the first gate.
- A second call, in the same process, for an intent the restored tool has
  already run returns a failed result saying that the outcome may be unknown,
  before any gate, and records nothing.
- Any failure after the first gate returns a failed result saying that the
  outcome may be unknown: the endpoint recheck, credential decryption, a token
  refresh, a denial by the second gate, the call itself, a timeout, or a
  cancellation of the call by the in-flight watcher or by shutdown. If the Run's
  own cancellation or lease check fires first, the Run stops without receiving a
  result.
- Every exit after the first gate tries to record an observation: `ready` after
  a successful call, `needs-auth` after an authentication failure (HTTP 401 or
  403, or any failed token request, including one that times out, is refused
  admission, or meets a network or server error), and `degraded` after anything
  else. That includes transport and protocol faults, an oversized or unsupported
  result, a remote tool error, a missing remote tool, a refresh preflight that
  exceeds `discovery_timeout` as a whole or that the gate refuses, and Run
  cancellation, lease loss, or shutdown. Recording is best effort: it gets two
  seconds, and a failure to record only logs a warning.
- An observation is recorded only while the call's generation is still the head
  generation, the head is still enabled at the same activation epoch, and the
  Grant is still active at the same secret version. A call stopped by revoke,
  disable, delete, or a replaced credential therefore records nothing.
- A failed call leaves pinned definitions in place, and other Connection and
  built-in tools stay callable. The failed result names the local tool, says
  whether no call was sent or the outcome may be unknown, forbids automatic
  retry, and tells the final Answer to identify the unfinished part.
- A failed catalogue refresh publishes no generation and keeps the last-good
  one. A Connection without a published catalogue cannot be enabled.
- Store failures and pin or digest inconsistencies are internal errors and can
  fail the Run; an ordinary remote fault does not.
- A remote rejection of an older pinned schema is reported as a failed call; the
  Run never rediscovers or switches generation.

## Known limits

- A pinned definition cannot freeze remote implementation or data.
- A local Credential Grant does not prove external tenant or user isolation; it
  may deliberately authorize a shared external workspace.
- Between gate commit and socket I/O there is a window in which the call counts
  as in flight.
- Closing a request or MCP session does not roll back a remote side effect.
- OAuth works only where the provider interoperates with the locked SDK flow.
  Nothing invents a token lifetime, forces a refresh on 401, or replays an
  effect to compensate for missing expiry metadata.
- Automatic refresh leaves a bounded, nonzero catalogue staleness, and a running
  Run stays on its accepted generation.

## Operational lifecycle

- Writer startup applies the Connections migrations (`personal_connections`,
  `answer_connection_pins`, `connection_oauth_inbox`). Readers verify the
  tables, columns, keys, and indexes without creating or altering anything. An
  incompatible schema is a startup error, never a reset.
- Readers and writers both run refresh loops and owner commands; only writers
  run maintenance. Maintenance runs at startup and then at most 60 seconds apart
  (`min(60, refresh_seconds)`). Each pass re-encrypts up to 100 Grants that are
  not under the active key, deletes up to 100 expired OAuth flows, and collects
  up to 100 generations or deleted heads. `Application.connections.maintain()`
  runs one pass on demand and returns only `reencrypted` and `collected` counts.
  There is no separate CLI or scheduler.
- Re-encryption is a CAS on secret version and envelope and skips live refresh
  leases, so a zero re-encryption count during a live lease does not prove that
  no old-key ciphertext remains. The rotation guide verifies counts before
  removing a key.
- Re-encryption runs before collection in each pass. If a live Grant names a key
  that is no longer in the ring, every background pass fails at that Grant and
  only logs a warning, and an on-demand `maintain()` raises. GC then stops for
  every owner until the key is back in the ring or that Grant is retired by
  delete, revoke, or a new credential.
- The Web middleware keeps callback query strings out of the application's own
  logs. Upstream proxies and external tracing sit outside the application and
  record those query strings unless the deployment redacts them there.
- Shutdown stops refresh, maintenance, notifications, and pending authorizations
  before the Run coordinator drains, then cancels outstanding MCP calls. A
  possibly sent effect is never replayed.

## Verification

Tests use in-process fakes for every model, MCP server, and OAuth provider; a
dead refresher is modeled by durable lease expiry, not by killing a process.

- `tests/unit/test_connections_config.py`: key-ring loading through the settings
  pipeline and YAML rejection.
- `tests/unit/test_connection_presets.py`,
  `tests/unit/test_connection_client_metadata.py`: presets and the published
  metadata document.
- `tests/unit/test_connection_binding.py`: the binding wire shape and the single
  acceptance retry.
- `tests/unit/test_connections_transport.py`: address pinning, redirect and
  header rules, pagination, result limits, and effect-replay blocking.
- `tests/unit/test_connection_oauth.py`,
  `tests/unit/test_connection_tool_labels.py`: SDK OAuth against in-process
  servers, and Tool Activity labels.
- `tests/integration/test_connections_pg.py`, `test_connection_binding_pg.py`,
  `test_connection_dispatch_pg.py`, `test_connection_authorization_pg.py`,
  `test_connection_lifecycle_pg.py`, `test_connections_web_pg.py`: the owner
  lifecycle, atomic pins across Application, REST, MCP, and Web, the gate and
  revoke races, cross-worker OAuth, refresh leases, re-encryption, GC, reader
  startup, and Web authentication and CSRF against PostgreSQL.
- `frontend/api/connections.test.ts`,
  `frontend/ui/settings-connections.browser.test.ts`: the browser wire and
  Settings behavior.
- `uv run lint-imports`: the dependency directions above.
