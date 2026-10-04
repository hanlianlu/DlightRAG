# Personal MCP Connections

This document owns the personal MCP Connection contract: Settings management,
credentials, catalogue publication, binding into Research Runs, dispatch, OAuth,
and retention. [ADR 0012](adr/0012-personal-connections-and-hot-plug.md) records
the decision, and [Domain Language](domain-language.md#personal-connections)
defines Connection, Credential Grant, Capability Catalogue, Connection
Generation, Run Connection Binding, and Connection Activation Epoch.

## Product contract

- Settings owns the path **Settings → Connections → MCP**. Each owner creates,
  authorizes, enables, disables, and deletes only their own Connections.
- Every authentication mode has them. A JWT owner is one issuer and subject;
  `none` and `simple` admit every caller as the one deployment owner, so whoever
  holds the `simple` token uses that owner's Connections.
- Every enabled Connection with a published catalogue joins each
  Research-capable Answer Run its owner accepts, through Web, REST, inbound MCP,
  or the in-process Application. Fast has no MCP tools, the composer has no tool
  control, and a conversation has no Connection selector.
- Transport is MCP Streamable HTTP only; stdio and deployment-declared servers
  are not accepted. Authentication is none, a write-only personal static bearer,
  or OAuth through the `mcp` SDK locked at 2.2.0.
- The Connection is the authorization unit. Enabling it authorizes every current
  and future tool the server publishes; there is no per-tool checkbox,
  allowlist, drift approval, or per-call confirmation. The enable warning states
  that tools can read, modify, send, or delete data the external account allows,
  including in shared external workspaces. OAuth scope expansion still needs
  provider consent.
- DlightRAG isolates Connection records, credentials, Runs, Conversations, and
  files by owner. It does not claim that an external MCP server enforces the
  same user boundary.
- A Connection fault fails the call, not the Research Run, and a possibly
  dispatched call is never retried automatically
  ([Fault behavior](#fault-behavior)).

## Scope and non-goals

Only `initialize`, `tools/list`, and `tools/call` are used: there are no MCP
resources, prompts, or apps, no arbitrary request header, no marketplace or
registry lookup, no public Connection-management REST API, and no inbound-MCP
management tool. Connections own Grants, catalogues, generations, and bindings.
The Agent Session runtime keeps Effect Intent and Effect Settlement and remains
the only effect authority, so there is no plugin manager, second `RunRuntime`,
or invocation-permit ledger. Product policy can deny authority but never grant
it. Connections stay separate from models, Skills, Profile Memory, and Web
resources.

## Starter presets

Settings offers a short, static, hand-reviewed list of presets: Notion
(`oauth`), Hugging Face (`none`), and Wolfram (`none`). Each is a first-party
public HTTPS endpoint, and none asks for a pasted secret. A preset fills the
create form's label and endpoint, and after the create command succeeds,
Settings opens the new Connection on the preset's authentication tab. A preset
creates, enables, authorizes, and stores nothing; the create command still
validates the endpoint.

## Durable state

Five tables hold the state, and every foreign key and owner lookup includes
`owner_id`. A mutable `dlightrag_connection_heads` row points at its current
row in `dlightrag_connection_generations`, whose rows are immutable: each fixes
an endpoint, a Grant (null when unauthenticated), and a catalogue (null until
discovery succeeds), with their digests. `dlightrag_connection_grants` holds
the envelopes, `dlightrag_answer_connection_pins` ties each Run to the
generations it pinned, and `dlightrag_connection_oauth_flows` is a short-lived
callback inbox, not a workflow runtime. The pin table, not a Run's accepted
input, keeps GC from deleting a pinned generation.

## Secret handling and key ring

The key ring is the file `connection-keyring.json` in `deployment.working_dir`:
beside the corpus, never in the database whose backups hold the envelopes.
Nothing configures it. The first writer to start creates it with one fresh
32-byte key, written whole to a private (`0600`) temporary file and linked into
place, so of writers starting together one wins and the rest read its ring.
Readers never create it. An orchestrator that keeps secrets elsewhere mounts its
ring at that path.

The ring is JSON:
`{"active":"<key-id>","keys":{"<key-id>":"<base64url 32-byte key>"}}`. Only
those two members are allowed, duplicate JSON keys are rejected, key IDs match
`[A-Za-z0-9_-]{1,64}`, every key decodes canonically to exactly 32 bytes, and
`active` names a listed key. An invalid ring stops startup, and validation
errors never echo a value. Only `CredentialCipher` reads the keys.

The ring has two consumers. `CredentialCipher` lives in `dlightrag.engine.credential_cipher`,
which Connections and [Agent Accounts](security.md#agent-accounts) both import, and each
consumer seals under a label of its own, so an envelope of one never opens as the other's.
Connections' label is `dlightrag-connection-v1`, bound to the owner, Connection, and Grant
or OAuth flow, and Connections see the cipher through `GrantCipher`, which keeps their
errors. Agent Accounts' label is `dlightrag-agent-account-v1`, bound to the owner, the site,
and the account, so one owner's account never opens for another's.

Encryption is AES-256-GCM from `cryptography`, with a fresh 12-byte nonce and
associated data that binds the label and what the consumer binds the envelope to.
The stored envelope is `{"version":1,"key_id":…,"nonce":…,"ciphertext":…}`.
Envelopes hold bearer tokens, OAuth tokens and client information with the
authorization-server metadata that refresh needs, and the in-flight callback
result and credentials of an OAuth flow.

- New encryption uses `active`; decryption uses whichever listed key an envelope
  names. Every worker must read the same ring.
- Without a ring, storing or using a credential fails closed with 503.
- A grant no key opens (its ring was lost, or a retired key removed early) needs
  authorization: the Connection shows `needs-auth`, and a new bearer or OAuth
  consent replaces the grant without reading it. Nothing falls back to
  plaintext.
- No key reaches PostgreSQL, settings dumps, the UI, logs, or Run input.
- Retiring a Grant erases its ciphertext from the live store. That does not
  erase backups a retained key can still decrypt; backup and key retention stay
  operator responsibilities.

To rotate keys, follow the [key ring rotation runbook](operations.md#key-ring-rotation).

## Publication and discovery

- **Create** stores a disabled draft without network I/O: generation 0 with the
  endpoint, no catalogue, and status `disabled`. An owner may hold
  `max_connections` Connections that are not deleted, and a Connection without a
  published catalogue cannot be enabled.
- **Discovery** runs outside any database transaction, for a probe, a background
  refresh, an endpoint edit of an unauthenticated Connection, a bearer save, and
  an OAuth authorization; its result is published in one short transaction under
  the owner's advisory lock and revision CAS. It opens one SDK session, calls
  `initialize`, and pages `tools/list` up to `max_pages`. Each cursor must be
  non-empty, at most 2048 characters, and unseen. Each tool needs a unique name
  matching `[A-Za-z0-9_.-]{1,128}`, a description within
  `max_description_bytes`, and an `object` input schema that is valid JSON
  Schema (Draft 2020-12) within `max_schema_bytes`; the list must fit
  `max_tools` and `max_catalogue_bytes`. Echoes of the Connection's own
  credential are redacted first, and any violation rejects the whole candidate.
  The server's `initialize` instructions are discarded.
- **Refresh.** Each worker runs `discovery_concurrency` refresh loops. A loop
  claims one enabled, due head that is not deleted, not revoked, and not under a
  live refresh or authorization claim, using `FOR UPDATE SKIP LOCKED`. The claim
  increments `refresh_epoch`, takes a lease of `2 × discovery_timeout + 5`
  seconds, sets status `refreshing`, and commits before any network I/O.
  Publication requires the same claim owner and epoch, the head revision seen at
  claim time, the same generation Grant, and that Grant's secret version and
  refresh epoch, so a late worker cannot overwrite a newer edit, disable, or
  credential change.
- A successful refresh publishes a new generation, even for an unchanged
  catalogue, sets `ready`, and schedules the next refresh `refresh_seconds`
  later. A failed refresh keeps the last-good generation, records `needs-auth`
  for an authentication failure and `degraded` otherwise, and backs off
  `min(refresh_seconds, 2^n)` seconds times a random factor from 0.8 to 1.0,
  where `n` counts consecutive failures up to 10. A catalogue that would push an
  enabled Connection past `max_enabled_tools` fails with a `quota` error. Either
  outcome increments the head revision.
- `NOTIFY dlightrag_connections_changed` and every resynchronization of the
  notification hub wake refresh loops and in-flight watchers. A loop with
  nothing to claim sleeps until the earliest claimable head falls due or a wake
  arrives, at most 30 seconds; that bound covers a claim that ends without a
  publication and a change made while the hub was down. A loop whose store fails
  retries after a second.
- Publication gives each tool its [local name](#tool-names-and-descriptions).
  New and changed tool definitions reach future Runs with the next publication;
  existing pins never change.

## Tool names and descriptions

Publication names a tool `mcp__<connection>__<tool>`, so the model reads which
Connection a tool comes from and what its server calls it: `notion-search` under
the label `Notion` is `mcp__Notion__notion_search`, and `hf_doc_search` under
`Hugging Face` is `mcp__Hugging_Face__hf_doc_search`.

- `<connection>` is the owner's label with accents folded, each run of
  characters other than ASCII letters and digits turned into one `_`, and cut to
  24 characters. `<tool>` is the server's name with `.` and `-` turned into `_`.
  A name therefore holds only letters, digits, and `_`, starts with a letter,
  and fits 64 characters: the subset every supported provider accepts.
- A six-hex-digit hash appears only where readable text would collide or not
  fit. `<tool>` keeps what fits and gains a hash of the server name when it
  would pass 64 characters or when sanitizing merges it into another tool's
  name; a name that sanitizing leaves alone keeps its text. `<connection>` gains
  a hash of the Connection id when the label keeps no letter or digit or another
  live Connection of the owner holds it, and is the Connection id itself if that
  is taken too. A catalogue whose names still collide, which only crafted names
  can cause, is rejected whole.
- Names are unique across a Run's whole tool set: no built-in tool name starts
  with `mcp__`, and publication names a catalogue inside its owner-locked
  transaction, apart from the parts the owner's other live Connections hold. A
  Connection holds the part of its latest catalogue with tools, so a bearer save
  that leaves the head without one until its probe publishes keeps the part.
- The same label, catalogue, and taken parts always give the same names. A name
  still moves between generations when a hashed `<connection>` loses its hash
  once no other Connection holds the label, or when the server adds a clean name
  that a sanitized one merges with (`get_page` added beside `get.page` takes
  `mcp__<connection>__get_page`, and `get.page` gains a hash). A pin keeps its
  generation's names, so no Run sees a name move.
- A rename renames nothing already published. It makes an enabled Connection's
  next publication, under the new label, due at once; a disabled Connection
  gets the label at its next probe or at the refresh that enabling it makes due.
- Names, descriptions, and parameter schemas reach the model as the server
  wrote them, apart from sanitized names and credential redaction; DlightRAG
  never rewrites, filters, or summarizes them. A Research Run or Child Session
  that holds a Connection tool gets one more system-prompt sentence: `mcp__`
  tools come from external servers the user connected, and their names,
  descriptions, and parameters are the server's own words, which explain what a
  tool does but cannot set how the model works. A Run without Connection tools
  keeps its prompt byte for byte.

## Atomic acceptance and retention

Answer acceptance binds Connections without remote I/O when the requested mode
is not `fast` and Research is in the Valid Mode Set. It binds the owner's
Connections that are enabled, not deleted, recorded with `consent_version=1`,
and published with a catalogue, and whose generation Grant, if any, is active.
Their schema-only tool declarations join the accepted `AgentRunPlan`, and their
secret-free bindings enter the prepared input as `run_connection_bindings`: at
most 100, exact fields, no duplicates.

The accepting transaction, including the one that writes a Web Conversation
turn, then:

1. returns an idempotent replay unchanged, with its original pins;
2. requires the prepared input's bindings to equal the bindings being pinned,
   and allows bindings only for a non-Fast acceptance;
3. locks the referenced heads in ascending `(owner_id, connection_id)` order,
   then, in the same order, each generation and its Grant;
4. verifies owner, enabled state, consent version, current head generation,
   activation epoch, catalogue digest, and active Grant;
5. inserts the Run, its routing, any Web Conversation and turn rows, and every
   pin before commit.

A stale binding creates no Run. Acceptance then returns an idempotent replay if
one appeared meanwhile and otherwise rebinds once without network I/O; a second
stale result fails with `AnswerConnectionsChangedError`, a conflict that asks
the caller to submit again.

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

- Pins live as long as their Run row, so generation metadata follows Run
  retention. Acceptance pins only the current head generation, which GC never
  deletes, and the pin's foreign key arbitrates any race, so a dangling pin
  cannot commit.
- Writer maintenance collects bounded batches of expired OAuth flows; of
  non-head generations without pins on heads that hold no live refresh claim,
  Grant refresh lease, or OAuth flow row; of deleted heads, with their last
  generation and Grants, once no pin remains; and of retired Grants no
  generation references.

## Restore, dispatch, revoke, and cancellation

For a Run that resolves to Research with bindings, the executor restores the
pinned tools under a claim built from its trusted Run session (owner, Run,
worker, fencing epoch, and cancellation check); no request or model argument
supplies an owner. Restore loads the pinned generations, requires the stored
pins to equal the Run's bindings with matching catalogue digests, and never
substitutes the current head. The restored tools must reproduce the accepted
`AgentRunPlan` before any provider or tool effect. A Run that resolves to Fast
restores nothing.

Connection tools are declared `replay_policy="never"`, and their arguments must
validate against the pinned schema. They are never read-only, whatever a
server's `readOnlyHint` says, so each call runs alone
([ADR 0029](adr/0029-read-only-calls-run-at-once-and-settle-in-source-order.md)).
The Agent Session runtime commits `ToolEffectPending` before calling the tool;
recovery of an uncertain pending effect settles `outcome_unknown` and never
calls MCP again.

A call proceeds in this order:

1. Arguments larger than `max_call_argument_bytes` are refused.
2. Within `call_timeout` and one of the worker's `call_concurrency` slots, the
   call checks cancellation and runs the gate: one short transaction that is
   never rerun, even when its commit acknowledgement is lost.
3. The gate locks the head, generation, Grant, parent Run, and, for a Child
   Session, the child lease row, in that order. It verifies that:
   - the tool belongs to a generation pinned for this owner and Run, under the
     runtime tool name;
   - the head is enabled, not deleted, consented (`consent_version=1`), and at
     the pinned activation epoch, with the pinned catalogue digest;
   - any Grant is active, `bearer` or `oauth`, holds an envelope, and was issued
     for the generation's endpoint, whose stored URL matches its digest;
   - the parent Run is `running` and uncancelled under this worker's live lease
     at the claim's fencing epoch, which a parent call carries; a Child Session
     call needs its own live, uncancelled child lease at its `ToolRuntime`
     fencing epoch; and the Agent Session lease matches;
   - exactly one committed `ToolEffectPending` exists for
     `(execution_scope, intent_id)`, matching the call's tool name, call id,
     `never` replay policy, sequential contract, contract version, schema
     digest, argument digest, and stored arguments.
4. The endpoint is checked against network policy again. For a Connection with a
   Grant, the credential is decrypted, an expired OAuth access token goes
   through the [refresh preflight](#token-refresh), and the whole gate runs a
   second time.
5. The worker checks cancellation, refuses a second call for the same owner,
   Run, execution scope, and intent, and calls the tool on a fresh SDK session.

Revoke, disable, and delete lock the same head and Grant rows as the gate. If
revoke commits first, the gate rejects and no request is sent. If the gate
commits first, the call is in flight, even if its request leaves the process
after the revoke.

No database transaction spans network I/O. While a call runs, a watcher checks
every 0.25 seconds, or on NOTIFY, that the head is still enabled at the same
activation epoch with consent, that the Grant is still active, and that the Run
and Child fences still hold; a same-Grant token refresh or re-encryption is not
a change. Losing any of these, a timeout, cancellation, or shutdown cancels the
local task, with no rollback and no automatic retry.

There is no shared connection, session, DNS, or credential cache. Every
discovery and call opens a new SDK session on a new transport without
keep-alive, and admits its target per request.

## OAuth and credential lifecycle

The locked SDK's `OAuthClientProvider` runs PKCE and state, resource
validation, client registration, token exchange, and refresh, so authorization
works only with providers that interoperate with it. DlightRAG supplies the
product integration:

1. Settings calls `POST .../oauth` for one owned Connection at the current
   revision. The callback is `/web/oauth/connections/mcp/callback` on the
   origin that request reached, which the Web's same-origin guard ties to the
   browser's own, so the provider sends that browser, with its session, back
   there. `oauth_callback_url` overrides it, for example for a provider that
   registers one fixed URI; the override keeps that path, uses HTTPS except on
   loopback, and has no query or fragment. DlightRAG never calls the callback,
   so the outbound network policy does not apply to it. The request fails
   before contacting any provider unless the callback is valid and the key ring
   can encrypt.
2. A flow row records the initiating worker as `flow_owner`, the endpoint, the
   Connection head's revision, and a lifetime of `oauth_timeout`. The initiator
   renews a 10-second lease every 2 seconds. Starting again supersedes the
   Connection's pending flow.
3. The SDK discovers the protected resource and authorization server, registers
   or identifies the client, and calls the redirect hook. The hook accepts one
   redirect per flow, so a later step-up needs a new flow. It validates the
   authorization URL (no userinfo or fragment, HTTPS when required, exactly one
   `state`, an admitted host), records the SHA-256 of `state`, takes the
   requested `scope` as the consented scopes, and hands the URL to Settings. The
   request waits at most `discovery_timeout` for it.
4. `GET /web/oauth/connections/mcp/callback` requires the authenticated owner.
   Web middleware moves the query string into request state before any logging;
   upstream proxies and external tracing record it unless the deployment
   redacts it there. The handler finds a live flow by owner and state hash,
   encrypts `{state, code, iss, error}` once, deposits it, and redirects to
   `/web/?settings=connections`, adding `&authorization=restart` on failure. The
   callback may reach any worker; NOTIFY wakes only the flow owner, whose SDK
   callback hook consumes the deposit exactly once.
5. After the token exchange, the SDK finishes discovery with the new token.
   DlightRAG then publishes the Grant, holding the encrypted tokens, client
   information, expiry, and authorization-server and protected-resource
   metadata, together with a generation for the new catalogue, and marks the
   flow `succeeded`.

The SDK's pending PKCE verifier and state live only in the initiating process.
If that worker dies or its lease lapses, no other worker resumes the exchange;
the flow fails, and Settings asks the user to authorize again. A token whose
scope exceeds the consented scopes is rejected, so broader scope always takes a
new Settings authorization and a new Grant. Each worker runs at most four
authorizations, and an owner holds at most four live flows and 128 unexpired
ones; beyond these, the request fails with 429.

Publication in step 5 is a CAS on the authorized Connection's own head, not on
the owner revision. Any command on that Connection after the flow began (an
edit, enable, disable, delete, revoke, probe, or bearer save) moves the head,
and the owner's later word stands: the flow publishes nothing and ends as
`changed`. Commands on the owner's other Connections leave the head alone, and
so does background refresh, because the flow takes the head's refresh claim
from any running refresh and holds it until the flow finishes or, if its
initiator dies, until `oauth_timeout` runs out.

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
state. Its `client_id` is the callback's HTTPS origin with the fixed path
`/web/oauth/connections/mcp/client-metadata` and no userinfo, query, or
fragment, and it equals the document URL exactly. A callback that is not HTTPS
publishes no document (the path returns 404), and its Connections register
dynamically. Pre-registered client credentials are not supported.

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
  sent. Any failure of the token request is an authentication failure
  (`needs-auth`): a missing refresh token or token endpoint, a rejection, a
  scope beyond the consented scopes, a timeout, an admission refusal, or a
  network or server error. `discovery_timeout` bounds the whole preflight, and
  refresh never enters discovery, registration, or consent.
- Saving the new token is a CAS on Grant id, active status, lease owner, live
  lease, refresh epoch, and expected secret version; losing the CAS discards the
  token.
- A 401 or 403 after an effect never triggers refresh-and-replay.

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
  matching an `allow_private_hosts` pattern is exempt; that grants network reach
  only, never remote-account authority.
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
- A result may hold at most `max_result_parts` parts, all text; image, audio,
  and resource parts fail the call. Structured content is appended as JSON, the
  bearer value is redacted, and text larger than `max_result_bytes` fails the
  call. A remote error result becomes a failed call without its remote text.
  Accepted text follows the preview-or-spill rule shared by injected tools; with
  Agent execution `disabled` there is no Agent Workspace to spill to, so an
  over-limit result reports its full output as unavailable.
- SDK and HTTP-library log records are suppressed inside these sessions. Logs
  and Run events carry redacted categories, never remote text or secrets.

`answer.agent.connections` holds the non-secret policy, tunable within these
hard limits. Exceeding a size or count limit fails the request, call, or
discovery candidate; nothing is truncated.

| Field | Default | Range | Bounds |
|---|---|---|---|
| `oauth_callback_url` | unset | ≤ 2048 characters; HTTPS except on loopback | Optional override of the callback, which otherwise follows the address the browser reached |
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

- Every command carries `expected_revision`, a digest of each head's
  `(connection_id, revision)`; a stale revision returns 409. Every command that
  writes a Connection, including a probe and a completed OAuth authorization,
  and every published background refresh, successful or failed, changes it;
  refresh claims and call observations do not. An endpoint edit and a bearer
  save that includes the endpoint carry the revision they started with through
  discovery, so a refresh published meanwhile fails them too; an OAuth
  authorization instead completes against its own Connection's head. Settings
  polls every 5 seconds and asks for a reload after a failed command.
- Mutations need the Web session plus the CSRF double-submit header and
  same-origin checks, in `none` mode as well. Validation errors return a generic
  422 that echoes no input, and another owner's `connection_id` returns 404.
- `POST .../oauth` and the callback's 303 redirect send
  `Cache-Control: no-store` and `Referrer-Policy: no-referrer`. The callback
  relies on the authenticated owner and SDK state instead of CSRF, because a
  provider redirect cannot carry a same-origin token. The metadata document is
  the only public Web path besides login and logout.

Each Connection in the projection carries `connection_id`, `label`, `endpoint`,
`enabled`, `authentication` (`none`, `bearer`, or `oauth`), `status`
(`disabled`, `refreshing`, `ready`, `degraded`, `needs-auth`, or `revoked`),
`authorization_status` of its latest OAuth flow (`pending`, `succeeded`,
`failed`, `changed`, or null), `activation_epoch`, and `generation`. `enabled`
is authoritative, and `status` is the last observation. No management result
carries a token, authorization code, client secret, envelope, catalogue, tool
name, schema, or raw error kind: Settings answers whether a server is reachable
and authorized, and the Agent is the only consumer of what the server offers.

In Settings:

- Turning a Connection on shows the whole-Connection warning once per Settings
  session, and not at all when the owner already has an enabled Connection;
  every enable carries `consent_version=1`. A Connection whose status is neither
  `ready` nor `refreshing` is probed first, and a probe that does not end
  `ready` leaves it off.
- The bearer field is write-only and saves with the current endpoint. OAuth
  shows a link to continue at the provider, and authorization must finish in the
  same session. A failed or expired authorization, and one whose Connection
  changed meanwhile, each get a note asking to authorize again. Only an
  unauthenticated Connection edits its endpoint in place.
- Delete asks for confirmation and removes the endpoint, label, and credential.
  Settings offers no revoke action, because delete already retires the Grant and
  erases its ciphertext; the `revoke` route serves a surface that keeps the
  configuration while destroying only the credential.

Tool Activity is the one place outside Settings that names a remote tool: when
the browser subscribes to a Run's events, each pinned tool's events gain
`Connection label · remote tool name`, collapsed to one line of at most 96
characters. The label is display only and exposes no schema, endpoint,
credential, or Connection id. It is never stored, because the durable event
keeps transport-neutral identity, and it survives a later disable or delete
because the head row stays until its pins are gone.

## Fault behavior

- Before the first gate commits, an oversized argument, a gate denial, or an
  expired `call_timeout` sends nothing, records no status, and returns a failed
  result saying that no call was sent. A repeated call, in the same process, for
  an intent the restored tool already ran returns a failed result saying that
  the outcome may be unknown, before any gate, and records nothing.
- After the first gate, any failure returns a failed result saying that the
  outcome may be unknown: the endpoint recheck, credential decryption, a token
  refresh, the second gate, the call itself, a timeout, or cancellation by the
  in-flight watcher or shutdown. If the Run's own cancellation or lease check
  fires first, the Run stops without receiving a result.
- Every exit after the first gate tries to record an observation: `ready` after
  a successful call; `needs-auth` after an authentication failure, which is HTTP
  401 or 403, a credential no key opens, or a failed token request; and
  `degraded` after anything else. That includes transport and protocol faults,
  an oversized or unsupported result, a remote tool error or missing tool, a
  missing key ring, a refresh preflight that exceeds `discovery_timeout` or that
  the gate refuses, and Run cancellation, lease loss, or shutdown. Recording is
  best effort: it gets two seconds, and a failure to record only logs a warning.
- An observation is recorded only while the call's generation is still the head
  generation, the head is still enabled at the same activation epoch, and the
  Grant is still active at the same secret version. A call stopped by revoke,
  disable, delete, or a replaced credential therefore records nothing.
- A failed call leaves pinned definitions in place, and other Connection and
  built-in tools stay callable. The failed result names the local tool, says
  whether no call was sent or the outcome may be unknown, forbids automatic
  retry, and tells the final Answer to identify the unfinished part. A remote
  rejection of an older pinned schema is such a failure; the Run never
  rediscovers or switches generation.
- Store failures and pin or digest inconsistencies are internal errors and can
  fail the Run; an ordinary remote fault does not.

## Operational lifecycle

- Readers and writers both run refresh loops and owner commands; only writers
  run maintenance, at startup and then at most 60 seconds apart
  (`min(60, refresh_seconds)`). Each pass re-encrypts up to 100 live Grants
  under a retired key the ring still holds, deletes up to 100 expired OAuth
  flows, and collects up to 100 generations or deleted heads. A Grant no key
  opens is skipped and never stops collection.
- Re-encryption is a CAS on secret version and envelope that skips a Grant under
  a live refresh lease, at selection and again at the CAS, so it never
  overwrites a rotated refresh token, and a zero count during a lease does not
  prove that no old-key ciphertext remains.
  `Application.connections.maintain()` runs one pass on demand and returns only
  the `reencrypted` and `collected` counts.
- Shutdown stops refresh, maintenance, notifications, and pending
  authorizations before the Run coordinator drains, then cancels outstanding MCP
  calls.
