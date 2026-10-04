# Security

This document owns authentication, identity-provider boundaries, authorization,
ingress responsibility, resource handling, and browser Artifact isolation. Field
defaults live in [Configuration](configuration.md); public payloads in
[Interfaces](interfaces.md).

DlightRAG verifies credentials and maps claims to workspace/actions. It does
not issue OAuth tokens, manage users/passwords, or provide an identity-provider
login system.

Examples below are YAML. Each setting is also `DLIGHTRAG_ACCESS__<FIELD>`, which
is where a deployment of the checked-in `config.yaml` keeps it (see
`.env.example`), so its issuer, audience, and people stay out of the repository.

## Authentication Modes

| Mode | Intended use | Owner |
|---|---|---|
| `none` | Loopback development only | The deployment owner |
| `simple` | One owner's deployment behind a bearer token | The deployment owner |
| `jwt` | Externally issued, user-scoped signed tokens | One per issuer and subject |

An owner holds Sessions, Runs, Profile Memory, and Personal MCP Connections.
`none` and `simple` admit every caller as the same deployment owner, so switching
between them keeps that owner's data. Use `jwt` when several people share a
deployment.

A non-loopback REST/MCP listener with `none` is refused unless
`access.allow_insecure_no_auth: true`. CORS stays closed: the Web is same-origin,
so name origins in `access.cors_allow_origins` only for a cross-origin browser
client.

### Simple Bearer

```yaml
# config.yaml
access:
  auth_mode: simple
```

```bash
# .env or an orchestrator Secret
DLIGHTRAG_ACCESS__API_TOKEN=<generated-by-openssl-rand-base64-32>
```

Clients send `Authorization: Bearer <generated>`. Whoever holds the token is
the deployment owner, with its Profile Memory and its Connections, including the
external accounts those Connections authorize: treat the token as that owner's
password. `X-User-Id` on REST only names the actor in audit records and never
selects an owner. `simple` is admission control, not multi-user authorization.

### Static JWT

JWT mode requires `sub`, which becomes `user_id`.

```yaml
# config.yaml
access:
  auth_mode: jwt
```

```bash
# .env or an orchestrator Secret
DLIGHTRAG_ACCESS__JWT_VERIFICATION_KEY=<key-or-public-pem>
```

A static key is a shared `HS256` secret unless `jwt_algorithm` names another;
name `RS*`/`ES*` for an issuer public-key PEM. Issuer/audience claims are
validated when configured. DlightRAG never signs, renews, or mints tokens.

### Published Keys (OIDC)

Prefer an issuer's published, rotating keys (Entra, Okta, Auth0, Keycloak,
Cloudflare Access, Cognito, and similar):

```yaml
access:
  auth_mode: jwt
  jwt_issuer: https://login.example.com/tenant/v2.0
  jwt_audience: api://dlightrag
```

DlightRAG reads the key set from the issuer's OpenID discovery document
(`<issuer>/.well-known/openid-configuration`), which must name this exact
issuer, and verifies each token with the algorithm its key names. Set
`jwt_jwks_url` only for an issuer without discovery, and `jwt_algorithm` only to
pin one algorithm. Published keys verify any token their issuer signs, so
`audience` is required with them. It may be one value or a list; any match
passes.

### MCP OAuth Discovery

DlightRAG can advertise its MCP HTTP listener as an OAuth 2.1 resource server;
the external issuer still authenticates users and issues tokens.

```yaml
access:
  auth_mode: jwt
  jwt_issuer: https://auth.example.com
  jwt_audience: api://dlightrag-rest
interfaces:
  mcp:
    transport: streamable-http
    resource_server_url: https://rag.example.com/mcp
```

This publishes RFC 9728 metadata and a `WWW-Authenticate` discovery challenge.
The externally reachable URL cannot be inferred from a bind address. For native
MCP OAuth it is also the exact expected token audience, independent of broader
REST/Web audience settings. Omit it when MCP clients already hold directly
supplied JWTs.

## Edge-Asserted Web Identity

Web can verify a credential forwarded by an authenticating edge. REST and MCP
continue to verify their own bearer JWTs and never accept edge assertions.

```yaml
access:
  auth_mode: jwt
  jwt_issuer: https://<team>.cloudflareaccess.com
  jwt_audience: <application-aud-tag>
  web_identity:
    edge: cloudflare        # cloudflare | azure | aws
```

The edge only decides where the Web finds the token. The Web verifies it like a
REST bearer, but only against an issuer's published keys, never a static key.
`web_identity.issuer` and `.audience` default to the API's, and `.jwks_url` to
the API's key set while the Web keeps the API's issuer. Set them only when the
edge's tokens differ, such as an Azure ID token whose audience is the App
Registration client ID.

| Edge | Verified credential | Its issuer and audience |
|---|---|---|
| Cloudflare Access | `Cf-Access-Jwt-Assertion`, fallback `CF_Authorization` JWT cookie | Team domain + application AUD |
| Azure Easy Auth | `X-MS-TOKEN-AAD-ID-TOKEN` | Entra issuer + App Registration client ID |
| AWS Amplify/CloudFront | Forwarded `Authorization` bearer | IdP issuer + app client ID |

Owner identity is `(iss, sub)`, so changing issuer creates a different owner even
for the same human. Azure's unsigned `X-MS-CLIENT-PRINCIPAL` is never read.
Missing/invalid edge credentials return 401; DlightRAG renders no login page.

The origin must accept traffic only from the configured edge. Cryptographic
verification does not make arbitrary header injection safe. State-changing Web
routes also require exact same-origin `Origin` and a double-submit
`dlightrag_web_csrf` cookie echoed as `X-CSRF-Token`; a refused request gets
403 in the shared error envelope with `error_kind: "cross_origin_rejected"`. A
proxy that does not forward the browser's `Host` refuses every write this way.

### Entra Example

Use one API App Registration with access-token version 2, an exposed delegated
scope, and optionally App Roles assigned to groups.

```yaml
access:
  auth_mode: jwt
  jwt_issuer: https://login.microsoftonline.com/<TENANT_ID>/v2.0
  jwt_audience: <API_CLIENT_ID>
```

Common mistakes:

- v2 tokens use the `/v2.0` issuer and client-ID GUID audience; v1 differs.
- App Roles provide stable strings in `roles`; raw `groups` contains object IDs
  and can overage around 200 groups.
- A client must request the exposed API scope to receive the API audience.

Map App Roles through `access.control.rules`, described below.

### Cloudflare Access Example

Each person signs in to Access with a one-time code sent to their email, and the
Access application lists exactly the emails that may. Access then signs a JWT for
that person (`sub` is their stable Access identity, `email` their address) and
forwards it on every request as `Cf-Access-Jwt-Assertion`. The Web verifies it
against the team's keys, so every email is its own owner.

```yaml
access:
  auth_mode: jwt
  jwt_issuer: https://<team>.cloudflareaccess.com
  jwt_audience: <application-aud-tag>
  web_identity:
    edge: cloudflare
```

- Cover `/web` with the application and leave `/static` public; it holds only the
  Web's assets.
- Add a Bypass application for `/web/oauth/connections/mcp/client-metadata`: an
  authorization server fetches it without an Access session.
- REST and MCP clients present the same person's token as a bearer
  (`cloudflared access token -app=https://<host>/web`). The `jwt_*` settings
  verify it, so those clients act as that person too.

## Ingress Responsibilities

The application enforces semantic invariants:

- authentication, authorization, and owner scope;
- idempotency and changed-input conflict;
- URL redirect/DNS/SSRF checks and upload/fetch byte, archive, part, and pixel
  bounds;
- a streaming receive cap before parsers;
- durable lease fencing;
- client/model output sanitization; and
- provider concurrency and per-call timeouts.

Ingress owns TLS/certificates, DDoS and volumetric protection, WAF signatures,
IP/geo/bot policy, request quotas/rates, and connection caps. DlightRAG ships no
in-process WAF or rate limiter. SIEM systems such as Sentinel observe/correlate;
they are not an inline blocker.

Accepted Retrieval and Answer Runs queue rather than fail under local worker
saturation, up to the deployment-wide Query-lane nonterminal admission limit.
Corpus Mutation Runs use a separate limit, and each writer process executes a
bounded number of Corpus Mutations concurrently; both are
[configuration](configuration.md#runruntime-lanes-and-retention), and
deployment configuration owns process count and total active capacity. The
controlled admission-limit, authorization, sanitation, and 10k-client evidence
is recorded with the [RunRuntime targets](run-runtime.md#load-evidence).
Monitor PostgreSQL/blob growth and rate-limit acceptance before either admission
limit.
`none` and `simple` collapse callers into one deployment owner and require an
already restricted network boundary.

`GET /health` and `GET /ready` are unauthenticated by design. Health performs
no database, corpus, parser, or model I/O. Readiness short-caches only the
writable Operational State verdict; it does not probe corpus or provider state.
Both surfaces expose only fixed component details, storage class names, and the
confinement state an Agent's processes run under on this host (ADR 0024), never
credentials, endpoint URIs, or inherited environment values.

### Per-Surface Front Doors

REST and Web share the API process (default port 8100); MCP uses a separate
listener (8101). A browser proxy may front only `/web` and the `/static` assets
it loads, while direct REST/MCP clients supply their bearer tokens. A proxy that
terminates TLS must be trusted for `X-Forwarded-Proto` (uvicorn's
`FORWARDED_ALLOW_IPS`): otherwise the API sees `http`, the Web's exact
same-origin checks refuse every browser write, and the OAuth callback is
addressed over `http`. Compose trusts loopback and private networks, where a
proxy on the same host connects from. Bearer tokens verify under the `jwt_*`
settings; an edge token whose audience differs sets `web_identity.audience`.
Native MCP OAuth still requires the exact public MCP resource URL as audience.

## Authorization Model

Authentication asks who; authorization asks which product action on which
workspace.

```text
verified JWT claims
  -> deployment Access Rules, and the Workspace's creator
  -> canonical Workspace + Action
  -> allow or deny
```

A rule matches claim name/value, workspace pattern, and action pattern. Rules,
and the creator grant below, combine with OR semantics; there are no deny rules,
and no allow match means deny.

```yaml
access:
  auth_mode: jwt
  control:
    rules:
      - claim: roles
        value: finance.editors
        workspaces: [finance]
        actions: [editor]
      - claim: roles
        value: legal.readers
        workspaces: [legal]
        actions: [reader]
```

Rules require JWT auth, and any rule puts every action under rules. Without
rules every authenticated caller holds every action, as in local development.
Claim values may be strings or list members. Workspace patterns
are a canonical ID or `*`. Action patterns may be exact, `*`, a prefix such as
`workspace.*`, or a preset.

### Workspace Creators

Each workspace records the owner that created it. Once rules apply, its creator
holds `editor`, `workspace.reset`, and `workspace.delete` on it beyond what rules
grant, so a person sees and changes the workspaces they create, and others only
where rules grant them.
Operator facts (`workspace.storage_status`) and deployment-wide actions stay with
rules. A workspace the deployment registers itself, such as the default, has no
creator. Workspace ids remain deployment-wide: creating a name someone else holds
is refused as existing.

One administrator, a default everyone reads, and workspaces of their own for
everyone else:

```yaml
access:
  control:
    rules:
      - {claim: email, value: admin@example.com, workspaces: ["*"], actions: [admin]}
      - {claim: iss, value: "https://<team>.cloudflareaccess.com", workspaces: [default], actions: [reader]}
      - {claim: iss, value: "https://<team>.cloudflareaccess.com", workspaces: ["*"], actions: [workspace.create]}
```

A Corpus Mutation Run shows to whoever submitted it. Others reach it through its
workspace, and a workspace its creator holds shows them only the Runs that creator
submitted, so a deleted workspace's history never passes to whoever creates the
same name next. Seeing a Run never lets anyone cancel or resume it: that needs the
Run's action on its workspace now. The Web offers on each workspace only the
changes its caller may make.
[ADR 0031](adr/0031-a-workspace-creator-holds-it.md) records the decision.

### Actions And Presets

| Action | Meaning |
|---|---|
| `workspace.query` | Retrieve/answer |
| `workspace.ingest` | Start ingestion/replacement/retry, including repair resume for those actions |
| `workspace.list_files` | List files |
| `workspace.delete_files` | Delete files, including repair resume for deletion |
| `workspace.download_source` | Download retained source |
| `workspace.read_metadata` | Read metadata |
| `workspace.update_metadata` | Update metadata |
| `workspace.read_visual_asset` | Read rendered visuals |
| `workspace.create`, `.reset` | Workspace creation and identity-preserving Corpus Reset, including repair resume/supersession for reset |
| `workspace.delete` | Workspace Delete, including repair resume for it; unknown mutation actions fail closed to this action |
| `workspace.storage_status` | Read storage/promotion state |
| `model_catalogue.write` | Change deployment-wide model catalogue |

| Preset | Expansion |
|---|---|
| `reader` | query, list/download files, read metadata/visual assets |
| `editor` | reader + ingest and metadata/file mutation |
| `admin` | every action |

Presets affect only Actions, never Workspace matching. Deployment-wide actions
require `workspaces: ["*"]`.

### Source Of Truth And Revocation

The IdP owns users/claims; deployment configuration owns claim-to-workspace
rules. DlightRAG stores no users, custom roles, invitations, or membership ACLs;
it records only which owner created each workspace.
PostgreSQL is trusted application storage without row-level security; this is
not database-enforced tenant isolation.

Explicit workspace requests are checked before acceptance. `all_workspaces`
expands only to currently authorized query workspaces. Source/visual routes
recheck permission against the actual workspace.

An accepted Retrieval or Answer Run pins its resolved Workspace set for
execution, not mutable claims. Later rule or IdP changes do not revoke execution
of that Run; follow-up/fork recheck current access. Corpus Mutation status and
events recheck that the caller may see the Run, through its declared Workspace or
as its submitter; cancellation and repair resumption also recheck the Run's
action on that Workspace. All fail closed. REST status/event projection and MCP
status projection recheck current source-download and visual-asset actions;
canonical results never persist authorization-dependent URLs. Trusted Application
callers supply their own Retrieval projection. JWT changes become visible when a
new token arrives, so use short lifetimes where revocation latency matters.

Use a policy/membership store for user-managed, deny, hierarchy, or resource-level
policy. Use separate deployments/databases or PostgreSQL RLS where regulation
requires database-enforced tenant separation.

## Source Download Boundary

Public source payloads expose stable `source_uri` and, on HTTP surfaces, a
projected `download_url` containing only document ID and workspace. They never
expose local paths or stored locators. REST/Web download routes recheck
`workspace.download_source` before streaming retained bytes or redirecting to
Azure, S3, or queryless HTTPS. Authorization is necessary but not sufficient:
the metadata row must also carry `_dlightrag_finalization_complete=true`.
Guessed IDs for pending, failed-finalization, legacy-unproven, or direct
LightRAG-bypass documents return 404. Full and thumbnail image routes enforce
the same rule, including on cache hits, and answer `Cache-Control: private,
no-store` so no shared cache replays one caller's authorized image to another.

Signed/query-bearing URLs are fetch credentials, not durable locators. Retain
the bytes or provide a separate queryless `download_uri`; signed queries never
become public provenance or source-contract logs.

Durable Corpus Mutation Runs retain the bounded accepted fetch input, upstream
handoff checkpoint, and operator repair confirmation needed for recovery until
the fixed seven-day terminal retention floor permits pruning. Treat Operational State as secret
storage and restrict its access.

## Answer Resources And Execution

Attachments are bounded by count, per-item bytes, total bytes, archive expansion,
and decoded pixels before orchestration. Link resources and agent-supplied
`read(url=...)` targets admit only anonymous HTTP(S) without embedded credentials,
signed query credentials, fragments, localhost/private destinations, or unsafe
redirects. The shared egress boundary repeats scheme/host/DNS/SSRF checks at
every redirect, pins the validated address for each connection, and never permits
HTTPS to downgrade to HTTP. Agent reads can vary only `User-Agent`, `Accept`, and
`Accept-Language`; cookies, authorization, arbitrary headers, and browser sessions
are unavailable to `read`. The Agent's only sessions are the Run-scoped Agent
Browser's, whose pages are anonymous contexts
([Agent Browser Boundary](#agent-browser-boundary)). A successful acquisition becomes
one immutable run snapshot.

MarkItDown runs without plugins/network. OOXML files pass central-directory
zip-bomb checks before conversion. Full bytes never enter model context—only
bounded text windows, safe observations, and budgeted images. Only Evidence
ledger entries become citable; Profile Memory, Skills metadata, and incidental
child summaries do not. Public Child Session observation projects attributable
status, a bounded transcript, queued versus consumed controls, questions, and
Evidence handles. It strips host_state, pinned plan/budget, context snapshots,
fencing, and provider-private reasoning. Skill text is untrusted reference
context: loading `council` or any Skill cannot widen host-permitted child tools.
A child inherits its parent's tools except the Run's authority — roster controls,
durable owner memory writes, and publication. A parent that wants a narrower child
lists the tools it may hold, and a name the Run does not offer is left out, never an
error; the built-in `council` Skill asks for no narrowing, so its children hold the
default set (ADR 0025). User-cancelled child work cannot resume without an
explicit authorized override. Whole-Run user cancellation cascades to children;
browser/SSE disconnect only detaches the observer.

Admitted bytes are content-addressed within one owner. A fetched resource is
stored only after validation and is linked atomically before its effect settles,
so recovery reads the same bytes. Deduplication never crosses owners.

Only a settled parent-Research `attach_artifact` call authorizes a workspace
root for publication. Fast and Child Sessions do not receive that capability,
and answer or Markdown `artifact:` links grant no authority. At the terminal
boundary, the Host rechecks the attachment's raw digest and size, validates its
safe dependency closure, and fails closed for missing, stale, or unattached
roots. Published descriptors and bytes remain owner/run scoped. See
[ADR 0004](adr/0004-structured-artifact-attachment-authority.md).

Execution modes:

- `disabled`: no local execution tools;
- `trust`: rooted file tools, and every Agent process runs under a kernel-enforced
  allow-list — the Agent Workspace plus the runtime the toolchain needs and the
  layers capabilities declare. The corpus tree, the deployment's configuration, the
  project tree, and other Runs' workspaces stay outside that view
  ([ADR 0024](adr/0024-the-agent-sees-only-its-workspace.md)). Bash keeps the service
  user's network authority, which is the deployment's to enforce with a network
  policy rather than a path list.

Root checks are not a shell sandbox. Research reaches outside tools only through
its owner's Personal MCP Connections, pinned per Run and gated per effect, with
in-flight writes cancelled best-effort and never replayed
([contract](personal-mcp-connections.md)), and through the deployment's Agent Browser
([boundary](#agent-browser-boundary)). Network policy can deny access but
cannot grant external account authority. Public Web Search exists only for
configured Exa/Tavily provider chains, and Extract for those chains and the
deployment's Agent Browser, which no owner authorizes and the Agent reaches through
`read` and the `browser` tool; provider failures may fail over, while successful empty
results do not.

All Agent/child/Fast mutations are fenced by owner, run lease, epoch, and
register sequence. A completed child outcome is persisted so replay cannot
re-enter it. A staged Fast result replays without another model call.

## Agent Browser Boundary

The Agent Browser ([ADR 0032](adr/0032-the-agent-browser.md)) renders a public page for
a Research `read` and lets the `browser` tool drive one, in a Chromium that runs in a
pool container, outside the answering process. The Agent gains no process, path, socket,
or binary from it, and an owner authorizes nothing: it is a deployment capability, not a
Connection. Fast never reaches it. What the browser loads is untrusted, so the boundary
is the deployment's network, which fails closed, and DlightRAG checks only the URLs it
hands over and accepts back.

- **URLs.** The first URL passes the direct read's rules before the browser sees it:
  an HTTP(S) scheme, no embedded credentials or credential query parameters, and a host
  that resolves only to public unicast addresses. The URL the page ends at is checked for
  the same form, without a lookup, and a page that ends at one the rules refuse fails as
  `final_url_refused` with nothing admitted. Everything between, redirects,
  subresources, a script's requests, and a page's own navigations, is confined by the
  network below, not by DlightRAG. The `browser` tool's `navigate` checks the scheme,
  embedded credentials, and public resolution, before any browser is leased, and a URL
  they refuse ends the call with the reason. It leaves out the credential-parameter rule,
  which guards what becomes an Agent Resource, not where a page may go: a verification link
  from a mailbox often carries such a parameter.
- **Topology.** Each pool member sits alone on an `internal: true` network that has no
  route out and none to any other member. The egress proxy is on the default network
  and on every member network, and so are `dlightrag-api`, `dlightrag-mcp`, and
  `dlightrag-reader`. Members are kept apart because the server runs with `--unsafe`,
  which lets any client that can connect choose a browser's launch arguments and
  executable: only DlightRAG's processes may reach a member, and a compromised member
  must not be able to drive another. How to add a member without breaking that is in
  [Operations](operations.md#agent-browser-pool).
- **Egress.** Every browser launch carries the proxy, a Squid container
  (`agent-browser/egress/squid.conf`) that admits public destinations only: it denies
  loopback, RFC 1918, link-local (where cloud metadata lives), CGNAT, multicast,
  reserved, and documentation ranges and their IPv6 equivalents, the same set as
  `network_admission`, and admits ports 80 and 443 with `CONNECT` only to 443. The
  launch carries no bypass list, so loopback requests go through it too, and the
  connection never uses Playwright's `expose_network`, which would route browser
  traffic back through the application's own network. Every context, a render's and an
  Agent Page's alike, is made by one function in the adapter that passes no proxy
  option of its own, so each inherits the launch proxy. A missing or wrong proxy setting
  therefore reaches nothing beyond the member's network. Denials appear as `TCP_DENIED`
  in the proxy's log.
- **Sessions.** Every render uses a temporary anonymous context with no cookies,
  storage, or service workers, and downloads off. Each Agent Page is an anonymous context
  of its own in the Run's browser, empty at the start and with service workers blocked,
  with downloads on only so the tool can read them. Cookies and storage never outlive the
  Run, and two Agent Pages never share any
  ([when one closes](architecture.md#agent-browser)). The browser is launched for one
  connection and closed with it, and the lease gives a pool container to one Run at a
  time. The browser holds no credential of the owner.
- **Downloads.** A download is copied over the Playwright protocol into a temporary
  directory made for that file alone, under a fixed name that neither the page nor the model
  supplies, and read from it. The pool's copy is cancelled and deleted whatever the
  outcome, and the directory is removed when the copy ends. What a call admits, and the
  caps on a download's size, time, and number, are in
  [Resource reading](resource-reading.md#browser-captures-and-downloads).
- **Upload.** `upload` exists only where the Run has an Agent Workspace (`trust`). It reads
  regular files the workspace tools could read, through the same path rules and the same
  integrity latch, checked again once it holds the file, and hands them to a file input, at
  most 50 MiB in a call, measured before any file is read: it sends workspace files to
  whatever page asks for them, so it is as much authority as the page can borrow from the
  model's judgment. A Child holds it as its parent does.
- **Chromium's sandbox.** The container runs as the unprivileged `pwuser` under
  Playwright's recommended seccomp profile, with an init process and memory and process
  limits. `answer.agent.browser.chromium_sandbox` (default `true`) is whether each launch
  asks for Chromium's own sandbox, which the server honors only because it runs with
  `--unsafe`; without that flag, or with the setting `false`, Chromium runs with
  `--no-sandbox`. Whether a host can start the sandbox depends on its user namespaces and
  the seccomp profile; the operator states it, and DlightRAG does not probe for it. A host
  that cannot start it fails every launch and the pool is unreachable until the host is
  relaxed or the setting is `false` ([Operations](operations.md#agent-browser-pool));
  nothing runs unsandboxed on a guess. With `false` the container and its network are the
  isolation boundary. `GET /health` reports the setting.
- **Evidence.** Rendered and captured text is the browser's assertion. DlightRAG attests
  the binding between the returned page, the Resource Handle, and the URL; it does not
  attest that an anonymous GET serves the same page, and a site may serve a browser what it
  does not serve a client. The acquisition `browser_render`, `browser_capture`, or
  `browser_download` on every row says which tier produced it. Page text, snapshots, dialog
  messages, and screenshots are untrusted model context, never Evidence, like any fetched
  page: only `capture` and a downloaded file become Resources, and a download is Evidence
  only once it is read. How a Resource is cited, and the rule that keeps a signed or
  script-made URL private, are in
  [Resource reading](resource-reading.md#browser-captures-and-downloads).
- **Verification walls.** A CAPTCHA or any other human-verification check surfaces as an
  HTTP error or as page text. The `browser` tool's description tells the model to stop that
  path and report it. DlightRAG never solves, bypasses, or outsources one and holds no
  solver integration; this is a rule the model follows, not a mechanism, because nothing
  detects a CAPTCHA, and a vision model could read one from a screenshot.

Residual risks, recorded rather than solved:

- A page can try to steer the Agent through what it says, and the browser gives it forms
  as well as URLs to act through. As with Bash, the boundary is the deployment's egress,
  not a filter.
- A frame names the full URL of the page, so a token-bearing verification link appears in
  the result the model reads and in the Run's record of it.
- The application services share each member network with it, so a compromised pool
  container can reach their listeners. With the development default
  `access.auth_mode: none` those listeners are unauthenticated; a deployment that renders
  pages for untrusted callers sets access ([Authentication Modes](#authentication-modes)).
- UDP, WebRTC included, is not proxied. The member networks give it no route out.
- Squid resolves and checks a destination itself, unlike the direct read, which pins
  the validated address for its connection. The window between Squid's check and its
  connection is small but not zero.
- A pool container serves one Run at a time by lease, not one Run in its lifetime: a
  renderer compromise that outlives its browser can meet the next Run that leases the
  container. The server limits no clients, so a connection of a holder whose lease has
  expired, a worker that stalled rather than died, may still be open when the next Run
  leases the container, and the two browsers then share it until that connection ends.

## Answer Artifact Browser Boundary

Artifact descriptors expose validated media and owner-scoped URLs, never raw
blob bytes or Agent Workspace paths. Authenticated data/Markdown uses
`Cache-Control: private, no-store`; active/unknown formats download with
`nosniff`.

Published SVG is sanitized of scripts, handlers, external loads, and nested SVG
data URLs, then served under CSP sandbox. A Run's stored SVG Resource, such as a file a
page downloaded, is served inline only under the same sandbox policy. PDF preview is
sandboxed without same-origin capability.

HTML never executes as a same-origin document. After explicit consent, the
browser inserts authenticated inert bytes into one `srcdoc` iframe with
`sandbox="allow-scripts"` but without same-origin, forms, popups, downloads,
frames, workers, storage, device permissions, or application bridge. A prepended
CSP blocks normal fetch/subresource paths. The wrapper's only parent signal is a
private one-way Escape-close token removed before Artifact code executes. Close
or switch destroys the iframe.

This isolates DlightRAG cookies, storage, and authenticated DOM. It provides no
CPU/memory quota, executes no server code, and is not an absolute browser-egress
guarantee. Chromium is the active-preview security regression baseline.

## Reader-Activated External Video

A recognized prose link may offer a separate official player (ADR 0028), even
without a preview card. Recognition is local; only the reader's Play action
invokes the authenticated, CSRF-protected resolver. Anonymous oEmbed acquisition
reuses public-HTTP DNS pinning and redirect validation, with a 4-second
admission-inclusive deadline and 64 KiB response cap. Provider HTML never enters
the document: the pinned MIT oembed.com registry supplies publisher endpoints
and permitted domain families. A standard video response with one permitted
HTTPS iframe supplies the player URL and geometry. Small official mappings can
survive missing metadata; they are not the supported-provider list. The browser
checks the response against permissions already projected with the selected
link and refuses the application's own hostname. A registry candidate can still
return non-video or SDK-only content, which stays a link. No paid service or
provider SDK is used.

The resulting cross-origin iframe grants scripts, its own origin, presentation,
autoplay, encrypted media, fullscreen and picture-in-picture—not top navigation,
forms or popups. It sends at most the application's origin as cross-origin
referrer. The provider can observe the reader's IP/browser and its own cookies
subject to browser rules; existing cover images also contact their host before
Play. Selecting another video, removing the Answer or closing its Canvas destroys
the player. Player redirects/subresources are controlled by the provider and
browser, not the backend's DNS-pinned transport; this is not absolute browser
egress confinement. This capability does not relax Markdown sanitization or the HTML
Artifact boundary above, fetch a video into a Run, or rewrite stored Answers.

## Deployment Posture

| Deployment | Recommended posture |
|---|---|
| Local | Loopback REST/MCP + `none` |
| Trusted internal | `simple` behind network restriction |
| Enterprise multi-user | `jwt` with an issuer's published keys; Access Rules for workspace policy |

Public MCP requires non-loopback bind, authentication, and explicit
`interfaces.mcp.allowed_hosts`/`allowed_origins`; browser clients also need
`access.cors_allow_origins`. Host/Origin DNS-rebinding protection remains active
even with bearer auth.

### Personal Connection Authorization

Personal MCP Connections authorize with OAuth PKCE and state, deposit each callback
once into an encrypted inbox, and keep credentials sealed under the deployment key
ring; the [contract](personal-mcp-connections.md) has the details. A callback URL
carries the code and state in its query: the Web strips them before its own
logging, and operators must redact them in upstream proxies and external tracing,
which are outside this application. Removing live ciphertext does not erase
backups.
