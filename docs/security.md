# Security

This page is for operators exposing DlightRAG beyond local loopback. It owns
authentication, token verification, identity-provider boundaries, and
workspace/action access control. Configuration field defaults live in
[configuration.md](configuration.md); interface request shapes live in
[interfaces.md](interfaces.md).

## Security Model

DlightRAG authenticates bearer tokens and can enforce workspace/action
authorization. It does not issue OAuth tokens, manage users, store passwords, or
host a full identity-provider login system.

For enterprise deployments, use an external identity provider or gateway to
authenticate users and issue tokens. DlightRAG verifies those tokens and maps
verified claims to workspace permissions when access control is enabled.
The browser still pastes that bearer into an HttpOnly cookie when no
`access.web_identity` edge is configured; the edge-asserted Web identity path is
described below.

## Edge-Asserted Web Identity

The Web frontend is never bare-exposed: an edge (Cloudflare Access, Azure Easy
Auth, or AWS Amplify/CloudFront auth) authenticates the human, and the Web
surface verifies the edge credential per request — statelessly, with no login
page, no DlightRAG-issued cookie, and no token in any response JSON. REST and
MCP keep verifying their own bearer JWTs and never accept edge assertions.

```yaml
access:
  auth_mode: jwt
  web_identity:
    edge: cloudflare        # cloudflare | azure | aws
    issuer: https://<team>.cloudflareaccess.com
    audience: <application-aud-tag>
```

Per edge:

| Edge | Verified credential | Config |
|---|---|---|
| Cloudflare Access | `Cf-Access-Jwt-Assertion` header (fallback: `CF_Authorization` cookie JWT) | `issuer` = team domain; `audience` = application AUD tag; team certs derived automatically |
| Azure Easy Auth | `X-MS-TOKEN-AAD-ID-TOKEN` (a real AAD ID token) | `issuer` = `https://login.microsoftonline.com/<tenant>/v2.0` (or without `/v2.0`); `audience` = App Registration client id; discovery keys derived automatically |
| AWS Amplify / CloudFront | bearer token the edge forwards in the `Authorization` header | `issuer` = IdP issuer (e.g. Cognito user pool); `jwks_url` required; `audience` = app client id |

The verified token's `iss` and `sub` project the owner exactly like a REST
bearer: a human who used paste-token login and the same human via the edge are
**different owners** (different issuers). Migrating from paste to edge starts
with fresh owner-scoped data.

The unsigned Azure principal header (`X-MS-CLIENT-PRINCIPAL`) is parsed only as
display enrichment and never influences authorization. A missing, expired, or
unverifiable edge credential is `401`; the Web surface renders no login page.

### Edge trust and state-changing routes

The origin must only accept connections from the edge (deployment firewalling:
Cloudflare IP ranges, or the platform's own injection point). Header injection
is otherwise spoofable — the code verifies cryptography, never trust alone.

State-changing `/web` routes are hardened with a JS-readable double-submit
cookie (`dlightrag_web_csrf`) that browsers must echo as `X-CSRF-Token`,
plus exact same-origin `Origin` checks; cookie-authenticated (paste)
mutations always require an `Origin` header, and login POSTs reject
cross-origin browsers. `secrets.compare_digest` compares the token.

## Authentication Modes

| Mode | Use case |
|---|---|
| `none` | Local loopback development only |
| `simple` | One shared bearer token for trusted internal deployments |
| `jwt` | User-scoped deployments with externally issued signed tokens |

`access.auth_mode: none` returns an anonymous user context. Non-loopback REST or MCP
HTTP listeners are refused unless `access.allow_insecure_no_auth: true` explicitly
accepts that risk.

When auth is enabled, replace wildcard CORS origins with explicit origins.
Browsers reject credentialed cross-origin requests with `["*"]`.

## Simple Bearer Token

`simple` mode compares the bearer token with `access.api_token`.

```bash
openssl rand -base64 32
DLIGHTRAG_ACCESS__AUTH_MODE=simple
DLIGHTRAG_ACCESS__API_TOKEN=<generated>
```

Clients send:

```http
Authorization: Bearer <generated>
```

REST can receive `X-User-Id` in simple mode for request scoping. MCP treats the
shared token as one anonymous principal; it does not accept caller-selected
identity. `simple` is admission control, not per-user authorization.

## Static JWT

`jwt` mode verifies externally issued JWTs. Tokens must include `sub`; DlightRAG
uses it as the authenticated `user_id`.

```bash
openssl rand -base64 64
DLIGHTRAG_ACCESS__AUTH_MODE=jwt
DLIGHTRAG_ACCESS__JWT_VERIFICATION_KEY=<generated>
DLIGHTRAG_ACCESS__JWT_ALGORITHM=HS256
```

For `HS*` algorithms, `access.jwt_verification_key` is the shared HMAC key. For
`RS*`/`ES*` algorithms, it is the public key PEM from the issuer. DlightRAG
does not sign, renew, or mint these tokens.

If `access.jwt_audience` is unset, audience verification is disabled. If `access.jwt_issuer`
or `access.jwt_audience` is set, PyJWT validates those claims during token decoding.

## JWKS / OIDC Issuers

For Azure Entra, Okta, Auth0, Keycloak, and other OIDC-style issuers, prefer
JWKS so signing-key rotation is handled by PyJWT's `PyJWKClient`.

```yaml
access:
  auth_mode: jwt
  jwt_algorithm: RS256
  jwt_jwks_url: https://login.example.com/.well-known/jwks.json
  jwt_issuer: https://login.example.com/tenant/v2.0
  jwt_audience: api://dlightrag
```

`access.jwt_issuer` and `access.jwt_audience` are required when `access.jwt_jwks_url` is set.

`access.jwt_audience` may be a single value or a list; a token passes when its `aud`
matches any entry, so one deployment can trust tokens minted for different
audiences (for example a browser front door and direct API clients).

## MCP OAuth Discovery

MCP 2.0 can expose the HTTP listener as an OAuth 2.1 resource server. DlightRAG
still does not issue tokens: the configured issuer signs users in and issues an
access token, while DlightRAG verifies it through the existing JWT/JWKS settings.

Enable standards-based MCP client discovery by adding the externally reachable
MCP endpoint:

```yaml
access:
  auth_mode: jwt
  jwt_algorithm: RS256
  jwt_jwks_url: https://auth.example.com/.well-known/jwks.json
  jwt_issuer: https://auth.example.com
  jwt_audience: api://dlightrag-rest
interfaces:
  mcp:
    transport: streamable-http
    resource_server_url: https://rag.example.com/mcp
```

The MCP server then publishes RFC 9728 Protected Resource Metadata and returns a
standard `WWW-Authenticate` challenge that points clients to it. The public URL
cannot be inferred from `interfaces.mcp.host`: a bind such as `0.0.0.0:8101` does not reveal
the reverse-proxy URL clients use.

For native MCP OAuth, this public URL is also the exact expected JWT audience.
REST may keep its own `access.jwt_audience`; the MCP verifier narrows validation to the
resource URL instead of accepting a broader REST audience list.

Omit `interfaces.mcp.resource_server_url` when clients already hold a directly supplied
static-key JWT. JWT verification and claim-based access control continue to work,
but clients do not receive OAuth discovery metadata. TLS keys and JWT signing
keys remain separate.

## Azure Entra ID (MSAL + JWKS)

A concrete instance of the JWKS setup above. DlightRAG is the resource server:
a browser client acquires a token from Entra with MSAL, then calls DlightRAG,
which validates it against Entra's published keys. DlightRAG holds no secret and
never contacts MSAL itself.

Register one **App Registration** for the API and copy three values into config:

| Entra value | Where to find it | Used for |
|---|---|---|
| Directory (tenant) ID | App registration → Overview | building `access.jwt_jwks_url` and `access.jwt_issuer` |
| Application (client) ID | App registration → Overview | `access.jwt_audience` (v2 access tokens) |
| Application ID URI | Expose an API | the resource clients request a scope on |

Set the API app's `accessTokenAcceptedVersion` to `2` (Manifest) so tokens carry
the v2 issuer and audience below, and expose at least one delegated scope such as
`access_as_user`. The client must request that scope: a token's audience is
derived from the requested scope, so without it Entra never mints a token
audienced for DlightRAG.

```yaml
access:
  auth_mode: jwt
  jwt_algorithm: RS256
  jwt_jwks_url: https://login.microsoftonline.com/<TENANT_ID>/discovery/v2.0/keys
  jwt_issuer:  https://login.microsoftonline.com/<TENANT_ID>/v2.0
  jwt_audience: <API_CLIENT_ID>
```

JWKS serves public keys, so these values are not secret and can live in
`config.yaml`. DlightRAG uses the token's `sub` claim as `user_id`.

For per-workspace authorization, define **App Roles** on the API registration and
assign AD groups to them in the matching Enterprise Application. Assigned roles
land in the token's `roles` claim (including delegated user tokens), which
`access.control: jwt_claims` matches:

```yaml
access:
  control:
    mode: jwt_claims
    rules:
      - claim: roles
        value: finance.readers
        workspaces: [finance]
        actions: [reader]
```

| Gotcha | Detail |
|---|---|
| v1 vs v2 | With `accessTokenAcceptedVersion: 2`, `iss` ends in `/v2.0` and `aud` is the client-id GUID. Left at v1, `iss` is `https://sts.windows.net/<tenant>/` and `aud` is `api://<client-id>`. Decode a real token at jwt.ms and match `iss`/`aud` exactly. |
| Roles, not groups | Entra `groups` holds group object IDs (not names) and is replaced by an overage reference past ~200 groups. App Roles emit stable string values in `roles`. |
| Algorithm | Entra signs with `RS256`; do not leave the `HS256` default. |
| CORS | The bundled `/web` UI is same-origin, so `access.cors_allow_origins` does not affect it. Pin explicit origins for a separately hosted SPA; browsers reject `["*"]` once that SPA sends credentials. |

## Ingress Topology (Per-Surface Front Doors)

REST and Web share one process on `interfaces.api.port` (8100); MCP is a separate
streamable-http listener on `interfaces.mcp.port` (8101). All surfaces share one
`access.auth_mode`, yet each can sit behind a different front door -- as long as every
request carries a bearer JWT that the single `access.auth_mode: jwt` can verify.

| Surface | Port · path | Caller | Front door |
|---|---|---|---|
| Web UI | 8100 · `/web` | Browsers | oauth2-proxy injects `Authorization: Bearer` |
| REST API | 8100 · `/retrieve`, `/answer`, … | Programmatic | none -- client sends its own bearer |
| MCP | 8101 · `/mcp` | Agents | none -- client sends its own bearer |

Only browsers need an interactive redirect, so oauth2-proxy fronts `/web` alone;
API and MCP clients already hold a token, so DlightRAG verifies them directly --
no proxy, and no Easy Auth (unavailable on AKS regardless).

Apply the auth annotation to the `/web` route only:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dlightrag-web            # browser UI -- gated by oauth2-proxy
  annotations:
    nginx.ingress.kubernetes.io/auth-url: https://oauth2-proxy.example.com/oauth2/auth
    nginx.ingress.kubernetes.io/auth-signin: https://oauth2-proxy.example.com/oauth2/start?rd=$escaped_request_uri
spec:
  rules:
    - host: rag.example.com
      http:
        paths:
          - path: /web
            pathType: Prefix
            backend:
              service: { name: dlightrag, port: { number: 8100 } }
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dlightrag-api            # REST + MCP -- DlightRAG verifies the bearer itself
spec:
  rules:
    - host: rag.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service: { name: dlightrag, port: { number: 8100 } }
          - path: /mcp
            pathType: Prefix
            backend:
              service: { name: dlightrag-mcp, port: { number: 8101 } }
```

If oauth2-proxy forwards a token whose `aud` differs from direct REST clients,
list both REST/Web audiences in `access.jwt_audience`:

```yaml
access:
  jwt_audience:
    - api://dlightrag             # direct REST clients
    - <oauth2-proxy-client-id>    # browser token forwarded by the proxy
```

Native MCP OAuth remains separate: its token audience must exactly equal
`interfaces.mcp.resource_server_url`, regardless of the broader REST/Web audience list.

## Ingress Responsibilities

DlightRAG owns the semantic invariants no generic edge device can evaluate, and
deliberately owns nothing else at the network layer.

**The application enforces:**

- authentication, authorization, and owner scope on every run, event, artifact,
  workspace, and conversation read or write;
- idempotency keys and their 409 conflict on changed input;
- HTTPS-only fetches with redirect, DNS, and SSRF validation, plus byte,
  decompression, part-count, and pixel limits on uploads and fetched bytes;
- a receive-layer streaming body cap so an oversized or chunked body never
  reaches a parser, in addition to each route's semantic attachment limits;
- durable lease fencing, so a worker that lost its lease cannot mutate state the
  new owner holds;
- sanitized model-visible and client-visible output, with exception class and
  traceback kept for operators only; and
- provider concurrency and per-call timeouts on every external model, embedding,
  rerank, fetch, and parser request.

**The ingress owns** generic request-rate and quota limits, DDoS and volumetric
protection, WAF signature rules, IP/geo/bot reputation policy, TLS termination
and certificate lifecycle, and connection/concurrency caps. Use Azure Front Door,
Application Gateway WAF, API Management, NGINX, or the equivalent for your
platform. **DlightRAG ships no in-process rate limiter and no WAF**, and adding
one would duplicate ingress policy with worse context.

Microsoft Sentinel is SIEM/SOAR: it monitors, correlates, and drives response
playbooks over ingress and application logs. It is not the inline WAF and must
not be relied on to block a request in flight.

Because accepted answer runs queue rather than fail on capacity, a deployment
must bound and monitor PostgreSQL storage and apply ingress rate limits
appropriate to its `access.auth_mode`. `access.auth_mode: none` and `access.auth_mode: simple`
collapse every caller into one deployment owner, so they are only safe behind an
ingress that already restricts who can reach the port.

Both probe endpoints are unauthenticated by design: `GET /health` is liveness
only and never queries PostgreSQL, and `GET /ready` short-caches its database and
corpus verdict, so neither turns an unauthenticated poll loop into database load.

## Access Control

Authentication answers "who is calling?" Access control answers "what can this
authenticated caller do?"

Access control is disabled by default:

```yaml
access:
  control:
    mode: allow_all
```

Enable claim-based workspace permissions when the JWT issuer already supplies
verified group or role claims:

```yaml
access:
  auth_mode: jwt
  control:
    mode: jwt_claims
    rules:
      - claim: groups
        value: finance-rag-readers
        workspaces: [finance]
        actions: [reader]
```

`jwt_claims` requires `access.auth_mode: jwt` and at least one rule. Claim matching
supports string claims and list-like claims. Workspace patterns support `*`.
Action patterns support exact actions, `*`, prefixes such as `workspace.*`, and
the named presets `reader`, `editor`, and `admin` (see below).

Actions enforced by REST, Web, and MCP include:

| Action | Meaning |
|---|---|
| `workspace.query` | Retrieve and answer |
| `workspace.ingest` | Start ingestion |
| `workspace.list_files` | List files |
| `workspace.delete_files` | Delete files |
| `workspace.download_source` | Download source files |
| `workspace.read_metadata` | Read metadata |
| `workspace.update_metadata` | Update metadata |
| `workspace.read_visual_asset` | Read rendered visual assets |
| `workspace.create` | Create workspace |
| `workspace.delete` | Delete workspace |
| `workspace.reset` | Reset workspace |
| `job.read` | Read ingest job status |
| `job.cancel` | Stop a running ingest job |

### Source download boundary

Public source payloads expose stable `source_uri` provenance and, on HTTP
surfaces, an adapter-projected `download_url` containing only a document ID and
workspace. They never expose the stored `download_locator` or a server-local
path. REST `/files/raw` and Web `/web/api/files/raw` independently recheck
`workspace.download_source` against the source's actual workspace before they
stream retained bytes or redirect to Azure, S3, or queryless public HTTPS.

Signed/query-bearing HTTPS URLs are fetch credentials, not durable locators. A
caller must retain the bytes or provide a separate queryless durable
`download_uri`. DlightRAG never promotes the signed query into `source_uri`,
`download_locator`, public source payloads, or source-contract logs.

Durable ingest jobs do retain the caller's complete fetch input in their request
record so a worker can recover the job. Treat the ingest-job database as secret
storage, restrict access, and keep job pruning enabled; this recovery record is
not a public source/download field.

**Action presets.** `actions` is a list, and each entry may be an exact action,
a `workspace.*` prefix, `*`, or one of the built-in presets below. A caller is
allowed when any entry matches, so presets and exact actions can be combined
(for example `actions: [reader, workspace.update_metadata]`):

| Preset | Expands to |
|---|---|
| `reader` | `workspace.query`, `workspace.list_files`, `workspace.download_source`, `workspace.read_metadata`, `workspace.read_visual_asset` |
| `editor` | `reader` plus `workspace.ingest`, `workspace.update_metadata`, `workspace.delete_files`, `job.read`, `job.cancel` |
| `admin` | `*` (every action, including `workspace.create`, `workspace.delete`, `workspace.reset`) |

```yaml
access:
  control:
    mode: jwt_claims
    rules:
      - claim: roles
        value: finance.editors
        workspaces: [finance]
        actions: [editor]
```

## Answer Attachment Resources

Answer attachments are read as request-local resources and are bounded on every
channel before the orchestrator runs. `answer.generation.max_attachments` (default 6) caps
the count, `answer.generation.max_attachment_bytes` (100 MiB) caps each item, and
`answer.generation.max_total_attachment_bytes` (128 MiB) caps the request; REST multipart
uploads are refused with a stable 4xx before the body is buffered.

HTTPS link attachments are inert handles until they are read. Only `https` URLs
are admitted, embedded credentials are rejected, and full scheme/host/SSRF
validation is repeated on every fetch — a link cannot resolve to a private,
loopback, or link-local address. Fetched bytes pass the same per/total size and
decoded-pixel limits as uploads.

Binary conversion is defensive. MarkItDown runs with plugins disabled and no
network access, and OOXML archives (DOCX/PPTX/XLSX) pass a central-directory
zip-bomb preflight — entry-count, per-entry size, total size, and expansion-ratio
limits — before any converter opens them, so an archive that is admissible by
byte size can still be rejected if its internal expansion looks like a bomb.
Images pass MIME and decoded-pixel checks before inspection.

Full attachment bytes never enter model context: only bounded text windows,
capped tool observations, and budgeted image blocks do. Research Context
Contributions do not make Profile Memory, Skills, extension context, or child
summaries citable; only rows admitted to the Evidence ledger can become
sources. Child Evidence carries child-session and parent-call provenance before
the parent spawn effect settles.

The execution modes are exactly `disabled`, `trust`, and `sandbox`. Trust runs
as the service user and Bash can access the host/container network and every
path that user can reach; rooted file-tool checks are not a shell sandbox.
Sandbox selection without a trusted backend fails explicitly. Trusted Python
extensions are deployment code and have only tool, context, and execution
adapter seams—there is no model-facing approval or permission system.

Outbound MCP endpoints and tool names are deployment allowlists. Remote tools
run in foreground sessions and receive only their declared arguments; DlightRAG
does not provide endpoint discovery, OAuth token brokerage, or a management
plane. Protect endpoint credentials and network egress at deployment level.

The optional Exa
web-search capability is gated solely by the presence of
`DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY`; keep it in `.env`, and its absence removes the
capability entirely.

Admitted bytes become owner-scoped content-addressed artifacts owned by the
durable run, so deduplication never crosses an owner. Run-scoped fetched web
bytes are stored only after the HTTPS, redirect, DNS, SSRF, and byte validation
above passes, and the blob plus its run reference commit in one transaction
before the tool result may settle on the journal — a resumed run therefore reads the
bytes it originally fetched rather than whatever the page serves now. Workspace
authorization is evaluated once before the run-creation transaction and only the
resulting workspace set is stored, never a token or mutable claims; a later policy
change does not revoke an already accepted run, and its owner may cancel it.
Follow-up and fork are new ordinary runs, not mutations of that accepted run, so
every REST, MCP, SDK, and Web continuation rechecks current `workspace.query`
authorization before its own acceptance transaction.

### Answer Artifact browser boundary

Artifact descriptors expose validated media and owner-scoped URLs, never Blob
bytes or Agent Workspace paths. Authenticated Artifact data and Markdown
presentation responses use `Cache-Control: private, no-store`; active and unknown
formats are attachments with `nosniff`. PDF preview is sandboxed and carries no
same-origin capability.

HTML Artifact bytes are never served as executable same-origin documents. The
browser fetches them through the authenticated inert data route and, only after
explicit user consent, creates one `srcdoc` iframe with `sandbox="allow-scripts"`
but no `allow-same-origin`, forms, popups, downloads, frames, workers, storage,
device permissions, or application bridge. A CSP inserted before Artifact bytes
blocks normal fetches and subresources. The only parent signal is a private,
one-way Escape close token installed by the wrapper and removed before Artifact
code runs; malformed messages and all Tool/MCP-shaped messages are ignored.
Closing or switching Artifact Canvas aborts loading and destroys the iframe.

This boundary isolates DlightRAG's authenticated DOM, cookies, and storage. It
does not provide CPU or memory quotas, execute server code, or justify a claim of
absolute browser network denial: browser engines do not expose a complete
all-egress primitive for arbitrary JavaScript. Chromium is the security-
regression baseline for active preview.

## Deployment Posture

| Deployment | Recommended posture |
|---|---|
| Local development | Bind REST/MCP to loopback and use `access.auth_mode: none` |
| Internal shared service | Use `simple` only behind trusted network boundaries |
| Enterprise multi-user | Use `jwt` with JWKS from the external IdP and enable `jwt_claims` when workspace permissions are required |

MCP streamable HTTP binds to loopback (`127.0.0.1`) by default, reachable only
from the local machine. To make it reachable from other hosts, set a
non-loopback `interfaces.mcp.host` and enable auth -- an unauthenticated non-loopback MCP
would let any client call ingest/delete. Host/Origin DNS-rebinding protection is
always enabled, including when bearer auth is active. Public deployments must
add the externally visible host and browser origins to `interfaces.mcp.allowed_hosts` and
`interfaces.mcp.allowed_origins`; browser clients must also be allowed by
`access.cors_allow_origins`. Authentication does not replace Host validation.
