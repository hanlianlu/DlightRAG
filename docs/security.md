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

## Authentication Modes

| Mode | Use case |
|---|---|
| `none` | Local loopback development only |
| `simple` | One shared bearer token for trusted internal deployments |
| `jwt` | User-scoped deployments with externally issued signed tokens |

`auth_mode: none` returns an anonymous user context. If the API binds to a
non-loopback host while auth is off, config validation emits a warning.

When auth is enabled, replace wildcard CORS origins with explicit origins.
Browsers reject credentialed cross-origin requests with `["*"]`.

## Simple Bearer Token

`simple` mode compares the bearer token with `api_auth_token`.

```bash
openssl rand -base64 32
DLIGHTRAG_AUTH_MODE=simple
DLIGHTRAG_API_AUTH_TOKEN=<generated>
```

Clients send:

```http
Authorization: Bearer <generated>
```

REST can receive `X-User-Id` in simple mode and uses it as the user id for
request scoping. MCP simple-mode requests use the default anonymous user id.
`simple` mode is admission control, not per-user authorization.

## Static JWT

`jwt` mode verifies externally issued JWTs. Tokens must include `sub`; DlightRAG
uses it as the authenticated `user_id`.

```bash
openssl rand -base64 64
DLIGHTRAG_AUTH_MODE=jwt
DLIGHTRAG_JWT_VERIFICATION_KEY=<generated>
DLIGHTRAG_JWT_ALGORITHM=HS256
```

For `HS*` algorithms, `jwt_verification_key` is the shared HMAC key. For
`RS*`/`ES*` algorithms, it is the public key PEM from the issuer. DlightRAG
does not sign, renew, or mint these tokens.

If `jwt_audience` is unset, audience verification is disabled. If `jwt_issuer`
or `jwt_audience` is set, PyJWT validates those claims during token decoding.

## JWKS / OIDC Issuers

For Azure Entra, Okta, Auth0, Keycloak, and other OIDC-style issuers, prefer
JWKS so signing-key rotation is handled by PyJWT's `PyJWKClient`.

```yaml
auth_mode: jwt
jwt_algorithm: RS256
jwt_jwks_url: https://login.example.com/.well-known/jwks.json
jwt_issuer: https://login.example.com/tenant/v2.0
jwt_audience: api://dlightrag
```

`jwt_issuer` and `jwt_audience` are required when `jwt_jwks_url` is set.

## Access Control

Authentication answers "who is calling?" Access control answers "what can this
authenticated caller do?"

Access control is disabled by default:

```yaml
access_control:
  mode: allow_all
```

Enable claim-based workspace permissions when the JWT issuer already supplies
verified group or role claims:

```yaml
auth_mode: jwt
access_control:
  mode: jwt_claims
  rules:
    - claim: groups
      value: finance-rag-readers
      workspaces: [finance]
      actions:
        - workspace.query
        - workspace.list_files
```

`jwt_claims` requires `auth_mode: jwt` and at least one rule. Claim matching
supports string claims and list-like claims. Workspace patterns support `*`.
Action patterns support exact actions, `*`, and prefixes such as
`workspace.*`.

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

## Deployment Posture

| Deployment | Recommended posture |
|---|---|
| Local development | Bind REST/MCP to loopback and use `auth_mode: none` |
| Internal shared service | Use `simple` only behind trusted network boundaries |
| Enterprise multi-user | Use `jwt` with JWKS from the external IdP and enable `jwt_claims` when workspace permissions are required |

MCP streamable HTTP binds to loopback by default and includes Host/Origin
allowlists for DNS-rebinding protection. Set explicit allowed hosts/origins and
enable auth before exposing MCP beyond loopback.

