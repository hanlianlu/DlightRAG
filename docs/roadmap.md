# Roadmap

Later product slices, not the current milestone. Living spec milestones stay
authoritative for M7+ scope; this page records decided follow-ups that are not
a milestone yet.

## Browser identity UX (after M7)

Web today stores a pasted bearer token in an HttpOnly cookie. REST and MCP
already use the same JWT. That identity model stays: DlightRAG is a resource
server. It does not become an IdP, store passwords, or keep a user directory.

The upgrade is the browser default, not a second principal:

- For `auth_mode: jwt` with an issuer/JWKS, the Web login default is
  Authorization Code + PKCE against that issuer. Tokens remain in the existing
  HttpOnly cookie. Refresh is the issuer's job.
- Paste-token login stays as an operator and development hatch, not the human
  default.
- `none` and `simple` do not grow a login product. Operators who want a hosted
  login page in front of every surface keep using a gateway or IAP.

Owner projection stays `jwt` + issuer + `sub`. Memory and Answer Runs keep
using that owner. This slice does not invent a cookie-only identity.

## Deferred from the M8 deviation audit

- Durable compaction runtime: Context Projection seeds `summary=None`;
  `select_compaction_boundary` / `should_compact` are vocabulary only.
- Measured token anchors: production seeds `TokenAnchor` zeros; live capacity
  accounting is estimate-only.
