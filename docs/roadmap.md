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

Session hardening ships with the PKCE slice, not as a separate product
(reference: DeerFlow 2.0 auth audit, `docs/plans/2026-08-20-pkce-deerflow-research.md`;
DeerFlow has no OAuth/PKCE of its own, so nothing there is copied for the flow):

- a `token_version` claim lets a password change or operator reset revoke
  existing cookies immediately instead of waiting out `exp`;
- CSRF double-submit cookie plus an Origin check on state-changing and login
  routes; `secrets.compare_digest` comparisons;
- the access token lives only in the HttpOnly cookie and never appears in
  response JSON;
- auth middleware stays fail-closed behind an explicit public-path allowlist;
- rate limiting for login uses shared storage, not in-process counters;
- no OAuth config fields or callback shapes are declared before the flow
  exists (dead scaffolding).

Owner projection stays `jwt` + issuer + `sub`. Memory and Answer Runs keep
using that owner. This slice does not invent a cookie-only identity.

## Deferred from the M8 deviation audit

Both M8-D13 items shipped after Milestone 8: the durable compaction runtime
commits `CompactionEntry` projections at the proactive `H` trigger, and
provider-measured token anchors feed live capacity accounting. Nothing from
the audit remains deferred.
