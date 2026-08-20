# Roadmap

Later product slices, not the current milestone. Living spec milestones stay
authoritative for M7+ scope; this page records decided follow-ups that are not
a milestone yet.

## Browser identity UX (after M7)

Web today stores a pasted bearer token in an HttpOnly cookie. REST and MCP
already use the same JWT. That identity model stays: DlightRAG is a resource
server. It does not become an IdP, store passwords, or keep a user directory.

The Web frontend is never bare-exposed: every deployment fronts it with an
edge that already authenticates humans (Cloudflare Access / Zero Trust, or
Azure Easy Auth / Entra front door). So the browser upgrade is **edge-asserted
identity**, not a new in-app login flow:

- For `auth_mode: jwt` behind Cloudflare Access, the Web surface accepts the
  edge-verified assertion (`Cf-Access-Jwt-Assertion` header or the
  `CF_Authorization` cookie JWT) and verifies issuer + audience against the
  team JWKS. Behind Azure, it verifies the passed-through AAD ID token
  (`X-MS-TOKEN-AAD-ID-TOKEN`) against tenant discovery keys; the unsigned
  `X-MS-CLIENT-PRINCIPAL` header is display-only enrichment, never
  authorization. The origin renders no active login page and issues no Web
  token.
- Owner projection stays `jwt` + issuer + `sub` (the edge token's issuer for
  the Web surface; the external issuer for REST/MCP bearer tokens).
- Paste-token login shrinks to a local-development hatch, not a production
  surface.
- **PKCE is dropped** (decided 2026-08-20): with an always-edge-fronted
  browser surface the edge owns login, so an in-app authorization-code flow
  has no deployment to run in.
- `none` and `simple` do not grow a login product. Operators who want a
  hosted login page in front of every surface keep using a gateway or IAP.

Session hardening ships with the edge-identity slice (reference: DeerFlow 2.0
auth audit, `docs/plans/2026-08-20-pkce-deerflow-research.md`):

- CSRF double-submit cookie plus an Origin check on state-changing routes;
  `secrets.compare_digest` comparisons;
- auth middleware stays fail-closed behind an explicit public-path allowlist;
- edge-injected identity is always verified (issuer/audience/JWKS) and never
  trusted as a plaintext header;
- no access token is ever issued in response JSON.

For the bare-exposed or no-edge case, the answer stays a gateway or IAP in
front of every surface — DlightRAG does not grow its own login product there
either.

Owner projection stays `jwt` + issuer + `sub`. Memory and Answer Runs keep
using that owner. This slice does not invent a cookie-only identity.

## Deferred from the M8 deviation audit

Both M8-D13 items shipped after Milestone 8: the durable compaction runtime
commits `CompactionEntry` projections at the proactive `H` trigger, and
provider-measured token anchors feed live capacity accounting. Nothing from
the audit remains deferred.
