# Roadmap

No decided follow-up is currently pending. Living specification milestones remain
authoritative for future scoped work; this page records completed follow-ups so
they are not mistaken for open commitments.

## Completed Follow-ups

- [x] **Browser identity UX.** Edge-asserted Web identity shipped for Cloudflare
  Access, Azure Easy Auth, and AWS/CloudFront. Issuer, audience, signature, and
  JWKS validation remain fail-closed; unsigned enrichment headers never grant
  access. CSRF double-submit and exact-Origin checks protect state-changing Web
  routes. Paste-token login is a local-development hatch, and PKCE/in-app user
  management remain deliberately out of scope. See
  [security.md](security.md#edge-asserted-web-identity).
- [x] **M8 deviation-audit follow-ups.** Durable proactive compaction commits
  `CompactionEntry` projections, and provider-measured token anchors feed live
  capacity accounting. Nothing from that audit remains deferred.
