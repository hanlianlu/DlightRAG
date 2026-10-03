# Agent Accounts and the Agent Mailbox

The Agent may register accounts on third-party sites under an identity of its own,
never the owner's. DlightRAG generates each password, fills it into the page by
reference, and seals it under the deployment key ring, so no password enters model
context and no surface ever shows one. A parent Agent Session's accounts persist per
owner for that owner's later Runs; a Child's last only for the Run. An optional
Agent Mailbox delivers verification mail to per-owner aliases from an S3-compatible
bucket that the deployment fills. A CAPTCHA still stops the Agent, and the Settings
section for accounts is designed with the product owner later and not built now.

## Status

Accepted; implementation in progress. It lands as slices 4 and 5 of the sequence
[ADR 0032](0032-the-agent-browser.md) records, and each passes a review on four axes
— Standards, Spec, correctness and security, and performance — before the next one
begins.

It amends one earlier statement.
[ADR 0005](0005-public-web-resource-acquisition.md)'s closing line keeps
authenticated browsing outside it; this decision takes authenticated browsing in,
with the Agent's own accounts only, and general crawling stays outside. It reuses
the key ring and `CredentialCipher` of
[ADR 0012](0012-personal-connections-and-hot-plug.md) and the Connections rule that
a credential is write-only, and it draws
[ADR 0025](0025-a-child-inherits-capability-not-authority.md)'s authority line
through account creation. [Security](../security.md)'s statement that DlightRAG does
not "manage users/passwords" stays true of DlightRAG's own users: an Agent Account
is a credential the Agent holds on a third-party site.

## Context

ADR 0032 gives Research an anonymous browser whose sessions end with the Run. Much of
the free Web sits behind a free account, and most sign-ups confirm an email address.
An Agent left to do this alone would choose a password in model context, where it
would land in Session Entries, compaction summaries, provider requests, and traces,
within reach of any page's prompt injection. It would also need a mailbox it can
read. The owner's address and accounts are not an option: the Agent never uses the
owner's credentials or personal information.

Three pieces already exist. The deployment key ring (`connection-keyring.json` in
`deployment.working_dir`) and `CredentialCipher`
(`application/connections/credentials.py`) seal Connection credentials as
AES-256-GCM envelopes whose associated data binds owner, Connection, and Grant
([key ring](../personal-mcp-connections.md#secret-handling-and-key-ring)). Settings
treats a credential as write-only: the bearer field is never shown again, and no
management result carries a token. And `aiobotocore`, already a dependency of the
Corpus S3 source, reads an S3-compatible bucket.

## Decision

**The Agent may register, and registration is on by default.**
`answer.agent.browser.account_registration` defaults to `true`. Set to `false`,
`browser` does not offer `register`.

**The identity is the Agent's own.** No credential, name, address, or other personal
information of the owner enters a form, and an alias carries nothing of the owner's
identity.

**DlightRAG makes the password, fills it by reference, and never shows it.**
`register` takes the snapshot refs of the form's fields. DlightRAG generates a strong
password and fills it into the referenced password fields, and with an Agent Mailbox
configured it also fills the owner's alias for that site into the email field. It
records the account and returns the email address used, never the password; the
Agent submits the form with `click` or `press`. The same action serves a site's
password-reset form: for a site where the account already exists, it generates a new
password, fills it, and replaces the stored one. An account whose sign-up the site
refused, or whose envelope no key opens any more, is therefore recovered the way a
person recovers one, through the site's reset mail to the same alias. `login` fills the stored
credentials for the page's registrable domain, the Public Suffix List's eTLD+1, and
only into fields whose frame belongs to that domain. A password is never in model
context, and outside its sealed envelope it appears in no tool argument or result,
Session Entry, event, trace, or log.

**A parent's accounts persist per owner, sealed under the key ring.** An account a
parent Agent Session registers is stored per owner. `CredentialCipher` seals its
password under the deployment key ring, with associated data binding owner, site,
and account, and later Runs of the same owner log in with it. The associated data
carries its own domain label beside the Connection's, so an account envelope never
opens as a Grant, nor a Grant as an account. Without a key ring, registering and
logging in fail closed, as Connection credentials do, and an account whose envelope
no key opens is unusable.

**Creating a persistent account is the parent's.** It writes durable owner state, so
only the parent may do it: ADR 0025's line, as with `remember`. A Child keeps the
`browser` tool and its `register` action, but a Child's registration is Run-scoped,
usable for the rest of the Run and discarded when the Run settles. A Child logs in
with the owner's persistent accounts, as it recalls the owner's memory: using
durable owner state is capability, and adding to it is authority.

**Credentials are write-only everywhere.** No route, Settings view, export, event, or
tool result returns a password, following the Connections precedent. The deferred
Settings section lists and removes accounts; it does not show passwords.

**A CAPTCHA stops registration as it stops browsing.** ADR 0032's boundary has no
account exception: a sign-up behind human verification is abandoned and reported,
never solved, bypassed, or outsourced.

**The Agent Mailbox is optional, and its contract is a bucket.** DlightRAG reads raw
RFC 822 messages from an S3-compatible bucket with the existing `aiobotocore`
dependency, and the repository holds no vendor code. How mail reaches the bucket is
the deployment's business. The development deployment routes a catch-all on its own
domain through Cloudflare Email Routing to an Email Worker that writes to R2; another
deployment could use SES receipt rules that write to S3. The Worker is a deployment
artifact that documentation may show as an example; it is not product code. The
bucket layout is the contract's one obligation on the deployment: each message is
written whole, as one object, under `<prefix>/<envelope recipient>/`, because a
`To:` header does not reliably name the alias a message was delivered to, and the
envelope recipient does. DlightRAG lists one alias's prefix and never scans the
bucket. `answer.agent.mailbox` holds the S3 endpoint, bucket, prefix, and alias
domain, and the bucket credentials live in `.env`.

**An alias belongs to one owner and one site.** Aliases are minted on the configured
alias domain, deterministically per (owner, site): the same owner registering on the
same site always gets the same address, so mail for that account keeps reaching it
in later Runs, and two owners never share one. A Child's Run-scoped registration
gets a fresh random alias instead, so an account discarded at settlement never
occupies the address the owner's persistent account on that site will need.

**`inbox` returns the session's recent mail as untrusted context.**
`browser(action="inbox")` lists only messages addressed to this owner's aliases,
or to the Child's own Run-scoped aliases, and received since the Agent Session's
most recent `register` or `login`. The window covers a verification link after
signing up and an emailed sign-in code after logging in, which a returning account
meets often because every Run's session starts empty. Each entry gives the sender,
subject, time, and the links and codes in the body. Mail is untrusted model context,
never Evidence, and a link it returns is followed with `navigate`, under ADR 0032's
first-URL check.

**Without a mailbox, the Agent uses temporary mail itself.** It may open a
temporary-mail website in the browser and use that address. No product code
supports it, and `register` records the address the Agent typed.

**The Settings section is co-designed later.** Agent Accounts get a Settings section
designed with the product owner, not a minimal list, in the phase that also
redesigns the Profile Memory and Conversation Sessions sections for compactness and
visual style. Nothing of it is built in slice 4.

## Considered options

- **The owner's own email address.** Rejected. It is the owner's personal
  information, it would fill the owner's inbox with third-party mail, and reading
  that inbox would take the owner's mail credentials.
- **An Outlook.com mailbox.** Rejected. Outlook.com ended basic authentication on
  2024-09-16, so reading it takes OAuth2 and a token lifecycle to run; plus
  addressing is unreliable; and an account holds at most 10 aliases, far fewer than
  one per (owner, site).
- **Third-party temporary mail as the primary path.** Rejected. Such an inbox is
  readable by anyone who knows its address and expires, so a persistent account
  could never be verified or recovered again; sign-up forms often refuse those
  domains; and the path depends on a third party. It stays the fallback without a
  mailbox, at no product cost.
- **No mailbox at all.** Rejected. Most free sign-ups verify by email, and persistent
  accounts would rest on disposable addresses. The mailbox stays optional, so a
  deployment without one still works.
- **Show passwords in a UI.** Rejected, following the Connections precedent that
  credentials are write-only: a shown password is a secret copied into a browser
  surface, and nothing in the product needs a person to read one.
- **Let the model choose or see the password.** Rejected. It would sit in the durable
  transcript, compaction summaries, provider requests, and traces, within reach of a
  page's prompt injection.
- **Persist the browser session instead of credentials.** Rejected. Sessions are
  anonymous and end with the Run (ADR 0032); a stored cookie jar is a bearer
  credential for every site in it, and logging in afresh keeps each Run's state
  explicit.
- **Let a Child create persistent accounts.** Rejected: a Child would leave durable
  owner state its parent never reviewed (ADR 0025).
- **Code against a mail vendor's API.** Rejected. The bucket is the contract, so any
  S3-compatible store works and the routing stays the deployment's.

## Consequences

Landing order, the sequence shared with ADR 0032 and ADR 0033:

1. The substrate and the Rendered Read (ADR 0032).
2. The interactive tool, capture, downloads, and Children (ADR 0032).
3. `materialize` ([ADR 0033](0033-resource-materialization.md)).
4. **Accounts and the mailbox backend** (this decision): `register`, `login`, and
   `inbox`; the owner-scoped account store, whose suite owns a scratch database
   (`tests/support/pg`); the cipher's second consumer; a pinned Public Suffix List,
   which the repository does not carry today; the bucket reader;
   `account_registration` and `answer.agent.mailbox`; and ADR 0032's acceptance 5 —
   register on a free site with email verification, verify, log in, read a gated
   page, and reuse the account in a later Run.
5. **The frontend co-design** (this decision): the Agent Accounts Settings section,
   with the Profile Memory and Conversation Sessions redesign.

`CredentialCipher` lives in the Connections package today, and its associated data
and its errors name Connections. A second consumer moves the cipher and the loading
of the key ring to a module both consumers own, while the ring's file stays where it
is; writer maintenance and the rotation runbook re-encrypt and count account
envelopes together with Grants.

Live documents to revise as slices 4 and 5 land:
[domain language](../domain-language.md) (Agent Account and Agent Mailbox),
[security](../security.md) (the opening statement on passwords, the Answer Resources
and Execution boundary, and the key ring paragraph),
[Personal MCP Connections](../personal-mcp-connections.md) (the key ring's second
consumer), [configuration](../configuration.md) (`account_registration` and
`answer.agent.mailbox`), `.env.example` (the bucket credentials), and
[operations](../operations.md) (key ring rotation, and the mail routing example).

Residual risks, recorded rather than solved:

- An account is recorded when its fields are filled, before the site accepts the
  sign-up, so a refused sign-up leaves a record that `login` will fail with. The
  remedy is the reset path above, not a second registry of site outcomes, which
  DlightRAG cannot observe reliably.
- A Child's Run-scoped account stays on the site after the Run discards it, under an
  alias nothing will read again. Sites keep such orphans whatever the address; a
  random alias keeps them from colliding with the owner's.
- Mail retention in the bucket is the deployment's; DlightRAG only reads it.
- A catch-all domain accepts mail for any alias, so anyone who learns one can write to
  the Agent. That is why mail is context and never Evidence.
- A site and its page scripts necessarily see the password; one generated password
  per account confines a leak to that account.
- A password crosses the deployment's internal network to the pool unencrypted,
  inside the Playwright protocol.
- Envelopes in database backups stay readable to any retained copy of the key that
  sealed them, as for Connections.
- A pinned Public Suffix List ages, and a domain it misclassifies shares or splits
  accounts wrongly.
- Whether a site's terms allow automated sign-up is the site's to say, and the
  product does not read those terms, as `read` does not read `robots.txt`
  (ADR 0005).
