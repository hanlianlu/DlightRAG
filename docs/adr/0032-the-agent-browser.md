# The Agent Browser

Research gains a browser. The Agent Browser is a deployment capability that the
Agent reaches through tools, never through its own processes or a Connection: each
Run leases one Chromium in a pool container over the Playwright protocol, and each
Agent Session works in its own context there. `read` renders a page as the last
automatic step of the Extract chain, or when the model asks with `rendered=true`,
and this Rendered Read appends its representation to the same Web Resource. One
`browser` tool drives multi-step interaction and admits what it captures or
downloads as citable Resources. Sessions are anonymous and end with the Run, a
CAPTCHA stops the Agent, the deployment's network confines what the browser
reaches, and Fast is unchanged.

## Status

Accepted; implementation in progress. It lands as slices 1 and 2 of the sequence
under [Consequences](#consequences), which
[ADR 0033](0033-resource-materialization.md) and
[ADR 0034](0034-agent-accounts-and-the-agent-mailbox.md) share. Each slice passes a
review on four axes — Standards, Spec, correctness and security, and performance —
before the next one begins.

It amends four earlier statements, each revised when the slice that changes it
lands:

- [ADR 0005](0005-public-web-resource-acquisition.md)'s closed acquisition set
  (`direct_http`, `exa_extract`, or `tavily_extract`) gains `browser_render`,
  `browser_capture`, and `browser_download`. The admission origins are unchanged.
- ADR 0005's closing line, "Browser automation, rendered-page interaction,
  authenticated browsing, and general crawling remain outside this decision", no
  longer holds for browser automation and rendered-page interaction, which this
  decision takes in. Authenticated browsing with the Agent's own accounts is
  [ADR 0034](0034-agent-accounts-and-the-agent-mailbox.md)'s; general crawling stays
  outside.
- [Security](../security.md#answer-resources-and-execution)'s "cookies,
  authorization, arbitrary headers, and browser sessions are unavailable" stays true
  of `read` and stops being true of the Agent, which has Run-scoped sessions in the
  Agent Browser.
- [ADR 0029](0029-read-only-calls-run-at-once-and-settle-in-source-order.md)'s
  sentence that Extract text for bytes that hold none is "their text view … never a
  second representation" stays true of the hosted providers. The chain's browser
  entry appends a rendered representation instead.

It builds on five decisions without amending them:
[ADR 0024](0024-the-agent-sees-only-its-workspace.md)'s provider clause, which it
fulfils; [ADR 0025](0025-a-child-inherits-capability-not-authority.md)'s capability
line, which gives every Child the tool; ADR 0029's sequential default, under which
the tool runs alone; [ADR 0006](0006-configuration-ownership-and-deployment-bindings.md)'s
configuration ownership; and [ADR 0020](0020-uniform-environment-fast-inert-workspace.md),
which keeps Fast without tools.

## Context

Research reads the public Web through `read(url)` (ADR 0005): an anonymous GET,
admitted as a Web Resource and Evidence, with the configured hosted Extract chain
(Exa, then Tavily) supplying text once when the fetch fails or yields no text.
Three kinds of page stay out of reach.

- **A JavaScript application serves a shell.** The fetch succeeds, often with one
  line asking for JavaScript: neither a failure nor an empty text, so ADR 0005's
  objective triggers do not fire. Where they do fire, a hosted extractor is
  optional, paid, and renders on its own terms.
- **A multi-step task needs a session.** A search form, a filter, pagination behind
  a button, or a file behind a download control cannot be reached with one GET.
  ADR 0005 refused to turn `read` into a general HTTP client, and `curl` in Bash
  runs no JavaScript and yields model context, not Evidence.
- **A gated page needs an account.** That part is ADR 0034's.

ADR 0024 anticipated the shape: a capability whose state cannot live in a file, "a
browser session, a desktop", is a provider that runs in its own environment and is
reached through a tool, so the Agent's process view does not grow.

The deployment is Docker Compose on one host. `dlightrag-api`, `dlightrag-mcp`, and
the optional `dlightrag-reader` all run Query workers
([RunRuntime](../run-runtime.md#workers-and-scaling)), so any of them may execute a
Research Run, and a reclaimed Run resumes wherever it is claimed. Two measurements
on the development host's Docker Desktop fix the network design: a plain container
reaches the host-published PostgreSQL (5432), API (8100), and MCP (8101) ports
through `host.docker.internal`, and a container on an `internal` network reaches
neither the host nor the internet.

## Decision

**The Agent Browser is a deployment capability reached through tools.** It runs in
its own containers, in the shape ADR 0024 names for a browser session, and the
Agent's processes gain no path, socket, or binary: the Landlock allow-list is
unchanged. `BrowserProvider` is the port that leases a Run's browser endpoint, an
ordinary adapter boundary with one implementation, as `ExecutionEnvironmentAdapter`
has one in `TrustExecutionAdapter`. It is not a Connection. A Connection is an
owner's authorization of an external account over MCP
([ADR 0012](0012-personal-connections-and-hot-plug.md)): enabled per owner,
discovered as an untrusted catalogue, answered in text parts only, and refused at a
private endpoint unless an operator exempts it. The Agent Browser belongs to the
deployment, its endpoints are private by design, and what it returns becomes
Resources, Evidence, and image attachments under DlightRAG's own settlement.

**Reading has four tiers, and the browser is the last automatic one.**

1. `read(url)` over direct HTTP, Host-attested, always first.
2. The configured hosted Extract chain (Exa, Tavily).
3. A Rendered Read as the last entry of that chain. It runs only on ADR 0005's
   objective triggers — the direct fetch failed, or it produced no text for textual
   content — never on subjective quality and never on a heuristic.
   `answer.web_sources.extract_providers` admits the name `"browser"` and orders it
   like any provider. A derived order places it after `exa` and `tavily` when the
   Agent Browser is configured, and naming it without a configured Agent Browser is
   a configuration error, as naming a provider without its key is.
4. Interaction through the `browser` tool, for multi-step tasks only.

`read` stays a read. Sessions, form submissions, and their side effects live in a
separate tool that is neither read-only nor replayable, which is the line ADR 0005
drew when it refused to make `read` a general HTTP client.

**A Rendered Read appends to the Web Resource and never replaces its snapshot.** The
model asks for one with `read(url=…, rendered=true)` or
`read(resource_id=…, rendered=true)`. `rendered` is valid only for a URL or a Web
Resource handle; with any other target it is a validation error, never silently
ignored, under ADR 0005's rule for invalid overrides. The read renders the page in a
short-lived context of the Run's browser, serializes the rendered DOM, and converts
that HTML through the same route as direct HTTP HTML, so both text views come from
one converter. The result is appended to the same Web Resource as a representation
with acquisition `browser_render`. The admitted snapshot is never replaced, so
`view` and earlier citations keep reading what they read. The first successful
rendered representation is reused for the rest of the Run, Child Sessions included,
and a failed render pins nothing. When the automatic chain ends in the browser, a
plain `read` returns the rendered representation's text, as it returns Extract text
today. A rendered representation settles with the read that produced it, and
recovery restores it without rendering again, so `read` stays read-only and
replayable.

**What the browser returns is Evidence, and its assertion is the provenance.**
Rendered and captured content is citable. The Host attests the binding between what
the browser returned, the Resource Handle, and the URL; it does not attest that the
page is what an anonymous GET returns. That is the trust class of hosted
extraction, and the acquisition on every row says which tier produced it.

**One browser per Run, one context per Agent Session.** Each pool container runs
`playwright run-server --port 3000 --host 0.0.0.0 --max-clients 1`. DlightRAG
connects with the Python `playwright` package through `chromium.connect("ws://…")`.
The server launches a browser for that connection and closes it when the connection
ends, so the browser lives exactly as long as the Run's connection. The parent Agent
Session and each Child Session get their own browser context in it; Children run in
the parent's process and share the Run's connection. A Rendered Read uses a
temporary context that closes when the read ends. A context has one active page: a
popup or a new tab becomes the active page, and the tool says so.

**The protocol is Playwright's, pinned in lockstep.** The Python package and the
containers' Playwright are one version, pinned in both places. `playwright` 1.63.0
ships `py3-none` wheels classified for Python 3.14; `uv.lock` already holds it
through `pytest-playwright`, and it becomes a runtime dependency. The pool image
builds from Microsoft's `mcr.microsoft.com/playwright:v1.63.0-noble` and installs
the npm package `playwright@1.63.0` at build time, because the containers have no
route to the internet when they run. The server already refuses, with HTTP 428, a
client whose major or minor version differs; the pin also fixes the patch.
Observations come from `page.aria_snapshot(mode="ai", depth=…)`, whose `[ref=eN]`
markers name elements, and actions resolve a ref with
`page.locator("aria-ref=eN")`. 1.63 does not document that locator, and 1.64
introduces `get_by_ref`: the version stays pinned, a product test covers the
behaviour, and the upgrade moves to `get_by_ref`.

**A Run's browser is leased in PostgreSQL.** Every process that runs Query workers
may execute a Research Run, so the lease is shared state in the shape Run claims and
Connection refreshes already use: a claim over the configured endpoints with
`FOR UPDATE SKIP LOCKED`, taken at the Run's first browser use; an expiry renewed
from the Run's lease heartbeat; release when the Run settles; and expiry when its
worker dies. When every endpoint is leased, the call waits briefly and then fails
with a model-visible message that the browser pool is busy. The Run goes on, as it
does after any Tool error.

**A recovered Run gets a fresh browser.** The lease and the connection belong to the
worker that held the Run, and a Run recovered after a crash or a lease reclaim
leases a new browser with new contexts. `browser` calls are declared
`replay_policy="never"`, so a call pending at the crash settles `outcome_unknown`,
and nothing re-navigates to the last page on the Agent's behalf: the model reads the
unknown outcome and decides. Rendered representations, captures, and downloads that
settled are durable Resources and come back without the browser.

**DlightRAG checks the first URL; the deployment's network confines the rest.**
Before it hands a URL to the browser, DlightRAG checks it with the public-target
rules of `network_admission` and `public_http`: an HTTP(S) scheme, no embedded
credentials, and a host that resolves only to public unicast addresses. The rule
against credential and signature query parameters guards what becomes an Agent
Resource, so it applies to a Rendered Read's URL, as it does to `read(url)`, and not
to a navigation: a verification link from the inbox often carries such a parameter.
Everything the browser loads afterwards — redirects, subresources, a script's
requests, a page's own navigations — is confined by the deployment network:

- the pool containers sit only on an `internal: true` network;
- every context uses a proxy, a Squid egress container on that internal network and
  on the default network;
- Squid admits public destinations only. It denies loopback, RFC 1918, link-local
  (169.254.0.0/16, where cloud metadata lives), CGNAT, multicast, reserved ranges,
  and their IPv6 equivalents, and admits ports 80 and 443, with `CONNECT` only to
  443.

A context's proxy carries no bypass list, so loopback requests go through Squid too:
Playwright 1.63 adds `<-loopback>` to a Chromium context's proxy rules unless a
bypass names a loopback host. The connection never uses Playwright's
`expose_network`, which would route browser traffic back out through the
application's own network. A missing or wrong proxy setting therefore reaches
nothing beyond the internal network: the boundary fails closed. This is ADR 0024's
rule that egress is the deployment's to enforce, applied to the browser as it
applies to `bash`. The measured reach of a plain container is why the internal
network and the proxy are required rather than optional.

**One `browser` tool, with actions.** Its `action` is one of:

- navigation and observation: `navigate`, `snapshot`, `find`, `back`, `wait`;
- acting on the page: `click`, `type`, `select`, `press`, `scroll`, and `upload`,
  which puts a workspace file into a file input;
- content: `screenshot`, `capture`;
- accounts and mail: `register`, `login`, `inbox` (ADR 0034).

Only actions whose capability is configured are offered: `upload` needs an Agent
Workspace, `register` needs account registration, and `inbox` needs an Agent
Mailbox. The tool is not read-only, so each call runs alone (ADR 0029), and its
replay policy is `never`. `ToolResult.subject` names the current page or the
action's target. The frontend's `TOOL_VERBS` gains `browser` with its zh string,
which is the only frontend work before slice 5.

**What the model sees is bounded, and page text is context.** After a
state-changing action the tool returns a bounded accessibility snapshot,
`aria_snapshot(mode="ai")` at the configured depth, through the preview-or-spill
rule of 51,200 bytes or 2,000 lines. The wiring is explicit, because composition
wraps only injected Connection tools in that rule. With execution `disabled` there
is no Agent Workspace to spill to, and an oversized snapshot reports its full output
as unavailable, as a Connection result does. `find` searches the page for a targeted
element instead of returning the tree. A snapshot is untrusted model context, never
Evidence, like Bash output: a page becomes citable only through a Rendered Read or
`capture`. Screenshots come only from the `screenshot` action, as image attachments
that spend the Run's existing image budget (`answer.generation.max_images` with its
byte and pixel limits), which `view` shares.

**Captures and downloads are Resources.** `capture` serializes the rendered page
with `page.content()` and admits it as a Web Resource with admission origin `agent`
and acquisition `browser_capture`. The page's final URL is its citation identity,
its text comes from the same HTML converter, and the call returns the handle and the
first bounded window, as `read` does. Each capture admits its own Resource, because
what a page shows after interaction is not what its URL serves: a capture never
rebinds a URL's snapshot. A download streams back over the Playwright protocol
(`download.save_as` on the client side) and is admitted as a Resource with
acquisition `browser_download`, under the admission bounds of a fetched URL: the
per-item `max_attachment_bytes`, and no attachment slot. Neither lands in the Agent
Workspace by itself; `materialize` ([ADR 0033](0033-resource-materialization.md))
copies one there.

**Sessions are anonymous and belong to the Run.** Every context starts empty and is
discarded with its Run; cookies and storage are never persisted across Runs. The
browser never carries the owner's credentials or personal information. An account
the Agent registered (ADR 0034) is entered again in each Run's fresh session, never
restored from a stored one.

**A CAPTCHA stops the Agent.** On a CAPTCHA or any other human verification, the
Agent stops that path and reports it. DlightRAG never solves, bypasses, or
outsources one and holds no solver integration, and the tool's description states
the boundary. It has no exception.

**Every Child inherits the browser.** `browser` is capability, not authority
(ADR 0025): it is not in the forbidden table, so the computed default grants it to
each Child, which works in its own context of the Run's browser. Whether a Child's
registration outlives the Run is an authority question, and ADR 0034 answers it.

**Bounds are per operation; there is no browsing quota.** Navigation and action
timeouts, the snapshot depth with preview-or-spill, the download size, and the image
budget bound each call. There is no Web-only or browser-only cumulative quota
(ADR 0005). The pool's size bounds how many Runs browse at once across the
deployment: it is capacity, not a research allowance.

**The guidance teaches the tiers.** The `browser` and `read` descriptions say: read
pages with `read`; ask for `rendered=true` only when the text shows a JavaScript
shell; use `browser` only for multi-step interaction.

**Compose is the only implementation.** There is no Kubernetes code, manifest, or
Kubernetes-specific abstraction. Configuration follows ADR 0006.
`docker-compose.yml` owns the pool containers, the internal network, and the Squid
container. `answer.agent.browser` holds the non-secret settings — endpoints, the
egress proxy, timeouts, snapshot depth, and `account_registration: true` — with
final names fixed by the spec; `config.yaml` owns the behaviour, and because the
endpoints and the proxy address are Compose service names, `docker-compose.yml`
binds those values, as it binds the PostgreSQL host. Secrets belong in `.env`; the
Agent Browser has none, and ADR 0034's mailbox credentials are the first. `/health`
reports whether the Agent Browser is configured and performs no I/O, as it does for
every component.

**Fast is unchanged.** Fast has no tools and gains no hidden rendering (ADR 0005,
ADR 0020). The current-image link path and Corpus URL ingestion never invoke the
Extract chain, so they never reach the browser either.

## Considered options

- **playwright-mcp.** Rejected. It puts an MCP server between DlightRAG and the same
  Playwright and answers with generic tool results, which carry none of DlightRAG's
  Resource, Evidence, and settlement semantics — the reason ADR 0005 rejected Tavily
  over MCP. Its captures and downloads would land in the server's filesystem rather
  than as Run Resources, and its per-action tool catalogue would be a remote surface
  the product does not shape.
- **cua.** Rejected. It drives a whole desktop through screenshots and pointer
  coordinates. A Research Run needs a page, not a desktop: a screenshot per step
  would spend the image budget that `view` and `screenshot` share, and a coordinate
  gives the model nothing as stable as an accessibility ref.
- **A Personal MCP Connection to a browser service.** Rejected for the reasons in the
  first rule: a Connection authorizes an owner's external account and is refused at
  a private endpoint, while the browser is the deployment's and private by design.
- **Chromium in the answering container.** Rejected. ADR 0024 keeps the Agent's
  process view from growing, and the answering container reaches the host's
  PostgreSQL, API, and MCP ports through `host.docker.internal`; a browser there
  would inherit that reach.
- **Check every request in DlightRAG instead of at the network.** Rejected.
  Interception in the client cannot bind a check to the connection the browser then
  makes, which ADR 0005 requires against DNS rebinding, and requests from service
  workers and the browser's own machinery escape it. A check that fails open is not
  a boundary; the network fails closed.
- **Render on a heuristic, such as detecting a JavaScript shell.** Rejected under
  ADR 0005's rule against subjective fallback, which keeps cost and routing stable.
  The model asks with `rendered=true` when it sees a shell.
- **Replace the snapshot with the rendered page.** Rejected. A snapshot is the Run's
  fixed record of what a URL served; replacing it would change what earlier
  citations and `view` read.
- **Leave JavaScript pages to hosted extraction.** Rejected as the whole answer: it
  is optional and paid, renders on its own terms, and cannot interact. It keeps its
  place before the browser in the chain.
- **A browser per Agent Session, or one shared across Runs.** Rejected. Per session,
  a parent with eight Children would hold nine containers, while contexts already
  separate sessions. Shared across Runs, one renderer fault or exploit would cross
  Runs and owners, and recovery could not give a Run a fresh browser.
- **A persistent browser profile per owner.** Rejected. A stored profile is a cookie
  jar for every site the Agent visited — a credential with no per-account boundary
  that carries tracking identity across Runs — and it makes each Run depend on
  hidden state. Sessions start empty; ADR 0034 stores credentials per site instead.
- **A tool per action.** Rejected. The actions share one piece of state, the active
  page, so they share one sequential tool with one replay policy and one Tool
  Subject.
- **Kubernetes support now.** Rejected. The deployment is Compose on one host, and a
  second implementation of the port with no consumer would be the speculative
  abstraction ADR 0012 refused.
- **A cumulative browsing quota.** Rejected, as ADR 0005 rejected a Web-only quota:
  it truncates research arbitrarily while more expensive model work is governed
  differently.

## Consequences

Landing order, one sequence shared by ADRs 0032, 0033, and 0034:

1. **The substrate and the Rendered Read** (this decision): the pool image with both
   pins; the Compose pool, internal network, and Squid container; `BrowserProvider`
   and its PostgreSQL leases, whose suite owns a scratch database
   (`tests/support/pg`); `answer.agent.browser` and its `/health` state;
   `"browser"` in `extract_providers`; and `read(..., rendered=true)` with the
   `browser_render` representation, its durable row, and its recovery.
2. **The interactive tool, capture, downloads, and Children** (this decision): the
   `browser` tool with its bounded observations and screenshots, `capture` with
   `browser_capture`, downloads with `browser_download`, a context per Child
   Session, and the `TOOL_VERBS` entry.
3. **`materialize`** ([ADR 0033](0033-resource-materialization.md)).
4. **Accounts and the mailbox backend**
   ([ADR 0034](0034-agent-accounts-and-the-agent-mailbox.md)): `register`, `login`,
   and `inbox`.
5. **The frontend co-design** (ADR 0034): the Settings section for Agent Accounts,
   with the Profile Memory and Conversation Sessions redesign. Nothing of it is
   built earlier.

Acceptance is live, with real models, on a development deployment, without doubles:

1. A JavaScript-only page: the direct read gets a shell, `read(..., rendered=true)`
   gets the content, and the answer cites it (slice 1).
2. A page whose direct read fails is recovered by the browser at the end of the
   Extract chain (slice 1).
3. A multi-step search form on a public site, then `capture`, and the answer cites
   it (slice 2).
4. A CSV is downloaded, materialized, and computed with pandas in `bash`, and a
   chart is published with `attach_artifact` (slice 3).
5. Registration on a free site with email verification, then verification, login,
   and a gated page read; a later Run reuses the account (slice 4).
6. A parent spawns two Children that browse different sites without interfering
   (slice 2).

What changes in the code:

- The acquisition set is closed in three places that widen together: the registry
  (`direct_http` and its `_EXTRACT_ACQUISITIONS`), the `WebExtractResult` contract,
  and the executor's restore, which refuses any other acquisition on a durable Web
  row.
- `read` gains a field, so its contract version moves, and local Runs pinned to the
  old contract are reset rather than migrated.
- `WebSourceService` is process-wide and its hosted adapters stay so; the chain's
  browser entry is bound per Run, because it renders in that Run's browser.
- A Web Resource can now hold two representations, so a cursor binds the
  representation it pages as well as the Resource and the focus.
- Captures and downloads are rows of the Run's Resources, so
  [ADR 0016](0016-one-run-resource-read-surface.md)'s read surface serves them with
  no new route.
- The application image gains the Python package and its bundled driver, not a
  browser.

Live documents to revise as each slice lands: [domain language](../domain-language.md)
(Agent Browser, Rendered Read, Browser Capture, BrowserProvider, and the Web Resource
term), [resource reading](../resource-reading.md) (the `rendered` field, the new
acquisitions, and rendered HTML's conversion),
[retrieval and answer](../retrieval-answer.md) (the Research tool list and Web
Search), [security](../security.md) (the browser-session sentence and the browser's
egress boundary), [configuration](../configuration.md) (`answer.agent.browser`,
`extract_providers`, and the sentence that Research reaches external tools only
through Connections), [architecture](../architecture.md) (Agent Execution's
outside-tools sentence and the deployment's containers),
[interfaces](../interfaces.md) (the `/health` field), [operations](../operations.md)
(running and upgrading the pool), and the Status lines of ADR 0005 and ADR 0029,
which point here.

Residual risks, recorded rather than solved:

- The internal network also carries the application's connections to the pool, so a
  context created without its proxy reaches that network's peers — Squid, the other
  pool containers, and the application processes that lease them — though nothing
  beyond. DlightRAG sets the proxy on every context; a page cannot.
- Chromium runs without its own sandbox: Playwright launches it with `--no-sandbox`
  unless the client asks for the sandbox, and a server without `--unsafe` drops that
  request. The container and its network are the isolation boundary. A pool
  container serves one Run at a time, not one Run in its lifetime, so a renderer
  compromise that outlives its browser can meet the next Run that leases the
  container.
- `--max-clients 1` queues a second client instead of refusing it, and
  `chromium.connect` waits without limit by default, so a lease claimed while an
  expired holder's connection lingers waits on that connection. DlightRAG therefore
  passes a connect timeout.
- The `aria-ref=` locator is undocumented in 1.63; the pin and a product test hold
  it until `get_by_ref`.
- Rendered and captured content is the browser's assertion. A site can serve a
  browser what it does not serve a GET; a capture records state after interaction
  that its URL may not reproduce; and a script navigation during a Rendered Read
  renders where the page went, under the handle of where it started.
- A capture or a download whose URL carries a credential or signature parameter — a
  presigned download link, or a page reached from a verification link — would make
  that URL a citation identity. ADR 0005 already keeps a signed URL private and
  cites its Resource Handle instead, and the spec applies that rule here. A download
  that a script generates has no public URL of its own.
- Page content is untrusted model context. A page can try to steer the Agent, and
  the browser gives it forms as well as URLs to act through; with `trust`, `upload`
  can send a workspace file to a page. As with `bash`, the boundary is the
  deployment's egress, not a filter.
- The CAPTCHA boundary is a rule the model follows, not a mechanism: nothing detects
  a CAPTCHA, and a vision model could read one from a screenshot.
- The network facts were measured on Docker Desktop. A Linux Docker host is expected
  to behave the same and is measured before the deployment moves to one.
- The pool's size is the deployment-wide limit on Runs browsing at once, and a busy
  pool fails calls rather than queueing Runs.
