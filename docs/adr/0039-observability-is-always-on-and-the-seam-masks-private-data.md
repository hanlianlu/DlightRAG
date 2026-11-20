# Observability is always on, and the seam masks private data

## Status

Accepted and implemented. The trace seam has landed:
`langfuse_trace_sensitive_data` and `Telemetry.capture_sensitive_data` are
deleted, a call site no longer asks whether to capture and hands the seam bounded
content instead, a failed operation reports the provider's message rather than its
exception type, and the payload masker *is* the classifier this ADR defines. The
observability plane has landed too
([ADR 0040](0040-observability-lands-in-clickstack.md)): the Langfuse client is
replaced by an OpenTelemetry export into ClickStack, and the browser channel —
session replay, with `maskAllText` and `maskAllInputs` set — is live. The log
pipeline has landed with it: a record the process writes reaches ClickHouse through
the same seam, masked by the same classifier, carrying the trace and span id of the
observation that logged it, from a bounded queue a logging call never waits on.
Session replay is the one signal a deployment may sample, and a session the coin
flip passed over still reports its errors and its RUM: sampling is a volume knob,
not the off state this ADR deletes.

This decision revises the redaction clause of
[ADR 0017](0017-observability-span-contract.md): the seam stays the one place
that decides what leaves the process, but it **classifies** a value instead of
**switching content off**, the privacy configuration key is deleted rather than
renamed, and monitoring has no off state.

## Context

DlightRAG now observes two channels, and only one of them is governed by the
seam that ADR 0017 built.

- **The trace channel** is governed: one boolean decides whether an
  observation's `input` and `output` exist at all, and a masker replaces secrets
  and removes inline image bytes from everything that is exported.
- **The browser channel** is not. Session replay and RUM capture the DOM, input
  values, request and response bodies, and console output; the SDK's own masking
  is **off by default** (`maskAllText`, `maskAllInputs`). Nothing in the
  product's configuration reaches it.

A boolean cannot govern both. Suppressing content in the process leaves the
browser recording the same conversation, so the two channels would hold two
different privacy policies and the weaker one would be authoritative by
accident. A switch also makes observability a *deployment state* rather than a
property: a Run whose traces are missing because someone flipped a key is a hole
in a system whose Runs are durable, reviewable, and cost-attributed by design
([ADR 0007](0007-unified-durable-runs-and-execution-lanes.md),
[ADR 0017](0017-observability-span-contract.md)), and nobody can tell which Runs
lost their evidence after the fact.

What the trace channel carries today is deliberate, not incidental. The Run
root's input is the question (bounded to 2000 characters), its output is the
answer, a message list is bounded per message (4000 characters), a Tool's result
is a summary, and every observation keeps its counts, model, provider, duration,
status, and usage. ADR 0017 chose that so a review starts at one tree per Run
and the trace table shows the question and the answer. Privacy is not served by
turning that off; it is served by never letting a private value leave unmasked —
and that is one rule, which can be the *same* rule in both channels.

## Decision

**One classifier, applied at the one seam.** Every value that would leave the
process — an observation's `input` or `output`, a log record's message, a browser
replay field — passes one classifier that answers **private** or **exportable**,
and a private value is masked to presence, length, and digest
(`[private: 1,234 chars, sha256:ab12…]`) — except a credential, which is the one
private value that keeps no length and no digest: it becomes `[redacted]`, because a
digest of a low-entropy secret is a brute-force oracle.

- Private: credentials and tokens (the same fragments that already name a
  secret), inline image bytes, the browser's DOM text and input values, request
  and response bodies, console output, memory record bodies (already
  exported as counts only), and **a path that names a home directory** — which
  names the operator's account and the layout of a personal machine rather than the
  product's structure. A path is private as *content*, so it keeps this ADR's
  placeholder shape with its basename retained
  (`[private path: toolchain.py: 88 chars, sha256:ab12…]`): the directory, which
  carried the username, is gone, while the basename keeps a masked log line
  followable and the digest correlates records from one file. The rule reads the
  key and the value together, so it fires under any path-shaped name the
  application logs and leaves an endpoint path, a container path, or a sentence
  that merely mentions a home directory alone.
- Exportable: structural metadata — identifiers, counts, model and provider
  identity, endpoint fingerprint, durations, status, usage, error kind and error
  text, span and workspace names, and a path that names nobody — **and the
  product's bounded review content**: the Run root's question, its answer, a
  message list whose every text is bounded, and Tool summaries. Exportable content
  still passes the secret masker.
- The classifier is applied when an observation opens and when it updates, which
  is ADR 0017's property: no call site can leak by updating later.
- Masking happens where the value is shaped, never in the collector: a private
  value must not cross the network to be redacted.

**No switch.** `observability.langfuse_trace_sensitive_data` and
`Telemetry.capture_sensitive_data` are deleted, not renamed. A call site no
longer asks whether to capture: it hands the seam what it has, bounded. The only
observability configuration left is operational — destination, sampling,
retention. **Monitoring has no off state.**

**Masking fails closed and never raises.** A classification or mask that fails
yields the placeholder, never the original value, and its exception never
reaches the request path.

**The browser channel states the same rule in its own terms.** Replay stays on;
`maskAllText` and `maskAllInputs` are set; request and response bodies, console
output, and canvas recording are not captured.

**Monitoring failure never fails the product, and it is never noisy.** An
initialization that fails installs no-op telemetry and the process starts
anyway; an unreachable destination drops telemetry and nothing else, from a
bounded queue, without blocking a request, a Tool call, or the logging call
itself; a shutdown flushes under a timeout instead of waiting on a dead
collector. An outage is **reported once** — one warning when the destination
stops answering, one line when it answers again — and every failure in between
is counted, not logged, so an operator who turns the observing stack off gets a
quiet product rather than a log flood. The browser SDK is loaded off the render
path, where a failure to load or to send is ignored.

## Considered options

- **Keep the boolean and default it on.** Rejected. It cannot reach the browser
  channel, so it cannot state the rule the product needs, and a default is still
  a state a deployment can change.
- **Turn the browser channel off and keep only errors and RUM.** Rejected.
  Replay is the instrument for the front-end failures a reader reports and
  cannot reproduce; masking gives the interactions without the text.
- **Configure privacy per surface** — replay masking tiers per route, selector
  lists, per-field switches. Rejected. The operator's job becomes maintaining a
  policy matrix that drifts from the code, and a rule that needs a matrix to
  state is not a rule.
- **Mask everything the user authored, the Run's question and answer included.**
  Rejected. It deletes the review surface ADR 0017 exists to provide, for
  content that is already bounded, retained under the Run's own retention, and
  read by the same people who can read the Run.
- **Export structure only, no content at all.** Rejected. A trace that cannot
  show what was asked and answered cannot be reviewed, and the browser pane
  would hold nothing a reviewer could act on.
- **Keep `capture_sensitive_data` as a code-level flag so a call site can skip
  building a payload.** Rejected. It is the same switch under another name, and
  it puts one decision at six call sites instead of one seam.
- **Mask in the collector** (an OpenTelemetry processor before ClickHouse).
  Rejected. Private data would leave the process and cross the network first.
- **Let the deployment configure the private list.** Rejected. The list is a
  property of the product's data, not of one installation; changing it is a code
  change that passes review, which is the point.

## Consequences

- Configuration loses a key. A deployment that still sets it fails validation,
  as the key [ADR 0036](0036-traces-hold-only-product-work-and-usage-counts-each-token-once.md)
  removed does.
- A masked value stays diagnosable: presence, length, and digest say a prompt
  was long without saying what it held.
- The trace table keeps the question and the answer, and the browser pane keeps
  the interactions without the text.
- Tests hold the rule: every payload a call site can pass is classified; a
  raising masker yields the placeholder rather than the value; the browser SDK
  initializes with masking even when its own initialization throws; a failed
  initialization installs no-op telemetry; and a destination that stays
  unreachable for a sustained run produces one warning and no per-batch record
  at `WARNING` or above, in the process and in the browser console alike.
- A deployment that wants richer replay text changes the classifier and passes
  review, not a configuration key.
- This decision is independent of where the telemetry lands. It is written now
  because the browser channel does not exist yet: a rule stated before the
  channel opens costs one classifier, and the same rule stated afterwards costs
  a migration of whatever was already recorded.
- The classifier's rules are held against *real exported data*, not only against
  the exporter that produced it. The path rule above exists because a checkout path
  (`/Users/<account>/…`) and the search toolchain's binary path reached the shared
  store verbatim while the exporter, its tests, and this repository's documents all
  agreed that they should. A rule and the code that applies it can be
  self-consistently stale, so the private list is checked against the exported rows
  and the attribute names against their primary sources — the OpenTelemetry
  registries — rather than against each other.
