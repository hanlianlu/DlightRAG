# Observability is always on, and the seam masks private data

One classifier decides what leaves the process: a value is private or exportable, a private value is masked to its presence, length, and digest, and a credential is redacted without either. The privacy configuration key and `Telemetry.capture_sensitive_data` are deleted, and monitoring has no off state.

## Status

Accepted and implemented. The trace seam has landed: `langfuse_trace_sensitive_data` and `Telemetry.capture_sensitive_data` are deleted, a call site no longer asks whether to capture and hands the seam bounded content instead, a failed operation reports the provider's message rather than its exception type, and the payload masker *is* the classifier this ADR defines. The observability plane has landed too ([ADR 0040](0040-observability-lands-in-clickstack.md)): the Langfuse client is replaced by an OpenTelemetry export into ClickStack, and a record the process logs reaches ClickHouse through the same seam, masked by the same classifier, carrying the trace and span id of the observation that logged it, from a bounded queue a logging call never waits on.

This decision revises the redaction clause of [ADR 0017](0017-observability-span-contract.md): the seam stays the one place that decides what leaves the process, but it **classifies** a value instead of **switching content off**, the privacy configuration key is deleted rather than renamed, and monitoring has no off state.

## Context

DlightRAG's trace channel was governed by one boolean. `observability.langfuse_trace_sensitive_data` decided whether an observation's `input` and `output` existed at all, call sites that would have to *build* a payload asked `Telemetry.capture_sensitive_data` first, and a masker replaced secrets and removed inline image bytes from what was exported. The process's log records were a second channel the boolean never reached: they stayed on stdout, and exporting them to a store beside the traces would have put their messages and fields outside the policy the traces were under.

A boolean cannot state the rule the product needs.

- **It decides existence, not privacy.** Off, the trace table holds neither the question nor the answer, so the review surface ADR 0017 built is empty; on, everything a call site shaped is exported and the only protection is the secret-name masker. Neither state answers the question that matters, which is whether *this value* is one that may leave.
- **It makes observability a deployment state.** A Run whose traces are missing content because someone flipped a key is a hole in a system whose Runs are durable, reviewable, and cost-attributed by design ([ADR 0007](0007-unified-durable-runs-and-execution-lanes.md), [ADR 0017](0017-observability-span-contract.md)), and nobody can tell which Runs lost their evidence after the fact.
- **A second channel would have a second policy.** Suppressing content in the trace seam while a log record carries the same text leaves two privacy policies, and the weaker would be authoritative by accident.

What the trace channel carries today is deliberate, not incidental. The Run root's input is the question (bounded to 2000 characters), its output is the answer, a message list is bounded per message (4000 characters), a Tool's result is a summary, and every observation keeps its counts, model, provider, duration, status, and usage. ADR 0017 chose that so a review starts at one tree per Run and the trace table shows the question and the answer. Privacy is not served by turning that off; it is served by never letting a private value leave unmasked, and that is one rule, which can be the *same* rule for a span and a log record.

## Decision

**One classifier, applied at the one seam.** Every value that would leave the process — an observation's `input` or `output`, a log record's message and attributes — passes one classifier that answers **private** or **exportable**, and a private value is masked to presence, length, and digest (`[private: 1,234 chars, sha256:ab12…]`) — except a credential, which is the one private value that keeps no length and no digest: it becomes `[redacted]`, because a digest of a low-entropy secret is a brute-force oracle.

- Private: credentials and tokens (the same fragments that already name a secret), inline image bytes, memory record bodies (already exported as counts only), and **a path that names a home directory** — which names the operator's account and the layout of a personal machine rather than the product's structure. A path is private as *content*, so it keeps this ADR's placeholder shape with its basename retained (`[private path: toolchain.py: 88 chars, sha256:ab12…]`): the directory, which carried the username, is gone, while the basename keeps a masked log line followable and the digest correlates records from one file. The rule reads the key and the value together, so it fires under any path-shaped name the application logs and leaves an endpoint path, a container path, or a sentence that merely mentions a home directory alone.
- Exportable: structural metadata — identifiers, counts, model and provider identity, endpoint fingerprint, durations, status, usage, error kind and error text, span and workspace names, and a path that names nobody — **and the product's bounded review content**: the Run root's question, its answer, a message list whose every text is bounded, and Tool summaries. Exportable content still passes the secret masker.
- The classifier is applied when an observation opens and when it updates, which is ADR 0017's property: no call site can leak by updating later.
- Masking happens where the value is shaped, never in the collector: a private value must not cross the network to be redacted.

**No switch.** `observability.langfuse_trace_sensitive_data` and `Telemetry.capture_sensitive_data` are deleted, not renamed. A call site no longer asks whether to capture: it hands the seam what it has, bounded. The only observability configuration left is operational — destination, sampling, identity. **Monitoring has no off state.**

**Masking fails closed and never raises.** A classification or mask that fails yields the placeholder, never the original value, and its exception never reaches the request path.

**Monitoring failure never fails the product, and it is never noisy.** An initialization that fails installs no-op telemetry and the process starts anyway; an unreachable destination drops telemetry and nothing else, from a bounded queue, without blocking a request, a Tool call, or the logging call itself; a shutdown flushes under a timeout instead of waiting on a dead collector. An outage is **reported once** — one warning when the destination stops answering, one line when it answers again — and every failure in between is counted, not logged, so an operator who turns the observing stack off gets a quiet product rather than a log flood.

## Considered options

- **Keep the boolean and default it on.** Rejected. It cannot state a rule that treats a question and a credential differently, and a default is still a state a deployment can change.
- **Configure privacy per surface** — switches per call site, per field, or per channel. Rejected. The operator's job becomes maintaining a policy matrix that drifts from the code, and a rule that needs a matrix to state is not a rule.
- **Mask everything the user authored, the Run's question and answer included.** Rejected. It deletes the review surface ADR 0017 exists to provide, for content that is already bounded, retained under the Run's own retention, and read by the same people who can read the Run.
- **Export structure only, no content at all.** Rejected. A trace that cannot show what was asked and answered cannot be reviewed.
- **Keep `capture_sensitive_data` as a code-level flag so a call site can skip building a payload.** Rejected. It is the same switch under another name, and it puts one decision at every call site instead of one seam.
- **Mask in the collector** (an OpenTelemetry processor before ClickHouse). Rejected. Private data would leave the process and cross the network first.
- **Let the deployment configure the private list.** Rejected. The list is a property of the product's data, not of one installation; changing it is a code change that passes review, which is the point.

## Consequences

- Configuration loses a key. A deployment that still sets it fails validation, as the key [ADR 0036](0036-traces-hold-only-product-work-and-usage-counts-each-token-once.md) removed does.
- A masked value stays diagnosable: presence, length, and digest say a prompt was long without saying what it held.
- The trace table keeps the question and the answer.
- Tests hold the rule: every payload a call site can pass is classified; a raising masker yields the placeholder rather than the value; a failed initialization installs no-op telemetry; and a destination that stays unreachable for a sustained run produces one warning and no per-batch record at `WARNING` or above.
- A deployment that wants richer content in a trace changes the classifier and passes review, not a configuration key.
- This decision is independent of where the telemetry lands. It is written with the export it governs, because a rule stated after a channel opens costs a migration of whatever was already recorded.
- The classifier's rules are held against *real exported data*, not only against the exporter that produced it. The path rule exists because a checkout path (`/Users/<account>/…`) and a toolchain's binary path are exactly what a log record's code location and a toolchain's own logging carry, and a rule and the code that applies it can be self-consistently stale: the private list is checked against the exported rows and the attribute names against their primary sources — the OpenTelemetry registries — rather than against each other.
