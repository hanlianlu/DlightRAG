# Response API Family qualification

This record separates implementation evidence from endpoint qualification and
rollout. [ADR 0027](adr/0027-api-family-selects-the-provider-wire.md) owns the
contract; [Configuration](configuration.md#api-family) owns the opt-in settings.

## Decision

**The common transport is implemented; rollout is not cleared.** Keep the
configured roles on `chat_completion`. Neither a healthy process nor a passing
mock qualifies a live endpoint, and a passing retry does not erase a failed
canary. No family, capability, reasoning, or privacy fallback was introduced.

| Scope | Evidence / disposition |
|---|---|
| Common Response transport, five entrypoints, Tool/image ownership, invocation identity | Implemented and covered by SDK-backed HTTP/SSE contracts |
| Recovery, Fork, whole-exchange compaction, provider-anchored accounting | Product-seam regression tests; no new Session or compaction authority |
| Direct DeepSeek `deepseek-flash` | Live text, JSON Object, stream, reasoning replay, two-call loop, user image and stream close passed; Tool-image answer varied across repeated samples. **Rollout held** |
| OpenRouter `z-ai/glm-5.3-flash` | Independently exercised; a full reasoning/two-call loop passed, but another returned text without the requested calls. **Rollout held** |
| Official OpenAI | Experimental, installed-SDK mock HTTP/SSE tested; **no official live API key, not live-qualified** |
| ZDR | Not established by any of these tests; `store=false` is remote-state minimization only |

## Live method and bounds

The 2026-09-20 UTC session ran inside the API container rebuilt after `76d0040b`,
using OpenAI Python SDK **3.3.0** and image
`sha256:b31806806756bef16496c4d45dfe04c9fa9294914abe9f4bddfc1920f5d33fbd`.
Later mainline changes through `fbc40646` concerned answer links, not this provider.

Endpoints were fixed, not discovered or silently changed:

- Direct DeepSeek: `https://api.deepseek.com/responses`, model `deepseek-flash`.
  The `/v1/responses` alias was not tested.
- OpenRouter: `https://openrouter.ai/api/v1/responses`, model
  `z-ai/glm-5.3-flash`; this does not qualify any other OpenRouter model or upstream.
- Chat comparisons used the same root and exact model, with `/chat/completions`.

Only synthetic arithmetic, small locally generated solid-color PNGs and local
read-only function results were sent. Existing deployment credentials stayed in
process memory. No corpus, real user data, credentials, remote IDs, model text,
or reasoning payloads were persisted in the evidence. Request hooks checked
exact URLs and, for Response, `store=false`, `background=false`, disabled
truncation, full input, and absence of remote continuation fields. They checked
that earlier reasoning items/fields survived later tool-bearing requests exactly.

Each matrix invocation allowed at most **16 HTTP attempts per endpoint/family**,
**zero SDK retries**, a **60-second request timeout**, a 75-second case deadline,
and **2048 maximum output tokens per request**. The corrected full matrix made
35 attempts. Separate bounded smoke and diagnostic invocations are not included
in that total. The reasoning replay case selected typed `high`; other matrix
cases selected typed `low`. Defaults and running application configuration were
not changed. The local functions returned fixed numbers or a generated image;
the durable-effect tests, not this harness, prove restart settlement semantics.

Safe local evidence and the runnable harness are kept under the gitignored
`docs/research/` directory: `response-live-qualification.py`,
`response-live-qualification-final.jsonl`, `response-live-smoke-corrected.jsonl`,
`response-reasoning-high.jsonl`, `response-image-diagnostic.jsonl`,
`response-image-recheck.jsonl`, and the four `*-image-unhinted-*.jsonl` records.
These are local evidence, not files distributed by a checkout. For example:

```sh
docker compose exec -T dlightrag-api python - deepseek response image_tool_output \
  < docs/research/response-live-qualification.py
```

## Results and limitations

### Direct DeepSeek

The full local-history loop replayed reasoning from an earlier assistant turn
that made **no** Tool call, then made two distinct calls, consumed two local
results, and returned the correct sum. The final Response input had two native
reasoning items and two `function_call_output` items, with no response ID.

The repeated Tool-image case is still a falsifier: **two passes and two failures**
across the initial matrix, two targeted checks, and corrected matrix. In the
instrumented failure the endpoint returned HTTP 200, a completed `stop` turn,
zero further calls, and an incorrect color answer. It was not a token-limit
incomplete response: usage was 543 input / 59 output tokens, including 57
reasoning tokens. The wire retained one matched Tool output. No error was hidden
and no synthetic user image was added to make Response succeed.

The same locally generated image was identified correctly in separate unhinted
user-image checks on both families. This narrows the remaining observation to
the Tool-image task. A further unhinted blue/yellow/red differential used the
same generated pixels in user and Tool positions: user input scored 3/3 on both
families, Tool output scored 1/3 on Chat and 2/3 on Response. All were HTTP 200
completed turns. The failure is therefore **not isolated to the new Response
wire**, and a Chat fallback would not cure this workload. The 14-attempt probe
is recorded in local `response-image-matrix.py` / `response-image-matrix.jsonl`.
It does **not** establish whether model variability or provider-side normalization
caused the failure. The adapter remains unchanged
until a protocol defect has a falsifying contract test. Do not describe this
endpoint's whole multimodal Tool workload as qualified yet.

### OpenRouter

The corrected high-reasoning replay check passed a three-request loop, retained
the predecessor's native reasoning and returned the correct sum after two calls.
In the later full matrix, the second request instead returned a completed text
turn with **zero calls** (HTTP 200, 430 input / 56 output tokens). `tool_choice=auto`
does not guarantee that a model follows an instruction to call tools. No local
effect executed in that failed case. This is a workload-quality failure, not
evidence that the SDK lost a call; its cause remains unresolved.

User-image and Tool-image tasks passed in both families. OpenRouter owns upstream
selection and normalization; no upstream pinning or routing-policy change was
made, so these results are not direct-DeepSeek parity or stable cross-route proof.

### Harness corrections, not product fixes

The first smoke harness incorrectly accessed `CompletionOutput.text` instead of
its string value; those four results are invalid as endpoint verdicts. Corrected
smokes passed on both families for both endpoints.

Earlier matrix assertions also incorrectly required displayable reasoning on
every Tool turn and a nonempty *first* stream chunk. An endpoint can return only
opaque `reasoning_details`, produce no new reasoning on a particular turn, or
emit an empty initial chunk. The corrected harness instead requires and verifies
native predecessor replay, and closes after the first nonempty text delta. The
initial user-image prompt named the expected color; its later **unhinted** checks
are the image evidence. These corrections did not change provider code and do
not erase the genuine Tool-image or Tool-selection failures above.

### Cost, latency and input comparison

These are single synthetic samples, not a benchmark or rollout recommendation:

| Case | Chat | Response |
|---|---|---|
| DeepSeek three-request Tool loop latency | 2.543 s | 2.850 s |
| DeepSeek loop input counts | 52 / 418 / 496 | 52 / 429 / 506 |
| DeepSeek last-loop-request cached input | 256 | 256 |
| OpenRouter JSON Object latency | 3.131 s | 15.775 s |
| OpenRouter JSON Object reported cost | 0.000016779 | 0.000015900 |
| OpenRouter successful Tool-image latency | 3.675 s | 3.357 s |

DeepSeek returned no cost field, so no cost estimate is invented. OpenRouter costs
are the endpoint's reported totals, not an invoice or a controlled comparison;
its routing and generated reasoning differed between samples. Local estimates
and provider input remain under the existing accounting contract, covered by the
Slice 6 systematic-undercount regression. These short canaries do not prove
long-session accounting accuracy or compaction economics.

Delivery review found that flattened `input_tokens_details.cached_tokens` was
retained in usage but not recognized by the shared cache-hit reader. Two new
falsifiers (reported zero and a 9,000-token hit) failed before adding that field
to the existing mapping. The Run trace regression now covers both Chat and
Response spellings, including the cold-turn warning. This is a usage-dialect fix,
not a new estimator, context policy, or compaction algorithm; the earlier raw
live usage figures above are unchanged.

Closing a local stream also does not prove the upstream stopped computing or billing. Deterministic
error, cancellation and scheduler-release contracts remain offline tests; no
live overload or deliberate service failure was induced.

## Official OpenAI mock contract

The SDK-backed tests use an official OpenAI `/v1/responses` URL with only the HTTP
transport replaced. They cover text/structured output and image-bearing function
outputs across the existing entrypoints. Dedicated replay tests preserve both
`commentary` and `final_answer` phases from a non-Tool turn, and preserve finalized
encrypted reasoning rather than the partial ciphertext from `output_item.added`.
A test-only mutation that drops `phase` makes both streaming and non-streaming
replay tests fail; the unmodified implementation already preserves these fields.

SDK JSON and typed SSE fixtures also prove refusal/failed/filtering terminals
raise, while `incomplete(max_output_tokens)` exposes neither partial calls nor
replay state. No production transport change was needed for this coverage.
Official OpenAI live qualification remains deferred until an operator supplies an
official key; Codex login and OpenRouter-hosted models are not substitutes.

## Remaining gate

Resolve and requalify the unstable live workloads before reviewing any Query
rollout; ordinary roles, VLM and reranking follow independently. This is a held
qualification, not permission to drop image ownership, force remote state, relax
Tool validation, or change the Session/compaction design. Implementation/CI
completion and this live rollout gate must be reported separately.
