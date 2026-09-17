# Caller-chosen agent effort

One Answer request may name the effort its own answering agent runs at, from the three levels the product offers (`low`, `high`, `max`). The choice rides the shared client contract, is recorded in the accepted Run's input, re-levels **only** that Run's answering role, and leaves every child agent on its own configured level. A Run that names no effort keeps using the deployment's configured level.

## Status

Accepted and implemented. Level names, request field, bootstrap offer, pinning, and the composer control landed together; `docs/interfaces.md` and `docs/configuration.md` are revised with it.

## Context

`agentic_reasoning` is deployment configuration owned by `models.chat.roles.<role>` ([ADR 0006](0006-configuration-ownership-and-deployment-bindings.md)): one level per role, fixed at startup, with `reasoning` inheritance and an explicit `null` disabling typed agentic reasoning. Every caller therefore shares one effort, even though the cost/latency of a Research answer is a per-question decision — a quick factual lookup and a long multi-source investigation are submitted from the same composer.

The engine already knows how to run an agent at a level other than the configured one: `resolve_reasoning` maps a requested level onto what a model profile supports, clamping to the nearest supported level and preferring the higher one on a tie, and never clamping *down* to `off`. What was missing was a way for a caller to ask for a level at all, and a place for that ask to live durably.

Two facts about the current engine shape constrain the answer.

**The Run's pinned reasoning settings mirror the deployment, and validation compares them.** `PinnedModelProfile.reasoning_settings` records each role's `{"ordinary", "agentic"}` pair and execution refuses a Run whose pinned pair no longer matches the configured pair ([ADR 0007](0007-unified-durable-runs-and-execution-lanes.md)). Writing a caller's choice into that pin would make the Run permanently incompatible with its own deployment: the next accepted Run would pin the deployment pair again, and a config change would strand the choosing Run for the wrong reason.

**Child roles resolve by name from the same role table, and `query` is the default child role.** `ChildModelRole` includes `query`, `resolve_child_model` re-checks each child's binding against its pin, and `model_role` defaults to `query`. Overriding the `query` role at the pin would therefore either re-level subagents — which the product does not promise — or fail their binding check outright. The answering agent and a child that happens to ask for the `query` role are different consumers of one name.

## Decision

**The choice is a request field, not a pinned profile.** `AnswerRequestContract` — the one client-facing request vocabulary shared by REST, Web, and MCP — gains `effort: "low" | "high" | "max"`. `client_contracts.normalize_answer_effort` is the single entry point that decides what the three levels are, so the browser control, every public transport, and the engine's durable input cannot drift apart. An unsupported value is refused at the boundary with a validation error; it is never silently downgraded to a different level.

**The accepted Run records it.** The field survives into the immutable `AnswerRunInput` and its prepared payload, and it is part of the submission's idempotency fingerprint, so the same submission id with a different effort is a conflict rather than a replay of the earlier answer. Recovery replays the level from the durable input, never from current configuration. The input is the audit surface for as long as the store retains it: a terminal Run's prepared input is pruned by the existing retention rule, which keeps the fingerprint.

**Only the answering role is re-leveled, for that Run only.** `ModelRuntime.tool_model(role, agentic_reasoning=…)` builds a wrapper whose `agentic_reasoning` field is explicitly set, and caches it under `(role, level)` so a re-leveled Run cannot change what another Run or another role reads. `prepare_orchestrated_run(agent_effort=…)` applies the level to the `query` role's wrapper that the Run hands its orchestrator. Children keep the deployment default: their pins, their binding check, and their resolution path are untouched, and a child that asks for `query` gets the configured level. Pinned reasoning settings stay the deployment's own pair, so execution-time validation keeps its current, strict meaning.

**Fallback stays the engine's business.** The request boundary accepts exactly three levels; what a *model* supports is a profile fact, so `resolve_reasoning` keeps clamping a level the endpoint cannot express, preferring the higher neighbour and never choosing `off`. `effort` is never a whitelist per deployment.

**Refined after review (2026-09-16).** The accepted set stays the three levels and clamping still covers a ladder that stops below one; three facts around it were too loose to leave as they were. The Web control now offers the efforts the answering profile can actually express (`offered_answer_efforts`), so a model whose ladder ends at `high` never advertises `max` — the request boundary and the clamp are unchanged, only the promise the picker makes. Two configurations refuse the choice at admission instead of failing later. An answering model that names no non-off level has nothing to clamp to; and a role that owns its reasoning through `agentic_model_kwargs` cannot take a typed level at all — applying one would bypass the single-owner validation (`model_copy` re-validates nothing) and raise where the request is planned, which reaches the caller as a provider rejection on a run that already started. An effort a model cannot express *at all* — a catalogued profile that names no non-off level — is refused at admission with `unsupported_effort`, because there is nothing to clamp to and admitting it either answered silently without the requested thinking or failed inside a provider call, reporting a configuration fact as a provider rejection. And the run trace states `agent_effort` as `{requested, effective}`, because the stored `effort` alone misreports a clamped run and says nothing at all about a Fast answer, which enters no agent loop.

**The deployment's own level is reported, not invented.** Bootstrap (contract version 3) carries `agent_effort: {levels, default}`, where `default` is the configured agentic level for the answering role *when it is one of the three*, and `null` otherwise. The composer shows that level, marks it `Default`, and stores nothing until the caller picks one — so a caller who never touches the control runs exactly what the deployment configured, including a level the three-level control cannot name.

**Where the control lives.** The composer's mode switcher gains a sibling trigger in the same visual language: a menu of the offered levels, `menuitemradio` rows, arrow/Home/End/Escape keys, and a `Default` marker on the deployment's level. It is hidden while the mode is `fast`, because a Fast answer runs no agent turns at all, and a Fast submission carries no effort. The stored choice is the caller's own override and survives switching modes.

## Considered options

- **Pinning the caller's choice in `PinnedModelProfile.reasoning_settings`.** Rejected: the pin is the deployment's configuration fact and is validated as such, so an override there either strands the Run on the next config change or forces a second, weaker validation rule for one role.
- **Letting the choice propagate to subagents.** Rejected: a subagent's level is part of what the deployment configured for that child role, spawn guidance already publishes those effective levels, and silently re-leveling children would make one caller's choice change another agent's cost and behavior.
- **A per-deployment whitelist of allowed levels.** Rejected: the endpoint's support is a model-profile fact the engine already resolves by clamping, and a second configuration knob would create two places that decide what a level means.
- **A five-level control (`minimal … max`).** Rejected: the product offers three deliberate steps; a longer ladder multiplies labels, CLAUDE-like parity questions, and UI states without a measured need.
- **Keeping the control visible but disabled in Fast mode.** Rejected: Fast runs no agent turns, so the setting cannot apply, and a disabled control needs its own explanation where hiding needs none.
- **Storing the deployment default as the caller's choice on first render.** Rejected: it converts a deployment fact into a durable user preference the caller never made, and it would freeze today's default into every browser.

## Consequences

Effort becomes an auditable part of one accepted Run: the prepared input carries it, the idempotency fingerprint distinguishes it, and recovery replays it. Deployment configuration keeps exactly one owner for the default level, and no new configuration key was added.

`docs/interfaces.md` gains the request field and the bootstrap capability; `docs/configuration.md` points the default at the new caller-facing override. The Web bootstrap contract version moves 2 → 3 because the capability is required, which is the same atomic bump the personal-Connections capability made ([ADR 0012](0012-personal-connections-and-hot-plug.md)).

Residual risks, to revisit rather than extend silently: a strict endpoint refuses a level it cannot express — best-effort profiles mark every level supported, so the request is sent and the provider judges — and that refusal is now classified (`is_provider_reasoning_rejection`) so the run failure names the reasoning control and its remedy instead of reading as a generic provider rejection; a caller can now raise the per-Run cost ceiling the deployment chose, so a per-owner cap on high-effort Runs is a candidate if measurement shows misuse; the three-level vocabulary is a product decision, so a fourth step needs its own argument and its own labels; and a deployment whose configured level is outside the three shows no `Default` marker, which is deliberate but means the control starts unselected there.

Stop conditions: effort selection being used to bypass a deployment's cost intent rather than express it; child-role levels being requested through this field; or measurements showing the level makes no difference to answer quality at the top of the ladder, which would argue for dropping the highest step instead of adding more.
