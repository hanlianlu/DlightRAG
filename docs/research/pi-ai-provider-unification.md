# How pi-ai (`@earendil-works/pi-ai`) unifies "thinking/reasoning" across LLM providers

**Research date:** 2026-08-29
**Source:** https://github.com/earendil-works/pi, shallow clone of commit `853a80d26c90a14c1886f0ebb8ffaae133ca2185` (2026-08-28, "Add [Unreleased] section for next cycle").
**Package version:** `@earendil-works/pi-ai` **0.84.4** (released 2026-08-28, `/tmp/pi-earendil/packages/ai/package.json:4`; CHANGELOG entry at `packages/ai/CHANGELOG.md:5`). The changelog traces back to 0.9.4 "Initial release with multi-provider LLM support" (2025-11-26, `packages/ai/CHANGELOG.md:2025-2027`); older entries reference the repo's former home `badlogic/pi-mono`, confirming the monorepo moved but is the same lineage.
**Scope of reading:** `packages/ai/src/types.ts`, `src/models.ts`, `src/api/*` (all nine adapters + shared helpers), `src/providers/*`, `scripts/generate-models.ts`, `packages/agent/src`, plus `packages/ai/README.md`, `packages/ai/CHANGELOG.md`, `packages/coding-agent/docs/models.md`, pi.dev docs, and rfc.earendil.com RFCs tagged `pi`.

---

## TL;DR

- **Response side is called "thinking".** The unified assistant message carries `ThinkingContent { type: "thinking", thinking: string, thinkingSignature?: string, redacted?: boolean }` (`packages/ai/src/types.ts:356-364`), streamed as `thinking_start`/`thinking_delta`/`thinking_end` events (`types.ts:540-542`). Every provider's reasoning surface — OpenAI Responses `reasoning` items, Anthropic `thinking`/`redacted_thinking`, Gemini `thought: true` parts, OpenAI-compat `reasoning_content`/`reasoning`/`reasoning_details` — is normalized into this one block type.
- **Request side has a two-layer vocabulary.** The simple/unified API knob is `SimpleStreamOptions.reasoning?: ThinkingLevel` with `ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh" | "max"` (`types.ts:83`, `types.ts:314-322`) — notably the *option is named* `reasoning` while its *values are called* thinking levels. `"off"` is *not* a value here: omitting `reasoning` means off. The higher-level agent package adds the `"off"` literal in its own `thinkingLevel` state and converts it to `undefined` before calling pi-ai (`packages/agent/src/types.ts:301`, `agent/src/agent.ts:450`).
- **The bridge between the six pi levels and every provider shape is per-model metadata**, not adapter hardcoding: `Model.reasoning: boolean` (capability flag) + `Model.thinkingLevelMap?: Partial<Record<"off" | ThinkingLevel, string | null>>` (`types.ts:827-832`), generated from models.dev / OpenRouter reasoning metadata (`scripts/generate-models.ts:1217`, `scripts/openrouter-reasoning-options.ts:12-23`), with runtime clamping via `getSupportedThinkingLevels()` / `clampThinkingLevel()` (`src/models.ts:902-946`).
- **Effort string, not a token budget, is the currency.** Token budgets are a fallback: shared defaults `minimal: 1024, low: 2048, medium: 8192, high: 16384` (`src/api/simple-options.ts:57-62`) are used only for budget-based surfaces (older Anthropic, Gemini 2.5, Bedrock budget mode). `xhigh`/`max` are opt-in per model and clamp down to `high` on budget-only providers (`simple-options.ts:64-66`).
- **OpenAI-compatible endpoints are the hard part** — pi-ai defines 11 `thinkingFormat` variants (`openai`, `openrouter`, `deepseek`, `together`, `baseten`, `zai`, `qwen`, `qwen-chat-template`, `chat-template`, `string-thinking`, `ant-ling`) so the same `reasoning` knob produces `reasoning_effort`, `reasoning: { effort }`, `thinking: { type }`, `enable_thinking`, `chat_template_kwargs`, or string thinking as each server requires (`types.ts:578-590`, `src/api/openai-completions.ts:857-950`).
- **Replay is signature-keyed.** Each adapter stores its provider's opaque replay payload in `thinkingSignature` (Anthropic signature, OpenAI `ResponseReasoningItem` JSON with `encrypted_content`, Gemini `thoughtSignature`, or a field-name/`reasoning_details` marker) and re-serializes it on the way back in; cross-model history degrades thinking blocks to plain text (`src/api/transform-messages.ts:101-116`).
- **Escape hatches are deliberate and narrow:** provider-specific full option sets via `models.stream()`/`hasApi()` narrowing, `samplingParams` merged last over named fields, `onPayload` payload rewriting, `headers`, and per-model `compat` flags (`types.ts:179-225`, `types.ts:557-625`).
- **No RFC or doc states a "thinking vs reasoning" naming rationale.** RFC 0054 ("Responses Lite Investigation") discusses Responses transport but not reasoning vocabulary (https://rfc.earendil.com/0054/). The README sidesteps the choice with a "Thinking/Reasoning" section title (`packages/ai/README.md:786`). The evolution (visible only through the CHANGELOG) went: hardcoded `ReasoningEffort` + `supportsXhigh()` → `compat.reasoningEffortMap` → top-level `Model.thinkingLevelMap`.

---

## 1. The unified model (response side)

### 1.1 Core content types

pi-ai's message model has exactly three assistant content block types: `TextContent`, `ThinkingContent`, `ToolCall` (`types.ts:427-447`, `AssistantMessage.content: (TextContent | ThinkingContent | ToolCall)[]`). The thinking block (`packages/ai/src/types.ts:356-364`):

```ts
export interface ThinkingContent {
	type: "thinking";
	thinking: string;
	thinkingSignature?: string; // Provider-specific opaque or serialized reasoning replay data
	/** When true, the thinking content was redacted by safety filters. The opaque
	 *  encrypted payload is stored in `thinkingSignature` so it can be passed back
	 *  to the API for multi-turn continuity. */
	redacted?: boolean;
}
```

Design points:

- **One generic string field `thinkingSignature` doubles as the replay envelope.** Its meaning is per-adapter: an Anthropic signature string, a JSON-serialized OpenAI `ResponseReasoningItem` (including `encrypted_content`), a Gemini `thoughtSignature`, or — on OpenAI-completions — even just the name of the response field to echo reasoning back into (`"reasoning_content"` / `"reasoning"`; `src/api/openai-completions.ts:610-614`).
- **Redaction is a boolean, not a separate block type.** Anthropic's `redacted_thinking` becomes `ThinkingContent` with `thinking: "[Reasoning redacted]"`, `thinkingSignature: <opaque data>`, `redacted: true` (`src/api/anthropic-messages.ts:629-634`; placeholder constant shared with Bedrock at `src/api/bedrock-converse-stream.ts:113-114`).
- **Text blocks can also carry reasoning metadata.** `TextContent.textSignature` holds either a legacy id string or a versioned `TextSignatureV1 { v: 1, id, phase?: "commentary" | "final_answer" }` so OpenAI Responses message ids/phases survive replay (`types.ts:344-354`, `src/api/openai-responses-shared.ts:49-64, 700`).
- **Tool calls carry Google's thought signature**: `ToolCall.thoughtSignature` ("Google-specific: opaque signature for reusing thought context", `types.ts:377`), which must be preserved *on the tool call* for Gemini multi-turn correctness (documented protocol note at `src/api/google-shared.ts:54-69`).
- **Token accounting has a reasoning slot**: `Usage.reasoning?: number` — "Reasoning/thinking tokens… a subset of `output`" — populated from Anthropic `output_tokens_details.thinking_tokens`, OpenAI `reasoning_tokens`, and Gemini `thoughtsTokenCount` (`types.ts:389-394`; adapter examples at `src/api/anthropic-messages.ts:761-766`, `src/api/google-generative-ai.ts:232`).
- **Capability metadata lives on the model**, not the message: `Model.reasoning: boolean` and `Model.thinkingLevelMap` (`types.ts:827-832`).

### 1.2 Per-provider normalization into `ThinkingContent`

| Provider wire shape | Normalized to | Where |
|---|---|---|
| Anthropic `content_block_start type=thinking` + `signature_delta` | `ThinkingContent` with `thinkingSignature` accumulated from `signature_delta` (`block.thinkingSignature += event.delta.signature`) | `src/api/anthropic-messages.ts:624, 695-696` |
| Anthropic `redacted_thinking` | `ThinkingContent { thinking: "[Reasoning redacted]", thinkingSignature: data, redacted: true }` | `src/api/anthropic-messages.ts:629-634` |
| OpenAI Responses `response.output_item` `type: "reasoning"` | One `ThinkingContent` slot per reasoning item; text from `summary[].text` or `content[].text`; **`thinkingSignature = JSON.stringify(item)`** (whole item incl. `encrypted_content`) | `src/api/openai-responses-shared.ts:463-476, 602-627, 685-696` |
| OpenAI Responses `reasoning.encrypted_content` arriving only in the terminal event (Azure quirk) | Patched into the stored item JSON before finalizing | `src/api/openai-responses-shared.ts:533-548` |
| OpenAI-completions delta fields `reasoning_content` / `reasoning` / `reasoning_text` | First non-empty field wins (dedup for chutes.ai-style double-send); field name recorded in `thinkingSignature` | `src/api/openai-completions.ts:592-618` |
| OpenRouter `reasoning_details[]` (`reasoning.text` / `reasoning.summary` / `reasoning.encrypted`) | Streamed into thinking text; full array JSON stored in `thinkingSignature` for verbatim replay | `src/api/openai-completions.ts:205-275, 323-337, 655-668` |
| Gemini part with `thought: true` | `ThinkingContent`; `thoughtSignature` retained via `retainThoughtSignature` (first-delta rule) | `src/api/google-shared.ts:68-69, 73-80`; `src/api/google-generative-ai.ts:140-156` |
| Bedrock `reasoningContent` (`text` / `signature` / `redactedContent`) | Same `ThinkingContent`; `redactedContent` (non-Anthropic models, e.g. GPT-5.6 on Bedrock) buffered as bytes into `thinkingSignature` with `redacted: true` | `src/api/bedrock-converse-stream.ts:613-660` |
| z.ai / DeepSeek etc. via `thinking: { type }` servers (completions wire) | Same `reasoning_content` path as above (they are OpenAI-compatible) | `src/api/openai-completions.ts:592-618` |

### 1.3 What "off" means on the response side

For reasoning-capable models where thinking cannot be fully disabled, pi-ai sends the *lowest* supported level but omits `includeThoughts`, so hidden thinking never surfaces as blocks. Gemini example: "Gemini 3.1 Pro cannot disable thinking… For Gemini 3 models, use the lowest supported thinkingLevel without includeThoughts so hidden thinking remains invisible to pi" — Gemini 3.1 Pro → `{ thinkingLevel: "LOW" }`, 3 Flash/Gemma 4 → `{ thinkingLevel: "MINIMAL" }`, Gemini 2.x → `{ thinkingBudget: 0 }` (`src/api/google-generative-ai.ts:428-447`).

---

## 2. Request-side control: the exact vocabulary

### 2.1 The simple (unified) API

Verbatim, `packages/ai/src/types.ts:313-322`:

```ts
// Unified options with reasoning passed to streamSimple() and completeSimple()
export interface SimpleStreamOptions extends StreamOptions {
	/** Provider-neutral tool selection for simple requests. When omitted, adapters use provider-specific behavior. */
	toolChoice?: ToolChoice;
	reasoning?: ThinkingLevel;
	/** Ask a capable provider to return a durable handle and continue the request asynchronously. */
	deferred?: boolean | { window?: "15m" | "1h" | "24h" };
	/** Custom token budgets for thinking levels (token-based providers only) */
	thinkingBudgets?: ThinkingBudgets;
}
```

with `types.ts:83-105`:

```ts
export type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
export type ModelThinkingLevel = "off" | ThinkingLevel;
export type ThinkingLevelMap = Partial<Record<ModelThinkingLevel, string | null>>;
...
/** Token budgets for each thinking level (token-based providers only) */
export interface ThinkingBudgets {
	minimal?: number;
	low?: number;
	medium?: number;
	high?: number;
}
```

Key facts:

- The unified knob is **`reasoning`** (per-call option, not per-message and not model config), of type **`ThinkingLevel`** — a six-value *ordered effort scale*, not a boolean and not a budget. There is **no `"off"` member**: absent == off. (`SimpleStreamOptions` is also the base of the agent loop's `AgentLoopConfig`, `packages/agent/src/types.ts:149`.)
- `"off"` exists only in the *model-metadata* vocabulary (`ModelThinkingLevel`) and in the agent layer. The agent package defines `ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"` (`packages/agent/src/types.ts:296-301`) and converts: `reasoning: this._state.thinkingLevel === "off" ? undefined : this._state.thinkingLevel` (`packages/agent/src/agent.ts:450`). So the user-facing off-switch is `thinkingLevel: "off"` at the agent, and `undefined`/omitted at the pi-ai boundary.
- Every adapter's `streamSimple` starts the same way: `const clampedReasoning = options?.reasoning ? clampThinkingLevel(model, options.reasoning) : undefined` — e.g. `src/api/openai-responses.ts:209`, `src/api/openai-completions.ts:732`, `src/api/google-generative-ai.ts:315`, `src/api/anthropic-messages.ts:836-847` (via `options?.reasoning` guard), `src/api/mistral-conversations.ts:194`, `src/api/openai-codex-responses.ts:505`. If the model has `reasoning: false`, "options are silently ignored" (`packages/ai/README.md:788`).

### 2.2 The three-level clamp pipeline

1. **Model-supported levels.** `getSupportedThinkingLevels(model)` returns `["off"]` for non-reasoning models; otherwise all levels through `high` unless the map nulls them, plus `xhigh`/`max` only if the map explicitly maps them (`src/models.ts:900-911`):

   ```ts
   const EXTENDED_THINKING_LEVELS: ModelThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
   export function getSupportedThinkingLevels<TApi extends Api>(model: Model<TApi>): ModelThinkingLevel[] {
   	if (!model.reasoning) return ["off"];
   	return EXTENDED_THINKING_LEVELS.filter((level) => {
   		const mapped = model.thinkingLevelMap?.[level];
   		if (mapped === null) return false;
   		if (level === "xhigh" || level === "max") return mapped !== undefined;
   		return true;
   	});
   }
   ```

2. **Nearest-level clamp.** `clampThinkingLevel(model, level)` walks forward then backward from the requested index to the nearest supported level, defaulting to the first available (`src/models.ts:913-946`).
3. **Effort/budget down-convert.** For adapters that cap at `high`, `clampReasoning()` maps `xhigh|max → high` (`src/api/simple-options.ts:64-66`), and `thinkingBudgetForLevel()` picks a token budget from `DEFAULT_THINKING_BUDGETS` (overridable per call via `thinkingBudgets`) (`simple-options.ts:57-72`). Budgets are also clamped so the answer keeps room: `MIN_ANSWER_TOKENS = 1024`, `clampThinkingBudgetToAnswerRoom()` (`simple-options.ts:55, 74-77`) and `adjustMaxTokensForThinking()` grows `maxTokens` to fit the budget when possible (`simple-options.ts:79-95`).

### 2.3 Per-provider mapping of the `reasoning` knob

The consistent pattern: `streamSimple` translates `options.reasoning` into that API's **full native option type** and delegates to the raw `stream`; `buildParams` then consults `model.reasoning`, `model.thinkingLevelMap` (map string values are sent verbatim; `undefined` falls back to the requested level) and compat flags.

| API / provider | What pi-ai sends | Effort-level translation | Source |
|---|---|---|---|
| **openai-responses** (OpenAI, Azure, xAI, Copilot) | `reasoning: { effort, summary }` + `include: ["reasoning.encrypted_content"]` | `effort = thinkingLevelMap[level] ?? level`; off → `effort: thinkingLevelMap.off ?? "none"` (omitted entirely for Copilot or when `off: null`); per-model `xhigh`/`max` only via map (GPT-5.2+ → `xhigh`, GPT-5.6 → `max`) | `src/api/openai-responses.ts:94-99, 209-215, 323-339`; `scripts/generate-models.ts:888-896` |
| **openai-codex-responses** | Same Responses `reasoning` block over Codex transport | Same clamp + `reasoningEffort` pass-through | `src/api/openai-codex-responses.ts:505-512` |
| **openai-completions** (format `"openai"`, default) | `reasoning_effort: <mapped level>` | `thinkingLevelMap[level] ?? level`; when off and `off` maps to a string (e.g. `"none"`), sends that; skipped if `supportsReasoningEffort: false` | `src/api/openai-completions.ts:945-956` |
| **openai-completions** `thinkingFormat: "openrouter"` | `reasoning: { effort }` | `thinkingLevelMap[level] ?? level`; off → `effort: thinkingLevelMap.off ?? "none"`; map generated from OpenRouter `supported_efforts` + `mandatory` (`off: null` for reasoning-mandatory models) | `src/api/openai-completions.ts:915-924`; `scripts/openrouter-reasoning-options.ts:12-23` |
| **openai-completions** `thinkingFormat: "deepseek"` (DeepSeek) | `thinking: { type: "enabled" }` / `{ type: "disabled" }` **plus** `reasoning_effort` when `supportsReasoningEffort` | on/off toggle from `options.reasoningEffort` presence; level mapped via `thinkingLevelMap` (DeepSeek `xhigh` → `max`) | `src/api/openai-completions.ts:905-914`; CHANGELOG `#3944` |
| **openai-completions** `thinkingFormat: "zai"` (z.ai / GLM) | `thinking: { type: "enabled", clear_thinking: false }` / `{ type: "disabled" }` (+ `reasoning_effort` when supported) | toggle + map, GLM-5.3 exposes `low/high/max` | `src/api/openai-completions.ts:857-869`; CHANGELOG `#8336` |
| **openai-completions** `thinkingFormat: "together"` | `reasoning: { enabled: bool }` (+ `reasoning_effort` when supported) | boolean toggle derived from level presence | `src/api/openai-completions.ts:930-938` |
| **openai-completions** `thinkingFormat: "baseten"` | `chat_template_args` with `$var` placeholders (`thinking.enabled` / `thinking.effort` / `thinking.budget`) (+ `reasoning_effort`) | placeholders resolved from the same level/budget machinery | `src/api/openai-completions.ts:888-904`; `buildChatTemplateValues` 1010-1049 |
| **openai-completions** `thinkingFormat: "qwen"` / `"qwen-chat-template"` | top-level `enable_thinking: bool` or `chat_template_kwargs: { enable_thinking, preserve_thinking: true }` | boolean only (`!!options?.reasoningEffort`) | `src/api/openai-completions.ts:870-882` |
| **openai-completions** `thinkingFormat: "chat-template"` / `"string-thinking"` / `"ant-ling"` | `chat_template_kwargs` per config; `thinking: "<string>"`; `reasoning: { effort }` only when mapped non-null | `$var` substitution or raw mapped string | `src/api/openai-completions.ts:883-887, 925-944` |
| **anthropic-messages** (Claude) | Modern adaptive models: `thinking: { type: "adaptive", display }` + `output_config: { effort }`; older models: `thinking: { type: "enabled", budget_tokens, display }`; off: `thinking: { type: "disabled" }` (skipped when `off: null`) | `AnthropicEffort = "low" \| "medium" \| "high" \| "xhigh" \| "max"`; `mapThinkingLevelToEffort`: `minimal|low → "low"`, `medium → "medium"`, `high → "high"`; `xhigh`/`max` only via `thinkingLevelMap` (native on Opus 4.7/4.8, Sonnet 5, Fable 5; `max` on all adaptive models). Budget mode: `DEFAULT_THINKING_BUDGETS`, xhigh/max clamped to high's 16384 | `src/api/anthropic-messages.ts:167, 805-822, 831-870, 1063-1092`; `src/api/bedrock-converse-stream.ts:1240-1252` |
| **bedrock-converse-stream** | Anthropic models via `additionalModelRequestFields` (same adaptive/budget shapes); non-Anthropic reasoning via `reasoning: ThinkingLevel` passthrough | same effort map for Claude; budget table with explicit `xhigh/max → 16384` comment "Budget-based Claude clamps extended levels to high" | `src/api/bedrock-converse-stream.ts:74-75, 520-560, 1218-1262` |
| **google-generative-ai / google-vertex** (Gemini) | Gemini 3 / Gemma 4: `thinkingConfig.thinkingLevel: "MINIMAL"\|"LOW"\|"MEDIUM"\|"HIGH"` + `includeThoughts: true`; Gemini 2.5: `thinkingConfig.thinkingBudget: <tokens>` (per-model tables, e.g. 2.5-pro `high: 32768`; default `-1` dynamic) | `resolveGoogleThinkingLevel` lowers the mapped string; Gemini 3 Pro collapses `minimal|low → LOW`, `medium|high → HIGH`; off → `getDisabledThinkingConfig` (see 1.3) | `src/api/google-generative-ai.ts:307-335, 390-400, 428-483, 485-530`; `src/api/google-shared.ts:26-50` |
| **mistral-conversations** | `promptMode: "reasoning"` (some models) and/or `reasoningEffort` | `mapReasoningEffort(model, level)` gated by per-model capability helpers | `src/api/mistral-conversations.ts:193-206` |
| **pi-messages** (pi's own protocol) | `reasoning: options?.reasoning` passed through natively | no translation needed | `src/api/pi-messages.ts:371, 429` |

**What happens for providers/models that don't support it:** three layered answers — (a) `Model.reasoning: false` → the whole option is ignored (`README.md:788`; `bedrock-converse-stream.ts:1220` guards `!options.reasoning || !model.reasoning`); (b) a level the model lacks → clamped to the nearest supported level (`models.ts:913-946`); (c) a *server* that can't parse a parameter → compat flags suppress it (`supportsReasoningEffort: false` skips `reasoning_effort` entirely, `openai-completions.ts:945-956`; "If the server also does not support `reasoning_effort`, set `compat.supportsReasoningEffort` to `false` too. This commonly applies to Ollama, vLLM, SGLang" — `README.md:1106`). Failures also fed back into metadata: e.g. Copilot `gpt-5-mini` 400s on `reasoning: { effort: "none" }` led to omitting the field (CHANGELOG `#2567`, `packages/ai/CHANGELOG.md:1120`), and OpenRouter reasoning-mandatory models get `off: null` so pi never sends `effort: "none"` (`#8454`/`#8614`).

---

## 3. Response-side replay (multi-turn continuity)

Replay is where `thinkingSignature` earns its keep. A shared pre-pass, `transformMessages()` (`src/api/transform-messages.ts:76-145`), decides fidelity by model identity:

- **Same model** (`provider === model.provider && api === model.api && model === model.id`): thinking blocks and their signatures are kept verbatim, including empty-text-with-signature blocks (OpenAI encrypted reasoning).
- **Different model**: `redacted` blocks are dropped entirely ("Redacted thinking is opaque encrypted content, only valid for the same model. Drop it for cross-model to avoid API errors"), non-redacted thinking is downgraded to a plain `text` block (no `<thinking>` tags — earlier tagging caused models to mimic the pattern, CHANGELOG `#561`), and Google `thoughtSignature` is stripped from tool calls.

Then each adapter re-encodes:

- **Anthropic** (`src/api/anthropic-messages.ts:1217-1256`): `redacted: true` → `{ type: "redacted_thinking", data: thinkingSignature }`; signed thinking → `{ type: "thinking", thinking, signature }`; **missing/empty signature (e.g. aborted stream) → converted to a plain `text` block**, unless `compat.allowEmptySignature` is set for Anthropic-compatible providers that accept `signature: ""` (flag documented at `types.ts:691-692`).
- **OpenAI Responses** (`src/api/openai-responses-shared.ts:221-224`): `JSON.parse(block.thinkingSignature)` → pushed back as the raw `ResponseReasoningItem` input item (with `encrypted_content`), because the request sets `store: false` and `include: ["reasoning.encrypted_content"]` (`openai-responses.ts:332-338`). Text blocks are replayed with their original message ids recovered from `textSignature` (capped at 64 chars, hashed if longer) (`openai-responses-shared.ts:226-247`). Tool-call pairing ids are dropped when replaying to a *different* model to avoid OpenAI's fc_xxx↔rs_xxx pairing validation (`openai-responses-shared.ts:254-262`).
- **OpenAI-completions** (`src/api/openai-completions.ts:1269-1361`): `requiresThinkingAsText` servers get thinking flattened to text; standard servers get the reasoning echoed back under the recorded field name (`reasoning_content`/`reasoning`/`reasoning_text`); OpenRouter `reasoning_details` are resent verbatim (`preservedReasoningDetails`); DeepSeek-shaped servers additionally require an **empty** `reasoning_content: ""` on every replayed assistant message when reasoning is on (`requiresReasoningContentOnAssistantMessages`, detected as default for DeepSeek URLs at `openai-completions.ts:1633-1634`).
- **Google** (`src/api/google-shared.ts:168-204`): same-model thinking replayed as `{ thought: true, text, thoughtSignature }` parts; cross-model → plain text; text blocks that carry a `thoughtSignature` (Gemini attaches signatures to non-thought parts too) are preserved as-is even when empty.

---

## 4. Escape hatches (provider-specific without leaking into unified types)

1. **Two-tier API surface.** `streamSimple`/`completeSimple` expose only the unified options; `stream`/`complete` accept each API's *full* native option type, reached type-safely via `hasApi(model, "anthropic-messages")` narrowing and `ApiOptionsMap` (`types.ts:243-262, 307-336`; `README.md:828-858`). Provider-specific shapes like `AnthropicOptions.thinkingEnabled/thinkingBudgetTokens/effort`, `OpenAIResponsesOptions.reasoningEffort/reasoningSummary`, `GoogleOptions.thinking: { enabled, budgetTokens, level }` never appear in `SimpleStreamOptions`.
2. **`samplingParams` pass-through.** "Arbitrary sampling parameters merged into the request body as-is, **after** the named request fields, so keys here override them" — only applied by OpenAI-compatible adapters (`types.ts:186-193`; merge point `openai-responses.ts:344-346`, `openai-completions.ts:948-950`). This is the sanctioned way to send keys pi does not model.
3. **`onPayload` interception.** "Optional callback for inspecting or replacing provider payloads before sending. Return undefined to keep the payload unchanged" (`types.ts:141-145`).
4. **Per-model `compat` flags** (only for the four OpenAI/Anthropic/Bedrock API families; typed as a conditional on `Model.api`, `types.ts:840-849`) — 20+ switches incl. `thinkingFormat`, `supportsReasoningEffort`, `forceAdaptiveThinking`, `allowEmptySignature`, `requiresThinkingAsText`, `thinkingTokenBudgetField`.
5. **Headers/env.** `ProviderHeaders` with null-to-suppress semantics, `ProviderEnv` overrides, per-request `headers` merged last (`types.ts:112-177`).
6. **Custom API registration.** `KnownApi | (string & {})` plus `registerApiProvider` lets third parties add whole transports; unknown APIs fall back to `StreamOptions & Record<string, unknown>` (`types.ts:17-29, 260-262`).

Notably, pi-ai has **no** generic `providerOptions: { provider: {...} }` bag (unlike the Vercel AI SDK); specificity is achieved by the typed per-API option layer + `samplingParams`/`onPayload` instead.

---

## 5. Version history & rationale

The CHANGELOG (2025-11-26 → 2026-08-28) documents the evolution of the control scheme:

| When | Version (approx.) | Change |
|---|---|---|
| 2025-12-06 | 0.13.0 | `xhigh` added to a `ReasoningEffort` type for OpenAI codex-max; for Anthropic/Google "xhigh is automatically mapped to high" (`CHANGELOG.md:1997`) — effort values hardcoded per adapter, no model metadata |
| 2026-01-08 | 0.38.0 | `thinkingBudgets` option added to `SimpleStreamOptions` for token-budget providers (`#529`, `CHANGELOG.md:1743`) |
| 2026-01-13 | 0.45.4 | First `thinkingFormat` compat flag: z.ai needs `thinking: { type: "enabled" }` instead of `reasoning_effort` (`#688`, `CHANGELOG.md:1665`) |
| 2026-02 | ~0.30-0.34 | `thinkingFormat: "deepseek"` + `reasoningEffortMap` + empty `reasoning_content` replay compat for DeepSeek V4 session 400s (`#3636`, `CHANGELOG.md:927`) |
| 2026-04 | ~0.52-0.56 | **Breaking:** `OpenAICompletionsCompat.reasoningEffortMap` replaced by top-level `Model.thinkingLevelMap`; `getSupportedThinkingLevels()`/`clampThinkingLevel()` added; `supportsXhigh()` removed (`#3208`, `CHANGELOG.md:825-838`) |
| 2026-05 | ~0.60 | OpenAI Responses sends `reasoning.effort: "none"` when thinking is off; Mercury 2 tool-call fix via `off: null` metadata (`CHANGELOG.md:780-781`) |
| 2026-06 | ~0.68 | Opt-in **`max`** thinking level added; native `xhigh`/`max` for GPT-5.6; Anthropic adaptive-thinking effort metadata per Anthropic docs (`CHANGELOG.md:378`) |
| 2026-06 | ~0.70 | `Usage.reasoning` token count added across Anthropic/OpenAI/Google (`#6057`, `CHANGELOG.md:426`) |
| 2026-07 | ~0.78 | Generated catalogs expose **only provider-verified** effort levels from models.dev (`#6928`, `CHANGELOG.md:243`); Copilot `minimal` overrides table appears (`generate-models.ts:472-476`) |
| 2026-08 | 0.84.3/0.84.4 | OpenRouter controls derived from `supported_efforts`/`mandatory` metadata so reasoning-mandatory models never receive `effort: "none"` (`#8454`/`#8614`); `GoogleThinkingLevel` renamed to `GoogleApiThinkingLevel` + `ResolvedGoogleThinkingLevel` (`CHANGELOG.md:24, 18`) |

**Why it changed (as inferable from the changelog — no design doc states it):** the original scheme hardcode-mapped a single OpenAI-derived effort enum inside each adapter, which broke as (a) providers multiplied incompatible request shapes (z.ai, DeepSeek, Qwen, Baseten → `thinkingFormat`), (b) the same *provider value* differed per *model* (DeepSeek `xhigh → max`, GPT-5.5-pro lacking low, Gemini 3 Pro collapsing levels → per-model `thinkingLevelMap`), and (c) "supported?" booleans like `supportsXhigh()` couldn't express holes (models exposing `high`+`max` but not `xhigh` → tristate map with `null`).

**Naming:** nothing published states why the response block is `thinking` while the capability flag and simple-API knob are `reasoning`. The README's section title "Thinking/Reasoning" treats them as synonyms (`README.md:786`), and the agent layer's own type comment says "Thinking/reasoning level for models that support it" (`packages/agent/src/types.ts:295-301`). The de-facto convention: **`thinking` = the content/what the model does** (blocks, events, `thinkingEnabled`, `thinkingBudgets`, `thinkingLevelMap`), **`reasoning` = the capability/effort knob** (`Model.reasoning`, `SimpleStreamOptions.reasoning`, `Usage.reasoning`, `reasoningEffort` mirroring OpenAI's own name). The RFC corpus tagged `pi` (9 RFCs, https://rfc.earendil.com/keyword/pi/) contains no unified-API/reasoning-vocabulary RFC; the closest, RFC 0054 "Responses Lite Investigation", concerns Responses transport mechanics and only mentions reasoning via "Sets reasoning.context to all_turns" (https://rfc.earendil.com/0054/). pi.dev docs mirror the in-repo coding-agent docs; the "Custom Models" page documents `thinkingLevelMap` with the same tristate semantics and the `reasoningEffortMap` migration note (https://pi.dev/docs/latest/models).

---

## 6. Lessons for DlightRAG

Grounding first — DlightRAG today (`/Users/hanlianlyu/Github/DlightRAG`):

- Request side is vocabulary-split and off-only: `ToolModel.stream_text(..., thinking: Literal["off"] | None = None)` (`src/dlightrag/engine/ai/tool_model.py:117`), merged with a provider hook `CompletionProvider.thinking_off_kwargs()` (`src/dlightrag/engine/ai/providers/base.py:195-203`) whose implementations are Anthropic `{"thinking": {"type": "disabled"}}` (`providers/anthropic_native.py:186-189`), Gemini `{"thinking_config": None}` (`providers/gemini_native.py:176-179`), and OpenAI-compatible `{"reasoning": {"enabled": False}}` (`providers/openai_compatible.py:142-145`).
- Response side is a flat `AssistantTurn.reasoning: str` (`src/dlightrag/engine/ai/messages.py:42`), with provider-specific extras stored raw: Anthropic `provider_state.thinking_blocks` replay (`providers/anthropic_native.py:90-99`), Gemini `thought`-part splitting (`providers/gemini_native.py:319-339`), OpenAI-compat `reasoning_content`/`reasoning_details` extraction (`providers/openai_compatible.py:82, 119, 153-156`).
- Capability + pass-through: `ModelProfile.supports_reasoning: bool` (`src/dlightrag/engine/ai/capacity.py:49`) and raw `model_kwargs`/`agentic_model_kwargs` dicts (`src/dlightrag/engine/ai/settings.py:77-78`), which `config.yaml` uses to express `thinking: {type: disabled}` (DeepSeek roles), `reasoning: {enabled: false}` (query/vlm roles), and `reasoning: {enabled: true}` (default chat via OpenRouter) — three shapes against the same `provider: openai` adapter (`config.yaml:8-57`).

Lessons, each grounded in a cited pi-ai mechanism:

1. **Adopt one ordered effort scale; keep "off" out of the wire vocabulary.** pi's `ThinkingLevel = "minimal"…"max"` with *omitted = off* (`types.ts:83, 317`) means callers never branch on "is off represented as `undefined`, `None`, or a provider string". DlightRAG's `thinking: Literal["off"] | None` (`tool_model.py:117`) already has the two-state shape but conflates the *user intent* ("off") with the *API value*; pi puts `"off"` only in agent state and metadata (`agent/src/types.ts:301`, `types.ts:84`) and erases it at the boundary (`agent.ts:450`). DlightRAG could keep the agent-level `"off"` but stop threading it into provider calls as a literal.
2. **Move per-model provider values out of adapters and out of config.yaml into a capability map.** DlightRAG's `config.yaml` hand-writes three different off/on shapes (`thinking: {type: disabled}` vs `reasoning: {enabled: false}`) for the *same* OpenAI-compat adapter, which is exactly the pre-`thinkingLevelMap` failure mode pi fixed in `#3208` (`CHANGELOG.md:825-833`). A per-model `thinkingLevelMap`-style entry (`off/…/max → provider string | null`, `types.ts:85`) would let config say `thinking: "off"|"low"|…` once and let the adapter produce `thinking: {type: "disabled"}` for DeepSeek, `reasoning: {enabled: false}` for GLM-via-OpenRouter, or `thinking_effort`-style values as needed. The tristate map also encodes "cannot disable" (`off: null`) and "cannot go below X", which DlightRAG's boolean `supports_reasoning` (`capacity.py:49`) cannot express — pi's clamp-then-degrade (`models.ts:913-946`) is what makes one call site safe across models.
3. **Name the response block "thinking" and the knob "reasoning" — but make the response structured, not a flat string.** pi's `ThinkingContent` carries `thinkingSignature` + `redacted` (`types.ts:356-364`) and still exposes convenience text. DlightRAG's `AssistantTurn.reasoning: str` (`messages.py:42`) loses three things pi keeps: (a) the **replay payload** — Anthropic `thinking_blocks` survive only in the ad-hoc `provider_state` dict (`anthropic_native.py:90-99`) instead of a typed field; (b) **redaction** — a redacted Anthropic block currently has no first-class representation; (c) **multi-block ordering** relative to text/tool calls. A small dataclass (`text`, `signature`, `redacted`) would subsume `provider_state` and make Anthropic replay lossless by construction, mirroring pi's single `thinkingSignature` envelope.
4. **Gate reasoning replay by model identity, and degrade to text cross-model.** pi's `transformMessages` converts cross-model thinking to plain text and drops redacted payloads (`transform-messages.ts:101-116`) after early bugs (models mimicking `<thinking>` tags, `#561`). DlightRAG already refuses to re-inject `reasoning_content` into history (`openai_compatible.py:131-135`) but Anthropic replay is unconditional on the stored blocks; an `is_same_model` check would prevent replaying Anthropic signatures into a different Claude model.
5. **Make the off-switch provider-shaped but centrally owned.** DlightRAG's `thinking_off_kwargs()` hook (`base.py:195-203`) is a good seam, and pi validates the approach — it also needed per-provider disable shapes (`thinking: {type: "disabled"}`, `reasoning: {effort: "none"}`, `enable_thinking: false`, `thinkingBudget: 0`, lowest-level-without-`includeThoughts` for Gemini 3, `google-generative-ai.ts:428-447`). But pi adds two refinements worth copying: (a) the **"cannot disable → cheapest level, hide the output"** fallback (`getDisabledThinkingConfig`) so "off" never silently still burns tokens *and* never errors; (b) **metadata-driven suppression** — when a server 400s on the off-value (Copilot `effort: "none"`, `#2567`; OpenRouter mandatory reasoning, `#8454`), the fix was recorded in the catalog (`off: null`) rather than in call-site conditionals.
6. **When adding an effort knob later, add budgets as a parallel concern, not a replacement.** pi keeps `ThinkingLevel` (portable) and `ThinkingBudgets`/`thinkingTokenBudgetField` (token-based servers: vLLM `thinking_token_budget`, Qwen/SGLang `thinking_budget`, llama.cpp `thinking_budget_tokens`, `types.ts:96-105`) separate, with a shared answer-room reserve of 1024 tokens (`simple-options.ts:55-77`) so reasoning cannot eat the whole `max_tokens`. DlightRAG's compaction/final-answer path — which forces a profile output cap precisely so "a reasoning model cannot burn the output cap on hidden reasoning" (`tool_model.py:121-127`) — is the same problem; an effort knob plus a budget-reserve rule generalizes the current hard off-switch.
7. **Verify level support against a source of truth and regenerate.** pi generates its catalog from models.dev `reasoning_options` and OpenRouter `supported_efforts` and only ships provider-verified levels (`generate-models.ts:1217-1232`, `models-dev-reasoning-options.ts:18-30`; `#6928`), then re-derives OpenRouter quirks (`openrouter-reasoning-options.ts:12-23`). DlightRAG's `model_catalog.json`/`capacity.py` could carry a `thinking_levels` field maintained the same way, instead of config authors guessing which DeepSeek/GLM model accepts which shape.
8. **Keep the escape hatch narrow and typed.** DlightRAG's raw `model_kwargs` pass-through (`settings.py:77-78`) is pi's `samplingParams` equivalent (`types.ts:186-193`) and is fine — but pi contains drift by *merging pass-through last so explicit keys override named fields* and documenting that rule. If DlightRAG later adds named thinking kwargs, decide and document precedence now (`model_kwargs` wins) to avoid the three-shapes-in-config situation recurring at the code level.

---

## Appendix: unverifiable / not found

- **No published naming rationale** for "thinking" (content) vs "reasoning" (knob) exists in the repo docs, README, CHANGELOG, pi.dev, or the 9 RFCs tagged `pi` (checked 2026-08-29). Section 5's convention summary is inferred from usage, not quoted from a source.
- The **generated model-catalog JSON** (`src/providers/data/*.json`) is produced at build time by `scripts/generate-models.ts` and is not committed, so concrete shipped `thinkingLevelMap` values per model were verified from the generator's hardcoded overrides and mapping functions, not from a built artifact.
- The monorepo shows **~99k stars / ~5,800 commits** per the task brief; the shallow clone contains 1 commit so those figures were not independently verified.
- `rfc.earendil.com` has no RFC describing the pi-ai option-surface design; rationale above relies on CHANGELOG entries and code comments.
