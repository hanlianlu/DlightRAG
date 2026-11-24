# ADR 0102 — a Web provider's keys rotate inside its chain position

**Status**: Accepted on 2026-11-24.

**摘要**：Web 搜索与网页提取原来每个供应商（Exa、Tavily、DeepSeek）只能配一个 API key；第二条额度或第二个账号只能靠改 `api_key` 顶掉旧 key，第一条额度用尽后整条链就落到下一家。现在每个供应商携带一组可互换的 key：`api_keys` 是唯一的设置，没有单 key 别名，单 key 就是长度为 1 的池子；供应商在自己这一个链位上轮换所有 key（每次取最久未使用的一把），只有全部 key 都因凭证/额度/频率被拒之后，有序链才落到下一级供应商。轮换是通用能力：`engine.credential_rotation.CredentialPool` 只管"下一个用谁、谁说过 not now、什么时候值得探一次"，不认识任何供应商；`web_sources/rotation.RotatingWebSource` 用工厂把一个 provider 适配器按 key 实例化并套上池子。搜索与提取各自独立轮换；池子全部耗尽时整家供应商被冷却一分钟，模型侧契约不变。

## Context

- Each Web provider accepted exactly one credential (`answer.web_sources.<provider>.api_key`,
  a single-key setting).
  A deployment with a second Exa or Tavily account could only replace the first key, so the
  second account's quota was unused, and when the first account ran out the ordered chain fell
  through to the next provider even though another account of the same provider could have
  answered.
- The policy the chain already applies is the right one for a key too: a credential that
  answered `401`, `402`, or `429` will not answer the next call either, while a timeout,
  an unreachable host, or one malformed body is about the call. The ordered chain parked such
  a provider for a minute and probed it when everything was cooling; a pool of keys needs the
  same two ideas one level down.
- Exa and Tavily quota enforcement is live on this deployment (2026-11-24: the primary Exa key
  answered `402 payment_required` on its first live search while two other Exa keys answered
  `200`), so "all keys out" is an observed state, not a hypothetical.

## Decision

1. **A provider owns a pool of interchangeable keys.** `WebSourceProviderConfig.api_keys` is
   the only credential setting: a blank entry is unset, duplicates collapse, order is rotation
   order, and a single key is a pool of one. There is no `api_key` alias — this is a forward-only
   clean break, so a deployment moves its key into the list. The environment carries the pool as a
   comma-separated run; the keys are secrets and stay in `.env` (ADR 0006).
2. **The pool is provider-neutral and operation-scoped.** `engine.credential_rotation.CredentialPool`
   holds only availability policy: it leases credentials least-recently-used first, parks one that
   reported a credential-level failure for a minute, clears it on an answer or a call-level failure,
   and yields exactly one recovery probe when every key is cooling. It knows no HTTP status, no
   provider, and no secret beyond the one it holds. Search and Extract each own a pool, because one
   leg's credit says nothing about the other's.
3. **A wrapper keeps the chain's shape.** `answer.web_sources.rotation.RotatingWebSource` implements
   the existing Search and Extract contracts for one provider `name` and builds one adapter per key
   through the adapter's own constructor. The chain therefore still sees Exa, Tavily, and DeepSeek as
   single entries; rotation never changes acquisition metadata, provider identity, or the model-facing
   tool. A provider whose whole pool refuses on credentials, credit, or rate raises the one reason
   `exhausted`, which the chain treats as cooling; a pool that also saw a call-level failure keeps its
   mixed reasons and is not parked, because another key may answer next time.
4. **A single key is a pool of one.** It rotates over nothing, and its credential-level refusal
   reads as `exhausted`, which the chain parks for a minute exactly as it parks any out-of-reach
   provider. An `.env` that carries the removed `api_key` field fails startup with the named field
   rather than being silently ignored.
5. **An operator can tell the keys apart, and the model cannot.** Every failed attempt is one log
   record naming the provider, operation, reason, status, and the key's position (`exa#2`), never the
   secret. A rotation that another key answers is transparent to the model; a fallback to the next
   provider still states the provider and the normalized reason, as before.

## Consequences

- A second account multiplies a provider's quota and spreads a per-key rate limit; a deployment
  keeps one key as a one-item `api_keys` run, and no compatibility alias survives.
- The setup wizard asks for one comma-separated key run per provider and writes it as the pool,
  removing a stale pool when the provider is no longer chosen.
- `CredentialPool` is reusable by any other provider whose credentials are interchangeable and whose
  failures split the same two ways; it carries no Web-search vocabulary.
- Live validation on 2026-11-24 rotated around a `402` primary Exa key across two healthy Exa keys
  with no model-facing degradation, and spent one probe per burst after a pool was parked.
