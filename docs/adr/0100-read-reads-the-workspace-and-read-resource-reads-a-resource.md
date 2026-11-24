# ADR 0100 — `read` reads the workspace, and `read_resource` reads a Resource

**Status**: Accepted on 2026-11-24.

**摘要**：`read` 原来一个工具管三种目标（工作区路径、url、resource_id），参数之间靠 7 条条件规则约束：offset/limit 只对路径有效，focus 只对资源有效，cursor 在不同目标下语义不同，http 只对 url 有效。这些规则只能靠校验器报错告诉模型。
现拆成两个工具：`read` 只读工作区路径（path、offset、limit、cursor）；`read_resource` 读一个 Resource，可以给 resource_id，也可以给公开 url（url 读取时会注册成 Resource；参数是 focus、cursor、http，有 Agent Browser 时还有 rendered）。
每个 schema 里不再有依赖目标类型的参数。子代理的工具白名单里写了 `read`，就同时保留 `read_resource`，能力和拆分前一致。证据账本里的行标签仍然用 `read`，所以早先轮次记下的证据和新读到的同一段文字身份相同。

## Context

- One `read` took exactly one of `path`, `url` or `resource_id`, and seven validator rules said which other arguments each target allowed. A model learned them only from refusals.
- The live store (2026-11-23) held 20 `invalid_arguments` results of `read`: 17 were a path's `limit`/`offset` passed to a url or resource_id read, one a `focus` on a path. Seven more calls failed on a cursor of one representation used on another, and two read a Skill's reference document with `read(path=...)`.
- A Resource is paged by token windows and opaque cursors, a file by lines; the two cannot share one paging argument without a second paging model in the registry.

## Decision

1. **Two Tools.** `read` (contract version 5) takes `path`, `offset`, `limit`, `cursor`, and is declared wherever the Run has a workspace. `read_resource` (contract version 1) takes exactly one of `resource_id` or `url`, plus `focus`, `cursor` and `http` (url only), and `rendered` where the Run has an Agent Browser (ADR 0032); it is declared wherever the Run has a resource reader. Each description names the other Tool for the other kind of target.
2. **Every continuation names the Tool that continues it.** A spill (`[spill] … read_resource(resource_id=…)`), a published product's handle, a visual page's next cursor, a download's resource id and a stale-cursor refusal all name `read_resource`.
3. **A child's tool list keeps what `read` meant.** A `tools` list that names `read` keeps `read_resource` as well, so the spill any Tool's long output continues through stays readable, and a list written before the split grants exactly what it granted. A list without `read` gains nothing.
4. **The evidence ledger keeps its label.** A `read_resource` row is labelled `read` in its chunk id and evidence key, as before the split, so the same text read in a later round is one row.

## Consequences

- A Run accepted before the deploy pinned the old `read` contract; a url or resource_id `read` of it after the deploy is refused as `contract_changed`, and its next round is offered the two Tools.
- The reader's step card shows both Tools as "Reading a document".
