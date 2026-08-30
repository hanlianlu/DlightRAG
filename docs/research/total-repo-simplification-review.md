# DlightRAG 全仓简化与冗余审查（人工复核版）

**审查基线：** `92b9d4fad718cf5dca28917e444fc2d5bea98454`
**范围：** `src/`、`packages/memory/`、HTTP/Web/MCP、PostgreSQL、前端、脚本、测试与架构约束。
**结论性质：** 这是对多路子代理报告逐条回查源码后的最终判断，不是子代理报告的直接汇总。

## 执行结论

仓库没有 P0 阻断问题，也不存在值得重写的架构性冗余。当前架构整体上是深模块：复杂度主要集中在 Answer Run、Agent Session、Workspace RAG 与 PostgreSQL 生命周期所有者内部，由较窄的接口隐藏；按文件大小拆分只会扩散状态所有权。

真正值得处理的范围较小：

- **3 个 P1 聚类：** HTTP Artifact 交付、HTTP 模型目录路由、源文件下载响应。
- **约 8 个 P2 小型去重/删除项：** Agent disposition 映射、Memory receipt/recency、ToolCall 投影、metadata path 包装层、少量死代码与陈旧注释。
- **1 个独立的 P2 可选耐久性加固：** metadata field-stats trigger 启动校验；它不是当前 trigger function 的缺陷，也不是简化项。
- 保守估计可净删约 **120–200 行产品代码**，另可删除或收缩若干只维护实现细节的测试；不是子代理声称的 350–500 行。

## 实施结果

上述建议已在基线之上完成实现：**3 个 P1、8 个 P2 清理项，以及 `/web/design-system` 一致性修复全部落地**。独立的 metadata trigger verifier 仍按结论保留为可选耐久性加固，未混入本次简化。

实际改动结果：

- 产品代码净减少 **161 行**（含 4 个新共享模块）；
- 测试代码净减少 **22 行**，同时新增 Web unavailable-artifact、Fast refresh corruption 与 design-system route 回归覆盖；
- 总代码净减少 **183 行**，落在人工复核的 120–200 行区间内；
- REST/Web/MCP 的认证与 owner scoping 未合并，26 个 import-linter 边界全部保留。

实施后完整验证：

- `uv run ruff check src packages/memory scripts tests` → 通过；
- `uv run ruff format --check src packages/memory scripts tests` → 通过；
- `uv run pyright` → **0 errors**（16 个既有 lazy-export warnings）；
- `uv run lint-imports` → **26 contracts kept, 0 broken**；
- `uv run pytest tests/unit -q --tb=short` → **3597 passed**；
- `uv run pytest tests/integration -q --tb=short` → **267 passed**；
- `tests/e2e/test_pg18_lightrag_smoke.py` 在未启用环境开关时 → **3 skipped**；
- 全量浏览器 E2E → **64 passed, 3 skipped, 16 failed**；失败均表现为既有 UI layout/visibility 几何问题。代表性 `artifact_canvas` failure 已在未改动的 `92b9d4fa` worktree 原样复现，因此不归因于本次 Python/route 简化；未把既有前端问题混入本次提交。

独立双轴复核未发现 P0/P1。复核提出的三个可操作 P2 已处理：收窄 disposition map key 类型、把包内 `operation_receipt` 纳入模块 `__all__`、恢复 metadata seam 契约说明及原日志 tag。Fast refresh 的对象身份检查已由既有 Agent runtime 使用，本次全量 unit/integration 也通过，未发现 adapter 重解码回归。

## 研究阶段验证

在当前工作树实际执行：

- `uv run python --version` → `Python 3.14.7`
- `uv run python -m compileall -q src/dlightrag packages/memory scripts tests` → 通过
- `uv run ruff check src packages/memory scripts tests --output-format concise` → 通过
- `uv run lint-imports` → **26 contracts kept, 0 broken**
- 目标单元测试 → **198 passed**

目标测试覆盖 REST/Web Artifact、Answer orchestrator、Agent runtime、Memory store/recall 与 metadata path。

---

## P1：优先实施

### P1-1：统一 REST/Web Artifact 交付，并修复 Web presentation guard 漂移

**证据**

- REST：`src/dlightrag/adapters/http/rest/routes/answer_runs.py`
  - `_published_artifact`
  - `_artifact_response_headers`
  - `read_answer_artifact` 中的 Range 解析
  - `read_answer_artifact_presentation`
- Web：`src/dlightrag/adapters/http/browser/routes/chat.py`
  - `_artifact_descriptor`
  - `_artifact_range`
  - `answer_artifact_data` 中的 header/disposition/CSP 组装
  - `answer_artifact_presentation`

两边重复实现相同的：单 Range 解析、suffix/open-ended Range、416 `Content-Range`、安全 inline 类型、Content-Disposition、`nosniff`、`private, no-store`、SVG inert CSP、artifact descriptor 查找。

已经发生行为漂移：REST presentation 要求 `descriptor.status == "available"`，Web 只检查 MIME 为 Markdown。Web 因而没有遵守 descriptor 的可用性状态。

**建议**

建立一个 HTTP 内部模块（例如 `adapters/http/artifact_delivery.py`），只拥有：

1. descriptor 查找；
2. Range 解析结果；
3. Artifact 数据响应 headers；
4. presentation 可用性谓词。

REST/Web 继续各自拥有认证、owner scoping、URL 前缀和最终 presentation 投影。

**删除测试**

- 两个路由文件中不再保留第二份 Range/header/CSP 算法。
- 新增 parity 测试：`status="unavailable"` 的 Markdown descriptor 在两个 presentation 路由均返回 404。
- 保持现有 200/206/416、suffix Range、空文件、SVG CSP 测试不变。

**风险：** 低；但属于鉴权后的字节交付路径，必须保持 404 indistinguishability 与全部安全 headers。

### P1-2：抽取 REST/Web 模型目录的共享 HTTP 操作

**证据**

- `src/dlightrag/adapters/http/rest/routes/model_catalogue.py`
- `src/dlightrag/adapters/http/browser/routes/model_catalogue.py`

Browser 路由已从 REST 路由导入私有 `_etag`、`_if_match`、`_response` 和 DTO，同时又逐字重复 read/upsert/remove 的应用调用、ETag 写入及 403/404/412/422/503 映射。差异只在：

- REST 使用 Bearer dependency 与 `user.user_id`；
- Web 使用 session access gate 与 `_actor(request)`。

这是同一个 HTTP 协议的机械实现重复，不是必要的 transport projection 差异。

**建议**

建立共享 HTTP 模块 `adapters/http/model_catalogue.py`，拥有 DTO、ETag/If-Match 处理和 HTTP 错误映射。两个路由只保留各自应用调用，以免把 access/actor 差异藏进新的回调接口：

1. 各自的访问控制；
2. actor 提取；
3. 调用共享操作。

**删除测试**

- Browser 不再从 REST route 私有符号导入。
- upsert/remove 的异常映射只剩一份。
- REST 与 Web 的 ETag、If-Match、404/412/422/503 合同保持原样。

**实际净删除：** 13 行。第一版抽取曾增加 28 行，未通过 deletion test；改为通用 mutation/error projector 后才保留。主要收益是删除 Browser → REST 私有符号依赖与第二份异常映射，而非追求虚高行数。
**风险：** 低。

### P1-3：统一 REST/Web 源文件下载响应

**证据**

- REST：`src/dlightrag/adapters/http/rest/routes/files.py::_download_response` 与 `serve_file`
- Web：`src/dlightrag/adapters/http/browser/routes/files.py::_source_download_response` 与 `download_source`

两边逐字重复：

- `LocalDownloadTarget -> FileResponse`
- `RedirectDownloadTarget -> 302 RedirectResponse`
- 未知 target 的 `TypeError`
- `SourceDownloadInvalidError/NotFoundError/UnavailableError -> 400/404/503`

认证和 workspace 解析不同，必须保留；认证之后的下载准备与响应投影相同。

**建议**

在共享 HTTP 内部模块中提供一个深函数：接收 corpus application interface、workspace、document id，返回 `FileResponse | RedirectResponse` 并拥有上述错误映射。

**删除测试**

`"Unsupported source download target"` 只保留一个定义；REST Bearer 与 Web cookie/session 权限检查仍在各自路由中先执行。

**风险：** 低。

---

## P2：安全的小型去重与删除批次

### P2-1：Agent tool disposition → outcome 映射单一化

重复位置：

- `engine/agent/session/interpreter.py::next_action`
- `engine/agent/session/runtime.py::_close_cancel_position`

`unknown_tool`、`invalid_arguments`、`plan_denied`、`truncated_call`、`contract_changed` 的结果映射重复，取消路径额外处理 executable。

共享 owner 应放在 `interpreter.py` 或 `operation.py`，**不应照子代理建议放入 `effects.py`**：`operation.py` 已导入 `effects.py`，反向导入 `ToolCallDisposition` 会形成循环风险。Runtime 已经导入 interpreter，因此从 interpreter 复用映射最窄。

### P2-2：FastSessionHost 复用现有 snapshot refresh 校验

- 共享校验已经存在：`engine/agent/session/repository.py::validate_snapshot_refresh`
- `AgentSessionRuntime._refresh_snapshot` 已使用它。
- `FastSessionHost.snapshot` 手写了较弱的 commit/entry/lane 回退检查。

只需让 FastSessionHost 调用现有校验；**不建议**为两套 runtime 抽取新的 `SessionSnapshotCache` 类。完整抽取会引入新的内部接口，并且两套 driver 的事务错误类型和生命周期并不完全相同。

### P2-3：Memory 两个 adapter 共享 receipt builder 与 recency owner

重复位置：

- `packages/memory/src/dlightrag_memory/store.py::_receipt`
- `packages/memory/src/dlightrag_memory/_storage/pg.py::_operation_receipt`
- `recall.py::recall_recency`
- `store.py::_cursor_time`
- `_storage/pg.py::_cursor_time`

建议：

- 共享一个包内 helper `operation_receipt`；它不从 `dlightrag_memory` 包根导出，因此不升级为公共包接口。
- pagination 直接使用 `recall_recency`，删除两个 `_cursor_time`。
- receipt equality 与分页测试保持不变。

### P2-4：ToolCall model-message 投影单一化

重复位置：

- `engine/agent/session/fold.py::fold_tool_call`
- `engine/agent/tools/executor.py::_tool_call_message`

两者逐字生成相同的 `id/type/function/name/arguments/thought_signature` 结构。可把 provider-neutral 投影放在 `engine/ai/messages.py`，由两个 Agent 调用方复用；实施前用 `lint-imports` 检查依赖方向。

### P2-5：删除 shallow metadata path wrapper

- `engine/rag/retrieval/metadata_path.py::metadata_retrieve` 仅调用 `MetadataScopeStore.resolve_scope` 并记录日志。
- 唯一产品调用方是 `UnifiedRetriever._resolve_candidates`。

它确实是浅模块；删除后复杂度不会扩散到多个产品调用方。把日志与契约说明移入 `_resolve_candidates`，删除独立模块。

但子代理“只有一个 caller”的陈述不完整：`tests/unit/test_metadata_path.py` 和 `tests/e2e/test_pg18_lightrag_smoke.py` 也直接调用它。实施时应让 E2E 直接验证 store seam，并删除或并入专门测试，而不是假装这些调用不存在。

### P2-6：删除已确认的死代码/死参数

已通过源码及全仓引用检查确认：

- `engine/answer/research/runtime.py::_child_status`：零调用。
- `engine/answer/history.py::_fit_episodic_summary`：零调用。
- `engine/answer/orchestration/orchestrator.py::answer_stream(run=...)`：参数第一行即 `del run`，调用方不传入。
- `engine/answer/research/runtime.py::_usage_from_snapshot`：仅包装 `_usage_from_snapshot_entries`，一个调用方，可内联。
- `application/answer_runs/capabilities.py::validate_startup`：明确为无操作，删除方法、启动调用与测试 fake。

这些适合一个独立纯删除提交。

### P2-7：修正 promotion 的陈旧说明

以下注释仍声称 worker/counter 不存在，但实现已经存在：

- `adapters/postgres/corpus/promotion_jobs.py` 模块与类 docstring
- `adapters/postgres/corpus/workspaces.py` 相关 docstring

只删除过时句子，不改 promotion transaction 或 recount。

### P2-8：共享 Memory receipt projection（低价值）

- REST：`adapters/http/rest/routes/memory.py::_receipt_payload`
- Web：`adapters/http/browser/routes/memory.py::_receipt`
- MCP：`adapters/mcp/server.py::_memory_receipt`

人工复查确认是三份八字段投影，而非最初记录的两份。现由 `application/memory/projections.py::memory_receipt_payload` 单一拥有，三个 transport 只消费该投影。

---

## P2：独立可选加固，不属于简化

### Metadata field-stats trigger verifier

维持此前结论：**P2 optional hardening，不是当前缺陷，不应列为 P1。**

`dlightrag_sync_metadata_field_stats()` 没有已知错误；校验器只防护数据库之外的破坏，例如手工 DROP/DISABLE TRIGGER、不完整 restore 或未来迁移遗漏。若实施，应保持 metadata-local catalog query，不扩展通用 `TableRequirement`，不 hash `prosrc`，不增加 reconciler/config/telemetry。

当前优先级低于上述全仓真实重复。

---

## 另行修复的产品一致性问题

### `/web/design-system` 的生产入口与 Vite 入口不一致

- 生产路由：`adapters/http/browser/routes/chat.py::design_system_page` 返回 `index.html`。
- Vite dev/build：`frontend/vite.config.ts` 和 `frontend/design-system.html` 使用独立 specimen 入口。

这不是简化项，而是可验证的 dev/prod 漂移。现已让生产 route 返回已有的 `design-system.html`，并新增路由测试锁定独立 design-system asset，而不是 `<dl-app>` shell。

---

## 明确否决或降级的子代理建议

### 1. “Python-2 except 语法导致全仓无法导入”——错误

`except A, B:` 是 Python 3.14 的 **PEP 758** 新语法，本仓库明确要求 `>=3.14.7,<3.15`，Ruff target 为 `py314`。本轮实际验证：

- `compileall` 通过；
- Ruff 通过；
- 直接执行多异常捕获输出 `PEP758_OK`。

因此不存在 P0，也绝不能把这些语法机械改回括号形式作为“修复”。

### 2. 把 trigger verifier 升为 P1——否决

它保护 out-of-band schema damage，不证明 trigger function 自身有错，也不减少仓库复杂度。

### 3. 删除 `application/answer_runs/images.py`——否决

该模块被 `adapters/http/browser/attachment_models.py` 使用，是 HTTP → Application 的 import firewall。让 Browser 直接导入 `engine.ai.media` 会破坏已通过的 `HTTP and MCP import Application, not Engine` import-linter contract。

### 4. 把 `InMemoryMemorySettingsStore` 移到 tests 当作“删除”——否决

复杂度只会从产品树搬到测试树，删除测试 adapter 后仍需重新实现相同 epoch/deactivation 行为；不通过 deletion test。它是一个真实的本地替代 adapter。

### 5. “MCP list_answer_runs 可请求无限页”——错误

虽然 MCP 参数没有 `le=100`，`PGAnswerRunStore.list_runs` 在存储接口内执行 `max(1, min(limit, 100))`。最多 100 行，不存在无界查询；最多只是 schema/OpenAPI 风格一致性问题。

### 6. 合并三种 continuation/fork orchestration——否决

REST、Web、MCP 必须在各自当前认证上下文中重新解析 workspace 权限；`AnswerService.continuation_request` 又会重新读取 terminal parent 并构建 durable context。把授权搬入 Application 会污染 trust seam，新增 callback/DTO 的复杂度大于删除量。

### 7. “365 天 retention 是三份重复事实”——否决

- 独立 memory package 需要 standalone 默认值；
- root config 把 memory retention 显式耦合到 Answer Run retention；
- runtime fallback 是另一模块的安全默认。

相同数值不代表相同 owner。跨包共享常量反而破坏 memory package independence。

### 8. 删除 `RoutingAcceptance` 两列/`fallback()`——暂缓

`model_fingerprints` 与 `context_policy_revision` 当前确实没有 worker reader，但：

- 删除需要数据库迁移，净代码不一定减少；
- routing table 中保留 acceptance audit facts 可能是有意设计；
- `fallback()` 被大量 PG adapter 直接测试/bootstrapping 调用依赖；正常 `AnswerService` 生产路径已经传入真实 routing，并非 Web 生产路径总是 fallback。

在没有运维查询与外部 SQL 消费者证据前，不列入 P1。

### 9. 删除 pre-recall `pin_probe`——否决

源码明确要求在 Session/effect 边界前后分别校验 accepted pins。两个时点具有时间语义；除非测试证明错误时序完全不可观察，否则它是 earned complexity。

### 10. 抽取完整 `SessionSnapshotCache`——降级

两套 driver 相似，但事务错误、initial snapshot 与生命周期不同。先复用现有 `validate_snapshot_refresh` 即可；新增 cache 模块不通过当前 deletion test。

### 11. 合并 provider message shaping——暂缓

Anthropic、Gemini、OpenAI 对 data URI 和 native block 的接受规则不同。增加中间 neutral representation 很可能只搬运复杂度；没有可证明的净删除前不做。

### 12. 拆分大文件/大类——否决

`PGAnswerRunStore`、`WorkspaceRag`、`AnswerExecutor`、`AgentSessionRuntime` 是总状态和生命周期所有者。它们的实现大，但对调用者提供的接口相对窄；按行数拆分会损失 locality，而不会删除状态或策略。

---

## KEEP 清单

1. `_compose.py` 单一私有 composition seam 与 `Application` façade。
2. 26 个 import-linter contracts。
3. Agent Session fencing、replay、operation-state 与 Fast reservation/CAS 两套不同状态机。
4. Answer publication/citation correction、durable event 与 artifact blob ownership。
5. Promotion delete → attach → stats rebuild 的单事务顺序，以及 pure-reconcile 路径的无条件 recount 修复能力。
6. Metadata field-stats trigger/backfill/recount、128-key planner bound 与单次 metadata scope probe。
7. LightRAG contract guard、受控 private-API seam 与 gated patches。
8. REST/Web/MCP 各自的认证、错误 envelope、URL/descriptor/frame projection。
9. `dlightrag_memory` 独立包及 InMemory/Postgres 两个 adapter。
10. Python 3.14 / PEP 758 语法与当前 patch-level floor，除非有明确兼容性决策。

## 实施顺序（已完成）

1. **Artifact delivery + Web guard parity**。
2. **Model catalogue 共享 HTTP contract/error projector**。
3. **Source download 共享响应**。
4. **纯删除批次**：dead helpers、dead `run` 参数、no-op startup gate、陈旧 promotion 文档。
5. **小型策略去重**：Agent disposition、Memory receipt/recency、ToolCall 投影、Fast refresh validator、metadata path。
6. **Design-system dev/prod 修复**。

唯一未实施的是独立、可选且不属于简化的 metadata trigger verifier；是否增加应由后续耐久性需求单独决定。