# 研判：DlightRAG 要不要 background 模式？回调点算不算预留得好？

> 日期：2026-10-24 · 基于仓库 HEAD `58bd92f9`（工作区有未提交的 docs/scripts/tests 改动，未触碰）
> 对照物：DeepSeek Harness（DSH）v0.2.1-rc.3（本机 app.asar 解包）、pi coding agent v1.0.0（本机安装包）
> 方法：4 份取证底稿 + Lead 复核关键断言。底稿见文末附录 B；本文只给判断与建议。

---

## 0. 直接回答

### Q1：是否需要 background 模式？

**不加"模式"，补"能力"。**

- 若指 **run 级后台**（客户端走了、Run 继续跑，回来收结果）：**不需要新增，这个能力已经存在，而且比 DSH 的 job 更持久**。四个入口都是先落库再返回，断连只摘观察者，SSE 可续传，进程重启可 requeue/reclaim，只有显式 cancel 才停。再加一个"后台开关"等于给同一件事造第二套生命周期语义。
- 若指 **DSH 那种 run 内 job**（`run_in_background` → 返回 id → `job_output`/`job_kill`，完成后通知）：**这是真实的概念差距，但优先级在通知/可见性之后，且必须证据驱动**。DlightRAG 已有替代物：Child Session（Run 内异步、durable、可 status/wait/cancel）、Goal Round（Run 结束后自动续跑）、Corpus Mutation Run（重活独立跑）、Memory Pass（后台服务）。真正缺的只是"**便宜、不花 token、只包一条命令**"的 job 原语。
- **当前真正该补的是感知层，不是执行层**：Run 已经在后台跑，但 Web 没有 Run/任务列表、会话列表没有"运行中"标记、完成通知只在"同一 tab + 同一会话 + 页面失焦"时才触发。**用户不知道它跑完了——这是产品缺陷，不是执行模型缺陷。**
- 建议节奏：**P0 = 完成通知 + 运行可见性（不动 RunRuntime 状态机）；P1 = 先用 `tool_end.duration_ms` / OTel 做长命令占比评估，只有过阈值才做 run 内 job（lifetime=Run、三工具、两事件、不跨 Run、不新建 Run kind）；P2 = 通用 job 平台/cron/worker-only 进程——没有证据不做。** ADR-0007 已经写明"worker-only 模式要有证据再引入"。

### Q2：和 pi agent、DSH 相比，回调点算预留得好吗？

**位置留了，契约没交付；观察/控制第一梯队，扩展/拦截第三梯队。**

- **观察面：A−**。Run 事件是 PostgreSQL 里带连续序号、可重放、`Last-Event-ID` 可续传、终态恰好一次的持久事实，经 SSE、in-process 订阅、OTel 对外（`subscription.py:59-99`、`docs/interfaces.md:426-445`、`docs/observability.md:69-142`）。这已经是一个合格的异步回调面。
- **控制面：A−**。cancel/steer/follow-up/fork/child control/guidance reply 都能不改码介入（`docs/interfaces.md:411-428`）。
- **运行时扩展面：C+**。没有公开的多播 hook / event bus / middleware / webhook / 插件加载器；`EventSink` 是单槽且硬编码；settlement observer 只在私有 `_compose.py` 里 append；第三方不 patch 代码挂不到 agent loop 上。
- **同步拦截/审批：C**。没有 pre-tool veto/改参，delivery check 明确写着 "advice, not a gate"（`engine/agent/session/runtime.py:1132-1141`）；ADR-0012 明确拒绝 per-tool/per-call approval。这是清醒取舍，但对要合规拦截的部署方是硬缺口。
- **契约卫生：C**。`RunEventType = str` 没有版本字段；**唯一文档化的运行时扩展点 `CheckpointProvider` 已从 HEAD 删除，ADR-0060 与 CONTEXT.md 仍在宣称它存在**（详见 §3.4）。这是最刺眼的一条：不是"没预留"，而是"预留过的点烂掉了，没有契约测试拦住"。
- 对照定位：**pi = 回调最富、执行最弱**（41 个类型化事件、`tool_call`/`tool_result` 可拦截改写、SDK/RPC，但无后台、单机、无沙箱）；**DSH = 扩展最强、执行重型**（cordis 插件、waterfall 钩子、审批/沙箱、jobs 事件、UI 槽、SDK/ACP/插件市场，但插件=宿主代码、job 进程内会丢）；**DlightRAG = 持久执行最好、观察最强、扩展最保守**——这是服务型产品的合理取舍，但需要把取舍契约化，否则"预留"只是叙事。

### 如果只做三件事

1. **完成通知 + Web 运行可见性**（让已经存在的后台能力被看见）。
2. **`RunObserver` 多播契约 + webhook/事件订阅 adapter**（把"观察/控制"优势变成可集成能力，零核心改动）。
3. **清掉文档-代码漂移 + 事件契约版本化**（`CheckpointProvider` 要么恢复要么删文档；`RunEventType` 进注册表并带 `schema_version`）。

---

## 1. 判定口径：background 有三层，先别混为一谈

"Background 模式"在 agent 产品里至少指三件不同的事：

| 层 | 定义 | 用户感知 | DlightRAG 现状 | DSH 对应物 |
|---|---|---|---|---|
| **A. Run 级后台** | 提交后离开，Run 在服务端继续，回来收结果 | "我关了页面它还在跑" | **已存在**（durable Run、断连续传、重启恢复） | Session 持久化 + goal round；但 DSH 的 job 本身不跨进程 |
| **B. Run 内 job 级后台** | 同一 turn 里起一个长任务，先返回 handle，稍后取输出/取消 | "它边跑我边干别的" | **不存在**（工具默认阻塞，bash 无 detach） | `dsh-jobs` + `run_in_background` + `job_output/list/kill` |
| **C. 任务/通知级后台** | 任务列表、运行中标记、完成通知、未读 | "我知道现在有几件事在跑、哪件完成了" | **部分存在**（Run durable、但 Web 无列表/通知面窄） | jobs 列表 + 完成 notice + inject/followup 唤醒 |

创始人说"不是产品本身，而是 run 或者 tasks 的时候的一部分"——按字面最接近 **B**；但"要不要 background 模式"这个问法本身可能也包含 **A/C**。三层的结论完全不同，所以下面逐层给证据。

---

## 2. Q1 逐层判断

### 2.1 A 层（run 级后台）：已经存在，且是默认语义——不要重造

证据链（完整引用见底稿 dlight-background.md，已移出仓库，见附录 B）：

| 能力 | 事实 | 证据 |
|---|---|---|
| 无"前台 Run" | 四个入口都先持久化再返回：Web `POST /web/api/answer` 202、REST `POST /answer` 202、MCP `answer` 立即返回 `run_id` 并要求 `get_run` 轮询、in-process `answer()`/`answer_stream()` | `docs/run-runtime.md:31-37`；`adapters/http/browser/routes/chat.py:173`；`docs/interfaces.md:394-399`；`adapters/mcp/tools/answer_runs.py:44-56` |
| 断连不取消 | "Disconnecting a client, closing an event stream, or abandoning a caller-awaited Application call only detaches that observer; explicit cancellation is the only client action that cancels a Run" | `docs/run-runtime.md:80-84`；`engine/runtime/subscription.py:1-9` |
| 事件可续传 | gap-free sequence；订阅先 replay 再 follow；SSE `id: <sequence>` + `Last-Event-ID`/`after`；终态唯一；游标冲突 422 | `docs/run-runtime.md:88-104`；`docs/interfaces.md:426-445,492-495`；`engine/runtime/subscription.py:59-99` |
| 重启可恢复 | claim 用 `FOR UPDATE SKIP LOCKED`、60s lease/20s 续租、fencing epoch；graceful shutdown 把 owned Run 回 `queued`；崩溃后 lease 过期被 reclaim，从 durable checkpoint 继续 | `docs/run-runtime.md:124-203,496-522` |
| 取消/修复语义 | 显式 cancel 才停；Corpus Mutation 模糊结果进 `waiting_for_repair`，等人工 repair/resume | `docs/run-runtime.md:374-415` |
| 多进程水平扩展 | 共享 PostgreSQL + 同一 workspace；任意进程可 reclaim；没有 worker-only 模式 | `docs/run-runtime.md:524-538`；ADR-0007:39-44 |

**判断**：A 层不但存在，而且比 DSH 的 job 模型更适合长任务——DSH 的 live job 和输出 ring 明确随 harness 进程死亡而消失，只有 Session 日志里的 `job/registered`/`job/settled` 事件可持久化，重启不恢复执行（`@deepseek-ai/dsh-jobs-local/README.md` §Known Limitations）。DlightRAG 的 Run 是 durable 的，**在"后台"这个维度上 DSH 是弱项、DlightRAG 是强项**。把 DSH 的进程内 job 模型照搬进来，等于把它的已知缺陷一起搬进来。

**唯一需要补的不是执行，是感知**：见 2.3。

### 2.2 B 层（run 内 job）：真实差距，但当前不是 P0

**现状：Run 内工具默认串行阻塞，只有只读调用能并发。**

- 规则写死在工具契约里："every other call runs alone and is a barrier between them"；只有声明为 `read_only` 的相邻调用最多 **16** 个并发且按 source order 结算（`engine/agent/tools/contracts.py:541-546`、`engine/agent/session/interpreter.py:53`、`docs/run-runtime.md:165-168`）。`read_only` 默认 False（`contracts.py:555`），浏览器动作也串行。
- `bash` 是唯一真正无界的工具：`timeout_seconds` 可选、**无默认值**、上限约 24 天；执行时 `await environment.run(...)` 阻塞到进程结束，输出通过 `emit_update` 流式回传（`engine/agent/tools/files.py:340-348,984-1030`）。
- Run 结束时 `ExecutionEnvironment.aclose()` 会终止所有子进程，注释明说成功 shell 留下的后台任务也要在释放 workspace 前终结（`engine/agent/environment/local.py:310-333,344-475,384-386`）——**进程级 detach 被主动消灭**。
- 其他长工具相对有界：browser 有 navigation/settle/action timeout；MCP Connection 调用默认 60s；chart/html-report 是 bash 里的 PATH 工具，仍走串行 bash；MinerU 解析不在 Answer Run 内，属 Corpus Ingestion lane。
- Run 本身没有 wall-clock 上限；实测研究 Run 从 9.7 分钟到 60 分钟以上（`docs/skills-routing-eval.md:300`）。

**替代物已经覆盖大部分场景，这是"不急着做"的核心依据：**

| 需求 | 现成替代 | 成本 |
|---|---|---|
| Run 内并行干重活 | **Child Session**：`spawn_agent` 立即返回 handle，子 Session 异步并行，父 Run 用 `subagent_status`/`wait_subagent`/`read_subagent` 收结果；父 Run 可 park 并被结果唤醒 | 一个完整模型循环（token 成本）；父 Run 终态会取消仍在跑的子 Session（`docs/run-runtime.md:384-386`） |
| Run 结束后继续干 | **Goal Round**：settlement observer 起下一轮 Run，同一 Session 续跑 | 单进程 arm、崩溃丢一轮（ADR-0060 "One process / No crash sweep"） |
| 重活独立跑 + 轮询 | **Corpus Mutation Run**（独立 lane、独立 durable Run） | 只适用于 corpus 操作 |
| 后台维护工作 | **Memory Pass**（Run 成功后置；DB 同时是队列，每进程 2 并发、600s sweep、claim + 退避） | 不是模型可见原语 |

**结论**：B 层缺的是"轻量、命令级、不花 token"的 job，而不是"后台"本身。它的价值只有在以下条件同时成立时才为正：

1. 单条 `bash`（或单次外部调用）的长尾已经让用户等待不可接受，或
2. 用户明确要"边等边聊/同时跑几件重活"，且 Child Session 的成本不可接受。

这两个都需要数据，而数据已经在系统里：`tool_end` 事件带 `duration_ms`（`docs/interfaces.md:478`），OTel 的 `execute-agent-tool` span 也带工具身份与耗时（`docs/observability.md:69-142`）。**先把长尾查出来，再决定要不要写代码。**

**如果做，最小形态必须"run-scoped"，不要做成平台：**

- 生命周期 = 当前 Run：Run 终态时取消全部 job（与 Child Session 同规则），不跨 Run、不跨进程重启恢复。
- 数据：`run_jobs(job_id, run_id, tool_call_id, command, status, started_at, finished_at, output_handle, exit_code)`；或更轻：直接复用 Run 事件日志 + spill/resource 句柄，不建表。
- 模型面：只加 `bash_background`（或 bash 的 `run_in_background` 参数）+ `job_status`/`job_output`/`job_cancel`。
- 通知：job settle 时按 child-notification 通道把一条消息注入父 Run（复用 `completed_dispatch_notifications`/父 Run park-wake 模式，`engine/answer/execution/executor.py:1813-1834`）；父 Run 空闲时用 follow-up 唤醒（已有 `RuntimeControlPort` 的 follow-up 语义）。
- 输出：有界 ring + 绝对字节 offset + 双游标（模型消费 / UI 只读）——DSH 这套设计值得直接抄（见 §5）。
- **不做**：跨 Run job、cron、webhook、worker-only 进程、新建 Run kind、跨重启 job 恢复。崩溃时按现有 effect 语义记 `outcome_unknown`（bash 的 `replay_policy="never"`，`contracts.py`），不要发明第二套恢复逻辑。
- 准入门槛：长尾指标 ≥ 阈值（例如单 tool call P95 > 2–3 分钟且占比不可忽略），否则不做。

### 2.3 C 层（任务/通知）：真缺口，应该是 P0

这是"后台能力已经存在，但用户感知不到"的典型问题：

| 缺口 | 事实 | 证据 |
|---|---|---|
| 完成通知面极窄 | 只有"同一 tab 正在跟这条 Answer + 页面失焦/隐藏"才发浏览器 `Notification`；running true→false 才触发 | `frontend/ui/notifications.ts:54-92`；`frontend/ui/app.ts:387-389` |
| 无服务端通知/未读 | 全仓无用户级 notification 表、无 service worker/Web Push、无邮件/webhook | grep 结果；底稿 dlight-background.md §5 |
| Web 无 Run 列表 | REST 有 `GET /runs`（`adapters/http/rest/routes/runs.py:113-119`），Web 没有对应页面；会话列表 summary 无 status/运行中字段 | `frontend/api/conversations.ts` conversationSummary；`frontend/lib/router.ts` |
| 一个 tab 只跟一个 Run | "this tab's one followed run"；离开会话路由即 detach 不取消 | `frontend/lib/run-controller.ts:109-111,174-216` |
| 长 Run 占用 Query slot | Answer 无超时，可跑数十分钟，占 1/16 进程 slot；但 lease/恢复/admission 已覆盖，**暂无证据表明是瓶颈** | `docs/configuration.md:814-817`；ADR-0007 |

**判断**：C 层是 P0，因为它的成本远低于 B 层，却直接决定 A 层 durable Run 的产品价值。用户不是需要"又一个后台模式"，而是需要**知道后台在跑什么、什么时候跑完**。

### 2.4 Q1 决策树

```
用户/产品要什么？
├─ "关页面它继续跑"            → 已满足，不开发（A 层）
├─ "离开后知道完成 / 看到在跑什么" → P0：通知 + Web 运行可见性（C 层）
├─ "run 内长命令不阻塞，边跑边干"  → 先查 tool duration 长尾
│    ├─ 未过阈值 → 不做；用 Child Session / 拆 Run
│    └─ 过阈值   → P1：run-scoped job 最小闭环（B 层）
├─ "跨 run 调度/定时/优先级"      → 当前无证据；用 Goal Round + 外部调度，不开 cron
└─ "第三方挂到 agent loop"        → 见 Q2：RunObserver + webhook，而不是 job
```

---

## 3. Q2：回调点评分卡

评级口径：**A** = 有公开、稳定、可核对的契约；**B** = 能用但需改码/私有根/部署配置；**C** = 基本没有，或只有内部硬编码。评级针对"作为可被外部集成/扩展的 agent 产品"，不针对"内部功能继续长"。

| 维度 | DlightRAG | DSH | pi agent |
|---|---|---|---|
| 生命周期事件（观察） | **A−** durable event log、序号、重放、终态恰好一次；但无模型调用边界（ephemeral 被丢）、无事件类型注册表 | **A−** Session 日志 + `agent/*`/`tools/*` 事件，类型丰富，嵌在插件体系里 | **A** 41 个类型化 `pi.on()` 事件，全公开类型 |
| 外部订阅/推送 | **B−** SSE/REST/in-process 订阅/OTLP；**无 webhook、无事件订阅注册 API、MCP 只能轮询** | **A−** SDK JSON-RPC、ACP、Typert Remote、jobs 事件；插件 UI 流 | **A−** SDK `subscribe()`、RPC 事件流 |
| 运行中介入（control） | **A−** cancel/steer/follow-up/fork/child control/guidance reply，幂等键与 scope 齐全 | **A−** SDK/ACP/steer/interrupt/审批；工具粒度控制多 | **B** steer/follow-up/abort；换会话要 `waitForIdle()` |
| 同步拦截/审批 | **C** 无 pre-tool veto/改参/审批；`delivery_check` 是 advisory；ADR-0012 明确拒绝 per-call approval | **A** `tools/pre-execute|execute|post-execute` waterfall、单调 `guard`、审批/沙箱升级 | **A−** `tool_call` 可改入参/阻断、`tool_result` 可改写结果；无内建审批 |
| 注册式扩展 | **B−** MCP 出站工具、Skills、模型 provider、Runtime Policies；无第三方工具/插件注册 | **A** cordis 服务/工具/prompt/命令/skill/projection/UI slot/插件市场 | **A** 工具/命令/provider/MCP/虚拟模型/UI 扩展 |
| 外部协议/SDK | **B** REST、MCP server（18 个公开工具）、in-process API；无 SDK/ACP/webhook | **A−** MCP/SDK/ACP/插件市场 | **A−** SDK/RPC/JSON 模式 |
| 隔离/安全 | **B+** owner/scope、confinement、pinned policies、workspace 配额；无 OS 级沙箱、无审批 | **B−** 沙箱与审批齐全，但插件=进程内宿主代码 | **C+** 无沙箱；project trust 只控加载 |
| 契约稳定性 | **B−** 事件语义文档化但 `RunEventType=str` 无版本；**CheckpointProvider 已删而文档仍宣称** | **A−** 包级合约、README/源码地图、版本化 | **A−** 公开 TS 类型、加载/失败语义明确 |
| **综合** | **B / B+**：观察控制强、扩展拦截弱 | **A−**：完整 harness，但重型且插件高危 | **A−**：单机可编程性上限 |

### 3.1 DlightRAG 真正"好"的三个回调点

1. **Run 持久事件序列 + SSE 契约**。序号即 SSE id，可断线续传、可重放、终态恰好一次、错误隔离、跨进程一致、脱敏统一。这是全仓唯一"像回调契约"的东西，也是应对外部集成的第一资产。
2. **外部控制面**。cancel/steer/follow-up/fork/child control/guidance reply 让外部在 Run 运行中介入而无需改码；幂等键、scope 授权、失败/终态 409 语义齐全。很多 agent 产品连这个都没有。
3. **内部的两个点位选得对**：`SettledRun`（终态后、schedule-only、失败不伤 Run，`coordinator.py:984-1006`）与 `AgentSessionEvent`（op/turn/tool 提交点，26 种 kind）；以及 `delivery_check`（起草即终稿前的 advisory 注入，`engine/agent/session/runtime.py:1132-1191`）。它们是最值得公开化的内部点位。

补充：内部其实还有一批"准回调点"——`AgentRuntimeEffects`（`assemble_request`/`call_provider`/`execute_tool`/`compact` 的闭合协议）、`RuntimeControlPort`（poll/ack）、`ToolRuntime.emit_update`（工具→运行时的活进度回调）、tool composition 里的 `_recording_execution`/`_ledger_backed`/`_bounded_injected_result` 包装（组合期中间件的雏形）。**位置是有的，问题是它们都没有被契约化、多播化、文档化。**

### 3.2 最缺的三个回调点

1. **没有统一、可多播、可注册的 hook 面**。`EventSink` 单槽硬编码（`engine/agent/session/runtime.py:394,414`）、settled observers 只在私有 `_compose.py:748,818` append、cancelled observers 也是 compose 里拼 list。想加第二个观察者就得改 `_compose.py`——这与"产品级回调点"距离最远。
2. **唯一文档化的运行时扩展点已从代码消失，文档仍在宣称**（见 3.4）。这说明"扩展点"目前是叙事而非契约：没有 Protocol 文件、没有注册处、没有契约测试，删除可以发生在一个标题无关的提交里。
3. **没有同步前置拦截/审批/可编程 gate**。工具调用前不可否决、不能改参、不能要求人工确认。ADR-0012 把 "no per-tool or per-call approval" 写成明确取舍——若产品坚持"agent 自主"，这不是缺陷；若部署方要在 loop 上做合规/成本拦截，这是硬缺口。

### 3.3 为什么说"回调点算中等偏好"

分两种目标看：

- **作为"内部功能继续长"的架构**：其实很好。端口/适配器、组合根、effects 协议、事件表、tool 契约、policies、delivery gate 都在正确的位置上，加功能不需要推翻结构。
- **作为"对外开放的 agent 平台/可嵌入产品"**：不够。外部能"看"和"控"，但不能"挂"和"改"；而且连"看"的契约都缺版本号与机器可校验的事件类型。

与 pi/DSH 的差距不是"谁更先进"，而是**产品形态决定回调面**：pi 是单用户本地 CLI，扩展代码与主进程同权限，天然可以给 41 个进程内事件；DSH 是插件化桌面 harness，用 cordis 把每个能力做成可替换插件，代价是插件即宿主代码；DlightRAG 是多租户服务端产品，让第三方代码进主进程会直接击穿安全模型——**所以它选择 skills/MCP/policies 这类声明式扩展是合理的**。真正的问题是：**声明式扩展之外的那层"观察/通知/集成"契约没有补齐**。

### 3.4 一个必须处理的发现：CheckpointProvider 文档漂移

- ADR-0060 第 4 条仍写："The extension point is a `CheckpointProvider`. The round prompt is composed from registered providers..."（`docs/adr/0060-...md:70-76`）；`CONTEXT.md:44-47` 也把 checkpoint provider 定义为 harness extension。
- 但 HEAD 的 `application/answer/goal_rounds.py:300-314` 只有写死的 `goal_round_prompt(goal, round_number)`，`src/` 检索不到 `CheckpointProvider`；`application/answer/checkpoints.py` 源文件已不在，只剩一个陈旧的 `__pycache__/checkpoints.cpython-314.pyc`。
- `git show fb456692` 确认：这次提交移除了 `providers`、`_findings()` 与 findings 注入，改成"Run 自己的 pre-delivery checkpoint 在轮内处理"；而提交标题是 "feat(agent): a pre-delivery checkpoint lets the model correct a skipped required read"。`85d70cff` 删除了 Protocol 文件。
- **影响**：文档在向客户/团队承诺一个不存在的扩展点；任何按 ADR 实现的集成都会扑空。要么恢复一个真正的 `CheckpointProvider` 注册表（把 delivery findings 作为它的一个实现），要么把 ADR-0060 与 CONTEXT.md 改成当前状态。并加一条契约测试：文档宣称的扩展点必须在代码里存在。

---

## 4. Q2 建议：把"位置"变成"契约"，不做插件框架

原则：**不引入通用 plugin 框架**（ADR-0012 已拒绝，且与产品语言冲突）；**不新造第二条事件真相**；所有回调以现有 durable sequence 为幂等键；**v1 只观察、不否决**。

### L1（P0，引擎内）：`RunObserver` 多播契约

```python
# engine/runtime/observers.py —— 只依赖 RunEvent / SettledRun，不 import application / skill
class RunObserver(Protocol):
    async def on_event(self, event: RunEvent) -> None: ...  # 按 sequence 有序、at-least-once
    async def on_settled(self, run: SettledRun) -> None: ...  # 终态后，schedule-only
```

- `RunCoordinator` 已有 `observe_settled`；再加一个事件多播队列，复用 `follow_run_events` 的游标语义。
- 语义：观察者失败永远不能失败 Run（沿用 `_emit_observe_only` 语义）；按 `(run_id, sequence)` 幂等；只读，不修改事件或 Run 状态。
- 注册：`_compose.py` 私有根注入；公开入口 `create_application(..., observers=())` 只传入，不导出 registry。这样嵌入方可用、第三方不能随意注入。
- 立刻收益：Memory Pass、Goal Round、未来的 webhook/审计/计费都从"改 compose 加一行"变成"注册一个 observer"。

### L2（P1，Edge 适配器）：webhook / 事件订阅 adapter

```python
# adapters/webhooks/dispatcher.py —— 新增 adapter，零核心改动
# 输入：application.runs.subscribe(owner_id, run_id, after_sequence=游标)
# 输出：POST {run_id, sequence, event_type, payload, schema_version} 到部署方 URL
# 投递：按 sequence 保序、at-least-once、幂等键 (run_id, sequence)、失败重试 + 死信
# 脱敏：复用应用层 reader content gate（skill_protection.mask）
```

- 这不是新架构：它与 SSE 并列消费同一个 Application 订阅面，仍以 PostgreSQL 为唯一真相；部署方只配置 URL/secret，不改代码。
- 它填补的是"第三方不 patch 代码无法挂到 agent loop"的最大缺口——90% 的集成需求其实是"run 完成通知我"，不是"改 agent 行为"。
- 与 `docs/run-runtime.md` 的"Run 是唯一权威"完全一致；失败重试不反写 Run。

### L3（P1，契约卫生）：事件契约版本化

- `RunEventType` 从 `str` 提升为注册表（或至少一个集中的 `Literal`/枚举），每个事件带 `schema_version`。
- 把 `tool_start/tool_progress/tool_end/progress/token/reasoning/reset/memory_operation_settled/done/error` 写成 machine contract 并加契约测试；phase 词表也应收口（现在 executor-owned 字符串）。
- 如果要把 SSE 当公开 API 卖，这是前置条件。

### L4（P1，恢复或删除）：CheckpointProvider

- 恢复：`application/answer/checkpoints.py` 定义 `CheckpointProvider` Protocol + 注册表 + 逐 provider try/except（这正是被删掉的实现所做的），Goal Round 与 delivery check 共用；把 delivery findings 作为它的一个实现。
- 或删除：同步修改 ADR-0060 第 4 条与 `CONTEXT.md`，明确"pre-delivery checkpoint 是轮内机制，不是可注册扩展点"。
- 加一条"文档-代码一致性"测试，防止再漂移。

### L5（P2，仅企业需求）：advisory pre-tool + 显式 veto 工具

- 不做通用 middleware。最小改动是 `delivery check` 的兄弟：一个 advisory 的 `pre_tool` 建议 + 一个显式 veto 工具（模型可调用、审计可见），而不是任意代码拦截。
- 保持 ADR-0012 的"agent 自主"哲学；只在合规/成本场景开启。

---

## 5. 竞品启示：抄什么、不抄什么

### DSH 值得抄的 5 条（详见底稿 dsh-reference.md，见附录 B）

1. **seam 抽象包 / 实现包 / 模型面 / 观察面四段分离**：`dsh-jobs`（抽象 Service，直接 load 抛错）/`dsh-jobs-local`（实现）/`dsh-tool-jobs`（模型工具）/`dsh-api-job-controller`（观察 API）。同一形状复用于 fs、sandbox、compaction、persistence、subagent。
2. **owner/session 是统一授权轴**：job id 可预测，控制操作按 owner 鉴权，不靠 id 保密。
3. **事件而非轮询；提交/通知/观察三分离**：先落 `job/registered`/`job/settled`，再放 waiter，再发含 `awaited` 的 settled 事件，最后才是 notice；模型侧唯一消费游标，观察者走非消费 offset。
4. **run 内 job 的"前台也是 job + 超时提升"**：前台 bash 也先注册为 job，超时后"提升"为后台 job 并复用同一个 id（`dsh-tool-bash` §Foreground commands as jobs）。这是整个 DSH 后台设计里最值得抄的单点——把"超时失败"变成"交接"。
5. **结算通知的 busy/inject · idle/wakeup 双 lane**，以及三种抑制条件（模型自己 kill / `awaited` / `teardown`）。

### DSH 不要抄的（对 DlightRAG 是倒退）

- **进程内 job、重启不恢复**：DlightRAG 的 Run 已 durable，不要引入"随进程死亡"的第二类任务。
- **不排队、达上限直接报错**：DlightRAG 已有 admission/队列语义，job 应服从 Run 的资源边界。
- **插件市场 = 进程内执行宿主代码**：多租户服务端不能接受。
- **turn 退役窗口的通知丢失洞**：DSH 自述的已知问题；DlightRAG 的 durable 事件 + settlement observer 正好可以做得更稳。

### pi 值得抄的 4 条（详见底稿 pi-reference.md，见附录 B）

1. **"settle 前可行动边界" vs "settled 终态"的二分**：`turn_end`/`agent_before_settle`（可追加记录 + `continue:true` 一次）vs `agent_settled`（终态仅通知）。这正好映射 Goal Round 的"settle 前 checkpoint provider 请求一次续跑"，天然防无界自驱。
2. **工具级拦截分执行前/结果后两个可改写点**：`tool_call` 可就地改入参/阻断（handler 异常 fail-safe 阻断），`tool_result` 可改 content/details/structuredContent/isError。对应 DlightRAG 的 in-loop gate（工具前阻断）与 delivery record preview（结果后脱敏）。
3. **`content`（给模型）与 `structuredContent`（给平台/前端）分离**：DlightRAG 的 `ToolResult.parts` vs `facts/details/telemetry` 已有同构，但可进一步明确"同步替换"的规则。
4. **错误分级**：通知型失败不影响主流程；拦截型失败 fail-safe 阻断；工具失败变 `isError`；另设 `extension_error` 事件流给客户端。DlightRAG 的 observer/delivery check 失败目前只写 log，可以按此分级暴露到 Web 与 observability。

### pi 不要抄的

- **无沙箱、扩展与主进程同权限、工具不可注销、无后台**：这是本地单用户产品的取舍，不是服务端产品的模板。
- **把"后台"交给宿主**：DlightRAG 的宿主就是自己的 RunRuntime，必须在产品内解决。

---

## 6. 风险与开放问题

### 风险（若按建议推进）

| 风险 | 说明 | 缓解 |
|---|---|---|
| RunObserver 变成隐形的第二事件总线 | 观察者多了以后，可能有人想改行为、加 veto | 契约写死"observe-only、失败隔离、按 durable sequence 幂等";改动行为必须走显式新 tool/新 gate，不走 observer |
| Webhook 多租户越权/泄露 | Run 有 owner/workspace 两种 scope，event 含 query/答案/工具信息 | 按 owner 鉴权；复用 reader 脱敏 gate；payload 默认只给元数据，正文需显式开关 |
| Job 原语把 Run 变复杂 | 崩溃恢复、effect 语义、workspace 冲突、UI trace 都要改 | 严格 run-scoped、lifetime=Run、outcome_unknown 兜底；先指标后实现 |
| 通知打扰/重复 | 多 job/多 run 同时结算 | 批量聚合、未读表、一次只发一条摘要 |
| 文档继续漂移 | 已有前科 | 文档-代码一致性测试 + 契约测试进 CI |

### 开放问题（需要 founder 拍板）

1. **"background 模式"到底指哪层？** A 层已有、B 层是差距、C 层是缺口。不澄清，方案会打偏。
2. **产品定位是"研究服务"还是"agent 平台/harness"？** 决定 Q2 是只做观察/集成契约，还是也要做运行时扩展/拦截。
3. **是否要 B2B 集成/嵌入？** 若是，webhook + 事件契约的优先级应高于 job。
4. **关掉 tab 后是否必须收到通知？** 决定 P0 是浏览器通知（现有）还是 service worker/Web Push/邮件（独立项目）。
5. **长命令/长 Run 的真实分布？** `tool_end.duration_ms` 与 OTel `execute-agent-tool` span 已可查；这是 B 层唯一准入证据。
6. **是否接受 Goal Round "单进程 arm、崩溃丢一轮"？** 这可能比 background mode 更值得先修。
7. **是否需要合规审批？** 若需要，会动到 ADR-0012 的产品哲学，应单独决策，不要混在 background 里。

---

## 附录 A：关键证据索引（仓库内）

| 主题 | 文件 |
|---|---|
| Run 生命周期/断连/恢复/取消 | `docs/run-runtime.md`（31-37, 80-84, 88-104, 124-203, 374-422, 496-538） |
| 事件/SSE 契约 | `docs/interfaces.md`（409-428, 437-495）；`src/dlightrag/engine/runtime/subscription.py` |
| 事件信封/类型 | `src/dlightrag/engine/runtime/records.py:20-26,257-263` |
| 结算观察者 | `src/dlightrag/engine/runtime/coordinator.py:543-555,984-1006`；`src/dlightrag/_compose.py:748,818` |
| delivery check（advisory） | `src/dlightrag/engine/agent/session/runtime.py:1132-1191` |
| 工具契约/只读并发 | `src/dlightrag/engine/agent/tools/contracts.py:530-620`；`src/dlightrag/engine/agent/session/interpreter.py:53` |
| bash | `src/dlightrag/engine/agent/tools/files.py:340-348,971-1060` |
| 子进程清理 | `src/dlightrag/engine/agent/environment/local.py:310-333,344-475` |
| Child Session 异步 | `src/dlightrag/engine/answer/tools/subagents.py:106-107,647-820`；`src/dlightrag/engine/answer/execution/executor.py:1807-1834`；`docs/domain-language.md:199-205` |
| Goal Round / CheckpointProvider 漂移 | `src/dlightrag/application/answer/goal_rounds.py:82-100,300-314`；`docs/adr/0060-...md:70-76`；`CONTEXT.md:44-47`；提交 `fb456692`、`85d70cff` |
| Memory Pass | `src/dlightrag/application/memory/session_pass.py:1-30,470-518` |
| 前端通知/可见性 | `frontend/ui/notifications.ts:54-92`；`frontend/ui/app.ts:387-389`；`frontend/lib/run-controller.ts:109-111,189-296`；`frontend/lib/chat-views.ts:19-21` |
| REST Run 列表 | `src/dlightrag/adapters/http/rest/routes/runs.py:113-119` |
| Observability 契约 | `docs/observability.md:1-142`；`docs/adr/0017`；`src/dlightrag/engine/ai/telemetry.py:24-68` |
| ADR-0012（拒绝通用 plugin / 无 per-call approval） | `docs/adr/0012-personal-connections-and-hot-plug.md:30` |

## 附录 B：四份取证底稿（已移出仓库）

本文的证据来自四份底稿：`dlight-background.md`（后台执行现状/缺口）、`dlight-callbacks.md`（回调点全景）、`dsh-reference.md`（DSH 后台任务解剖）、`pi-reference.md`（pi agent 后台能力）。它们已从仓库删除，可在 git 历史中找回（`git log --diff-filter=D -- docs/research/background-callbacks/`）。
