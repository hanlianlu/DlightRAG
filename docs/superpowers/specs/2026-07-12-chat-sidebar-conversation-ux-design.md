# DlightRAG Chat Sidebar 与多会话导航 UX 设计

**日期**：2026-07-12
**状态**：已审阅并完成；Slice A 已通过 Task 9 验收
**依赖**：[Web-only 多模态会话与 Answer 图片编排](2026-07-12-durable-scoped-multimodal-conversations-design.md)
**范围**：Web 左侧 Chat Sidebar、conversation lifecycle UX、响应式布局

## 1. 决策摘要

- Web 增加独立左侧 Chat Sidebar，提供多 conversation 导航。
- New Chat、conversation list、Rename 和 Delete 位于左侧栏；不提供 Clear messages。
- Primary action 的可见文案固定为 sentence case `New chat`。
- 现有 topbar Clear 按钮删除。
- 现有右侧 Files/Sources panel 保留，不能复用为聊天列表。
- 桌面端左侧栏默认展开并可折叠；窄屏使用 overlay drawer。
- 服务端 conversation list/history 是事实源；前端只保存 active conversation ID 和纯 UI 状态。
- conversation_id 由服务端创建，所有 lifecycle 操作保持 principal scoped。
- Sidebar 不重复展示 DlightRAG 品牌；视觉原则是 simplicity with elegance，所有顶区共享
  一条经过光学校正的基线，不增加装饰性分割线。
- 本版不增加 conversation search、pin、folders、sharing、archive 或 LLM title generation。

## 2. 为什么单独成篇

Conversation persistence 和 sidebar UX 属于同一产品迭代，但不是同一组件：

- Persistence 决定 ownership、数据生命周期、历史图片和 answer prompt。
- Sidebar 决定 navigation、layout、responsive behavior 和用户操作反馈。

分成两份 spec 可以让实现共享同一 lifecycle contract，同时避免在修复多模态
answer 边界时无限重构前端布局。

## 3. Slice A 实施前 UI（历史背景）

Slice A 实施前页面只有：

- topbar 左侧 workspace selector。
- topbar 右侧 Clear 和 Files。
- 中央 chat area/composer。
- 右侧可调整宽度的 Files/Sources slide-out panel。

当时右 panel 通过 body.panel-open 改变 chat/topbar/composer 的 margin-right，
并在 640px 以下变为全屏。

Chat Sidebar 必须新增独立的左侧 layout state，不能把现有通用 panel 改成左右
复用组件；Files/Sources 的内容和生命周期与 conversation navigation 不同。

## 4. 信息架构

~~~text
┌──────────────────┬────────────────────────────┬──────────────────┐
│ Chat Sidebar     │ Topbar / Chat / Composer   │ Files / Sources  │
│                  │                            │ optional panel   │
│ New chat     [<] │ Search in: All workspaces │ Files in: default│
│                  │                            │                  │
│                  │ messages                   │ source details   │
│ Current chat  ···│                            │ or files         │
│ Earlier chat  ···│                            │                  │
│ Another chat  ···│ composer                   │                  │
└──────────────────┴────────────────────────────┴──────────────────┘
~~~

左侧栏区域：

1. Top row：New Chat primary action 和 collapse/close 控件；不重复品牌。
2. New Chat、中央 Search in 和右侧 Files in 共享同一 top baseline。
3. Scroll region：按 updated_at 降序的 conversation list。
4. 每行：title、active state、actions menu。

不增加底部账户菜单；认证/account UX 不属于本功能。

## 5. Conversation row

每个 row 展示：

- title；空 title 显示 “New chat”。
- active conversation 的明确选中状态。
- hover、focus-within 或 active 时显示 actions button。

Title 使用单行省略，不把完整首条 query 撑高为多行。

Actions menu：

- Rename
- Delete

菜单必须是 button/menu 交互，不使用仅 hover 可见且无法键盘操作的 div。

不在列表中展示 workspace 标签。Conversation 可跨 workspace 延续，把 workspace
放入 row 会错误暗示 conversation 属于某个 workspace。

## 6. Lifecycle UX

### 6.1 Initial load

1. 请求当前 principal 的 conversation list。
   服务端不会返回 `updated_at` 已超过 30 天 TTL 的 conversation。
2. 若 localStorage active ID 仍在列表中，选择它。
3. 否则选择最近更新的 conversation。
4. 若列表为空，创建一个空 conversation 并选择。
5. 加载 selected conversation history。

Active answer stream 存在时，不允许切换 selected conversation；用户先 Stop。
Restored assistant turns 使用服务端 answer_sources 重建 citations、references 和
visual gallery；不能只渲染 assistant_text。

### 6.2 New Chat

- Active answer stream 存在时禁用；用户先 Stop 再创建。
- Composer 有未发送文字或图片时，先确认是否丢弃；本版不做
  per-conversation draft persistence。
- 调用 Create route。
- 服务端返回新的 conversation_id。
- 立即设为 active 并写入 localStorage。
- 清空当前 message viewport 和 pending composer images。
- 关闭并清空 SOURCES；保持 FILES 及其 ingest workspace 不变。
- 保留当前 active workspace selection。
- Composer 获得焦点。

New Chat 不复制上一 conversation 的 text、history images 或 resolver catalog。

### 6.3 Select

- Active answer stream 存在时禁用；用户先 Stop 再切换。
- Composer 有未发送文字或图片时先确认丢弃。
- 更新 active conversation ID。
- 中止旧 history request。
- 加载目标 conversation history。
- 更新 selected row。
- 关闭并清空 SOURCES；保持 FILES 不变。
- 不改变 active workspaces。

History load 使用 request generation/token 防止慢响应覆盖更新后的 active
conversation。

### 6.4 Rename

- Row 进入 inline edit。
- Enter 或失焦提交；Escape 取消。
- Trim 后为空则不提交。
- 服务端成功后更新 title。
- 失败时恢复原 title，并显示 toast。

Title 长度由服务端 contract 校验；前端不能成为唯一防线。

### 6.5 Delete

- 显示 destructive confirm dialog。
- 成功后从列表移除 conversation。
- 若删除 active conversation，选择下一条最近 conversation。
- 若没有剩余 conversation，创建并选择新的空 conversation。
- Active Delete/fallback 会关闭并清空 SOURCES；FILES 保持不变。

删除失败时保持当前 row、history 和 selection 不变。
如果最后一条已成功删除、随后创建空 conversation 失败，显示真实空列表和
“Retry New Chat”；不能伪装恢复已删除 row。

## 7. 标题

本版不为 title 增加额外 LLM call。

- 新 conversation 的 title 为空，UI 显示 “New chat”。
- 第一个成功 turn 保存时，如果 title 为空，服务端使用第一条 user query
  生成 whitespace 合并后的单行标题，最多 120 个 Unicode 字符。
- 用户 Rename 后，后续 turn 不自动覆盖。

Title 只用于 navigation，不进入 retrieval、resolver 或 answer prompt。

## 8. Desktop layout

### 8.1 Persistent sidebar

较宽 viewport：

- sidebar 固定在左侧。
- 默认展开。
- 宽度使用 --layout-chat-sidebar-width: clamp(260px, 22vw, 288px)。
- 中央区域使用 --layout-chat-min-width: 520px。
- topbar、chat-area 和 composer 使用同一 margin-left。
- Sidebar New Chat、中央 `Search in: …` 和右 panel 的 `Files in: …` 使用同一高度、
  line-height 和 optical top inset；不能只让 CSS box 对齐而文字视觉基线不齐。
- 展开/折叠动画共享同一 duration/easing token。
- 折叠状态保存在 localStorage，属于设备 UI preference。

折叠后 sidebar 完全退出内容区；topbar 提供打开按钮。首版不实现额外窄 icon rail，
避免同时维护第三种导航形态。

### 8.2 与右 panel 共存

- Chat Sidebar 只改变左侧占用。
- Files/Sources panel 继续改变右侧占用。
- 中央 message/composer 的 max width 同时考虑左右可用空间。
- 两侧都打开时不得让 composer 位于 panel 下方。
- 现有 right panel resize 只改变右侧宽度，不改变 sidebar。
- Right panel resize 上限为
  viewport width - open sidebar width - --layout-chat-min-width；
  不能继续只按 viewport 50%。
- 宽屏点击 sidebar 不触发 right panel 的 outside-click close。
- 切换 conversation 时，FILES 与 conversation 无关，应保持打开且不刷新。
- SOURCES 绑定当前 answer 的证据；切换 conversation 时必须关闭并清空。

## 9. Responsive behavior

### 9.1 Wide desktop（>=1200px）

- 左侧栏固定、默认展开、可折叠，使用统一 sidebar width token。
- 右侧 panel 保持现有 desktop shell。
- 左右可以同时打开，中央区域同时扣除两侧实际宽度。

### 9.2 Compact desktop/tablet（641–1199px）

- 左右都作为 overlay drawer，默认关闭。
- Chat Sidebar 宽度约 320px。
- 两个 drawer 互斥；打开一侧先关闭另一侧。
- FILES 正在执行 upload/delete mutation 时，不允许通过打开 Chat Sidebar
  静默关闭并取消该请求；保持 FILES 打开并提示等待 mutation 完成。
- FILES 的只读 load/poll 可以在 drawer 切换时取消，重新打开后刷新。
- 不改变 composer/message width。
- 点击 backdrop、Escape 或成功选择 conversation 后关闭 drawer。

### 9.3 Mobile（<=640px）

- Chat Sidebar 和 Files/Sources 都使用全屏 drawer。
- 两个 drawer 互斥；打开一侧先关闭另一侧。
- drawer 使用 modal semantics；打开时背景 inert、锁定滚动、focus trap，
  关闭后恢复触发按钮。
- Composer 和当前 stream 状态在开关 drawer 时不重建。

Escape 只关闭最上层 surface；lightbox、menu、dialog、left drawer、right drawer
不能在一次 keydown 中全部关闭。

## 10. Frontend state

建议状态：

- conversations：服务端 list 的 UI projection。
- activeConversationId：localStorage 持久化。
- history：当前 active conversation 的服务端 projection。
- sidebarOpen：当前 viewport UI state。
- sidebarCollapsed：desktop preference。
- pendingLifecycleAction：防止重复提交。
- hasUnsavedDraft：New/Select 前确认未发送文字或图片。
- request generation：防止 stale list/history response。

禁止：

- 把 conversation transcript 当成可提交给 answer 的事实源。
- 在前端生成 ownership principal。
- 用 workspace ID 拼 conversation ID。
- 在多个 store 各自保存不同 active conversation。

Conversation state change 应通过现有 typed event/store 路径通知 chat renderer，
不使用散落的 DOM custom events 直接拼状态。

## 11. Route interaction

Sidebar 使用 Web-only lifecycle routes：

- List conversations。
- Create conversation。
- Read selected history。
- Rename conversation。
- Delete conversation。

Create/Rename 成功后，以服务端返回 summary 为准更新列表；不根据客户端
猜测 updated_at 或最终 title。Delete 返回 204，成功后按已知 conversation_id
移除 row。

Failure 行为：

- 404：从本地 list 移除 stale row，选择 fallback conversation。
- 422：保留当前状态并显示输入错误；不得清空 viewport。
- 503：保留当前状态并提供 retry。
- 5xx/network：保留当前状态并提供 retry。

## 12. Streaming interaction

- 每个 answer stream 绑定发起时的 conversation_id。
- renderer 在处理 token/done/error 前验证该 ID 仍是 active。
- 当前 stream 进行中时可以打开/关闭 sidebar，但 lifecycle mutation 默认禁用，
  防止保存目标和可见目标分离；UI 提示先 Stop response。
- Stop 触发浏览器 abort；服务端取消 answer/upstream 并跳过 persistence。
- 只有当前 stream promise 完成 cancellation cleanup 后，才允许 New、Select、
  Delete。
- Rename 不改变内容 revision，可以在 streaming 时使用。
- 仅打开/关闭 sidebar 不 abort、重建或转移当前 stream。
- done event 若返回 conversation_saved=false、reason=conversation_changed，
  将可见 answer 标记为未保存，提示 Reload/Retry；不得把它加入 history projection。

## 13. Accessibility

- Sidebar 使用 nav landmark 和明确 aria-label。
- New Chat、collapse、row 和 actions 都是原生 button。
- Active row 使用 aria-current。
- Actions menu 支持 Enter、Space、Escape、Arrow keys。
- Rename input 有可见 focus ring 和 label。
- Drawer 打开后 focus 进入 sidebar；关闭后回到触发按钮。
- Compact/mobile drawer 设置 modal semantics、背景 inert 和 focus trap；
  desktop persistent sidebar 不是 modal。
- Confirm dialog 使用原生 dialog 或等价 focus trap。
- prefers-reduced-motion 时关闭位移动画。
- 所有 icon-only controls 必须有 accessible name。

## 14. Visual language

- 沿用现有 dark surface、gold accent、spacing、radius 和 typography tokens。
- Active conversation 使用低对比 surface + 清晰文本，不增加高饱和背景。
- New Chat 是 sidebar 内唯一持续可见的 primary action。
- Destructive Delete 只在 menu/dialog 使用 danger color。
- Sidebar 不显示额外 DlightRAG 品牌、`Chats` 标题或说明性 header；conversation rows
  本身已经表达内容类型。
- 顶区不增加贯穿页面的装饰性 divider；层级由 spacing、type scale 和 surface 建立。
- 不重做 message bubbles；workspace、Files、Sources 和 composer 只做本规格明确列出的精修。

### 14.1 Search、Files、Sources 与 composer 精修

- `Search in: …` 位于中央 topbar，选择结果跨多个问题和多个 Chat 保持；默认是当前
  principal 可查询的全部 workspaces，而不是 `default`。
- Chat Sidebar 只管理 conversation lifecycle；Search scope 不进入 Sidebar。
- `Files in: …` 独立表示右侧文件管理的单 workspace target。既然 panel 已显示
  `Files in: …`，内部不再重复 `FILES` header。
- Files 上传区收敛高度与视觉重量。点击上传区选择 files；folder upload 仅保留为
  低强调 `Choose folder` 文本 action，不并排放两个居中主按钮。
- File row 保持纯 filename 信息层级，不增加 checkbox；Delete 只在 hover、focus-within
  或键盘 focus 时出现。
- SOURCES 继续绑定当前 answer。Source row 在存在 download URL 时提供低强调下载 action；
  不把 locator/path 暴露给浏览器。
- Composer 保留成熟的图片上传、thumbnail、paste/drag-drop 和 send 交互。加号使用靠近
  composer 内边缘的 optical inset，不能漂在内容区中央。
- Send 只有 query text trim 后非空才点亮；仅有 attachment 时保持 disabled。
- 这些调整复用现有 Utopia/token/CSS-module 体系，不引入新的 design-system dependency。

## 15. Loading、empty 与 error states

- Initial list：轻量 skeleton，不用全页 spinner。
- Empty：正常进入一个 “New chat”，不显示错误。
- History loading：保留 chat shell，message area 显示 loading state。
- List failure：sidebar 显示 retry，已有 active chat 不被清空。
- History failure：message area 显示 scoped retry，不切换到其他 conversation。
- Mutation failure：toast + 原状态回滚。

## 16. 删除与迁移

删除：

- topbar Clear button 和对应独立 setup module；不在 Sidebar 提供替代 Clear。
- frontend sessionStore 的浏览器生成 session ID。
- conversationStore.historyWindow 作为 answer request payload。
- Web answer body 中 conversation_history。

替换：

- session_id localStorage key -> active conversation_id key。
- checkpoint restore -> selected conversation history load。

不保留旧 localStorage alias。开发期首次加载没有 active ID 时按 Initial load
规则选择或创建 conversation。

## 17. 保护区

Sidebar UX 不改变：

- Sidebar 组件不额外定义 REST/MCP/Python contract；公共无状态 breaking change 只由配套
  multimodal conversation spec 第 7、16 节定义和验收。
- retrieval 和 answer RAG 语义。
- workspace access control 与 principal 可访问 workspace 集合；selector 的布局、copy、
  persistence 和默认 all-workspaces 行为按 14.1 调整。
- Files/Sources 的现有 routes、数据语义和 mutation 行为；14.1 只调整其布局与视觉呈现。
- source gallery/lightbox。
- composer image capability gate。
- citation rendering 和 streaming event schema，除增加 request-local
  conversation correlation 外。

不能为了复用现有 right panel 而把它改造成左右两用通用抽屉。

## 18. 验收测试

### 18.1 Lifecycle

- Initial load 选择 local active、最近 conversation 或创建空 conversation。
- New Chat 创建 server-owned ID 并保留 workspace selection。
- Select 正确恢复 history、citations、references 和 gallery，不发生 stale response overwrite。
- Rename 成功、取消、失败回滚。
- Delete active 后选择 fallback；空列表时创建新 conversation。
- 最后一条 Delete 后 Create 失败显示真实 empty + retry。

### 18.2 Scope

- List/history/mutations 只操作当前 principal。
- 伪造其他 principal conversation ID 显示 404。
- workspace 切换不改变 active conversation。

### 18.3 Layout

- Desktop sidebar 展开/折叠后 topbar、messages、composer 对齐。
- `New chat`、`Search in: …` 与 `Files in: …` 文字基线在 desktop shell 精确对齐。
- Sidebar 无重复品牌/Chats header，topbar 无装饰性横向 divider。
- 左 sidebar 与右 panel 同开时无覆盖。
- Tablet/mobile drawer 不改变 chat 内容宽度。
- Mobile 两个 drawer 互斥。
- FILES active upload/delete 时 Chat drawer 不会导致 mutation 被 panel close 取消。
- Right panel resize 不影响 left sidebar width。
- Right panel resize 始终保留中央最小可用宽度。
- New/Select/active Delete 保留 FILES，但关闭并清空 SOURCES。
- Files panel 不重复 `FILES` header；上传区 compact，zone click 选择 files，
  `Choose folder` 为低强调 action。
- File rows 无 checkbox，hover/focus Delete 可用鼠标和键盘访问。
- Composer 加号使用 optical inset；空白 query 即使有 attachment 也不能启用 Send。

### 18.4 Streaming

- Active stream 时 New/Select/Delete 禁用，并提示先 Stop。
- Stop 必须等待 cancellation cleanup，partial answer 不保存。
- Sidebar toggle 不重建 composer 或 stream。
- 有未发送 draft 时 New/Select 要求确认。
- conversation_changed done event 标记未保存并提供 Reload/Retry。

### 18.5 Accessibility

- 全部 lifecycle actions 可用键盘完成。
- Drawer/menu/dialog focus 正确进入和返回。
- active row、icon buttons 和错误状态有正确 accessible names/status。
- reduced motion 生效。

## 19. 明确不做

- Conversation search。
- Pin/favorites。
- Folders/tags。
- Share/public links。
- Archive。
- Cross-principal collaboration。
- LLM-generated titles。
- Conversation duplication/fork。
- 多选批量删除。
- 无限滚动或复杂日期分组。

这些功能只有在真实使用需求出现后单独设计。

## 20. Slice A 实施记录

1. 已使用 conversation lifecycle contract tests 锁定服务端行为。
2. 已添加 frontend conversation list/active/history store。
3. 已新增 sidebar shell 和 desktop layout。
4. 已实现 Create/Select/Rename/Delete。
5. 已绑定 stream conversation correlation 与 stale-response protection。
6. 已实现 overlay/mobile/focus behavior。
7. 已删除 topbar Clear、sessionStore 和 client history payload。
8. 已完成视觉、响应式和 accessibility 验证。

配套逐文件实施计划和 Task 1–9 验收已完成；本节保留为已审阅的实施记录。
