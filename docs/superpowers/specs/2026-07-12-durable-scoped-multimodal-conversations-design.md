# DlightRAG Web-only 多模态会话与 Answer 图片编排设计

**日期**：2026-07-12
**状态**：架构已获用户确认，等待书面 spec review
**范围**：Web 会话、历史图片解析、answer 模型图片能力、最终 prompt 编排、旧会话路径清理
**Breaking change**：是

## 1. 决策摘要

本设计采用三个清晰边界：

1. Web 是唯一拥有会话状态的产品界面。
2. REST、MCP、Python 公共接口保持无状态，不接受 history、conversation ID 或历史图片 ID。
3. Retrieval 保持纯 RAG；只有最终 answer prompt 受 answer 模型图片 transport 限制。

具体决策：

- Web 会话按 “principal_id + conversation_id” 隔离，workspace 不参与会话身份。
- principal_id 由服务端认证产生，客户端永远不能提交或覆盖。
- conversation_id 是浏览器生成的非秘密 UUID，只表示该 principal 的哪段对话。
- PostgreSQL 持久化最近 100 个完整 turn 及其图片；旧 checkpoint/session 数据不迁移。
- Web 使用一次 structured LLM completion 解析多轮追问和历史图片引用，不开发 agent、tools 或循环。
- 当前 turn 图片永远先于历史图片；resolver 只选择历史图片，不能删除或替换当前图片。
- “answer.max_images” 默认 10，表示最终 answer 模型收到的全部原图 block 上限，不再只表示 RAG 图片。
- 当前 turn 上传默认最多 3 张，由同一后端配置驱动 Web、REST 和 MCP 校验。
- 启动期 eager 探测实际 answer query-role 模型的视觉支持及图片数量上限。
- Gallery 展示 cited visual sources，不绑定 answer 模型图片上限。
- 已 ingest 的 documents、chunks、KG、vectors、metadata、visual assets 和 jobs 全部不受影响。

## 2. 目标与非目标

### 2.1 目标

- Web 在页面刷新、API 重启和容器重启后恢复最近 100 个完整 turn。
- 当前用户不能读取、引用或删除其他 principal 的会话和图片。
- 切换 active workspace 不分叉对话；每轮仍记录实际查询的 workspace 集合。
- “之前三张图”“第二张收入图”“那些图”等引用由 LLM 结合真实图片 catalog 解析。
- 当前图片、选中的历史图片和 RAG 原图在一个最终 transport 预算中计数。
- RAG visual chunk 即使没有原图 slot，也保留完整的已存储 VLM description。
- 模型不支持图片或能力探测失败时，前后端呈现一致、可解释的行为。
- 流式 capacity 错误在任何可见 token 前处理。
- 删除旧实现，而不是双写、双读或保留兼容 alias。

### 2.2 非目标

- 不为 REST、MCP 或 Python SDK 实现多轮会话。
- 不让公共调用方上传 conversation history。
- 不开发通用图片 agent、tool loop、多阶段 fidelity framework 或批量视觉推理树。
- 不把聊天图片写入 workspace 知识库。
- 不修改 LightRAG retrieval、embedding、rerank 或 ingestion 的产品语义。
- 不迁移当前开发期 checkpoint/session/history 数据。
- 不给 gallery 设置与模型 transport 相同的图片数量上限。

## 3. 当前问题

### 3.1 两套 answer 图片预算造成 provider 超限

当前 AnswerEngine 分别计算 user/history 图片和 RAG 图片。日志中的 images_sent 只覆盖 RAG 部分，而 provider 对整个 messages payload 计数。

故障请求实际包含历史图片 3 张和 RAG 图片 4 张，共 7 张；MiMo 路由只允许 5 张，因此返回 400。

### 3.2 公共接口泄漏了 Web 会话职责

当前 REST 和 MCP contract 暴露 session_id、referenced_image_ids，REST answer 还暴露 conversation_history。它们把 Web persistence 语义扩散到所有调用方式。

上游 agent、REST client 和 SDK 本来就应该自行构造独立请求；DlightRAG 不应提供另一套不完整的 history protocol。

### 3.3 浏览器和服务端重复管理历史

浏览器目前把 conversation_history 和 session_id 一起发送给 Web answer。服务端同时保存 checkpoint，因此有两个事实源，且客户端可以提交与服务端 checkpoint 不一致的历史。

### 3.4 当前图片 bytes 不持久

checkpoint 保存内部图片引用，真实 bytes 位于进程内 SessionImageStore。进程或容器重启后图片消失，API/MCP 进程之间也不能共享。

### 3.5 当前 scope 错把 workspace 放入会话身份

当前 session key 包含 auth mode、user ID、workspace 集合和 session ID。同一用户切换 workspace 后会得到另一段历史。

### 3.6 Vision probe 只有表面三态

Manager 字段可以表示 True、False、None，但 probe 把 timeout、401、429、5xx、服务未启动和明确不支持图片都转换为 False。

此外，当前 probe 检查 llm.default；AnswerEngine 实际使用 llm.roles.query，未配置时才 fallback 到 default。

## 4. 总体架构

~~~text
Browser
  |
  | current query + current images + conversation_id + workspaces
  v
WebConversationService
  |-- WebConversationStore (PostgreSQL)
  |-- QueryImageEnhancer (current-image descriptions)
  |-- ConversationTurnResolver (one structured LLM call)
  |
  | prepared text history + materialized current/history images
  v
Stateless Answer Pipeline
  |-- existing QueryPlanner
  |-- existing RAGService.aretrieve / LightRAG aquery_data
  |-- existing rerank
  |-- AnswerPromptAssembler
  v
answer query-role model
~~~

依赖方向必须保持：

~~~text
web -> core -> retrieval/models
web -> storage

core -X-> web
retrieval -X-> web/session storage
~~~

## 5. Identity 与 scope

### 5.1 principal_id

principal_id 表示“谁拥有这段对话”。它是服务端内部 ownership namespace，不是请求字段。

- JWT：来自验证后 token 的 trust domain + sub。
- simple auth：当前只是共享 bearer admission control，因此按 deployment 共享 principal 处理。
- auth none：按本地 deployment 的 anonymous principal 处理。

当前 simple mode 的 X-User-Id 不是可靠的逐用户授权边界。真正的多用户隔离需要 JWT 或未来明确实现的 trusted-gateway identity；本功能不在 conversation 层伪造身份保证。

### 5.2 conversation_id

conversation_id 取代容易误解的 client_session_id：

- 浏览器生成 UUID 并保存在 localStorage。
- 它不是身份、凭据或 secret。
- 它只表示同一 principal 的哪段聊天。
- Clear History 删除当前 conversation 内容并生成新的 conversation_id。

所有 turn/image 的 select、update、delete 都必须同时过滤：

~~~text
(principal_id, conversation_id)
~~~

同一 principal 可以拥有多个 conversation。两个 principal 即使提交相同 conversation_id，也不能互相访问。

### 5.3 workspace

workspace 不属于 conversation identity。

- 每轮保存实际查询的 workspace 集合作为 provenance。
- 每个请求仍重新执行 workspace access control。
- 切换 workspace 不分叉历史，也不会让旧授权自动延续。

## 6. Web persistence

采用 PostgreSQL 独立表，不复用 workspace ingest 表。

### 6.1 web_conversations

- principal_id
- conversation_id
- created_at
- updated_at

主键为 principal_id + conversation_id。

### 6.2 web_conversation_turns

- turn_id
- principal_id
- conversation_id
- turn_number
- user_text
- assistant_text
- queried_workspaces
- created_at

唯一约束为 principal_id + conversation_id + turn_number。

### 6.3 web_conversation_images

- image_id：不可猜测 UUID
- principal_id
- conversation_id
- turn_id
- ordinal
- mime_type
- image_bytes：BYTEA
- byte_size
- content_sha256
- vlm_description
- created_at

图片保存为通过输入验证的规范化二进制，不把 data URI 写进 history JSON。

### 6.4 retention 与事务

- 默认保留最近 100 个完整 user/assistant turn，而不是 100 条 message。
- 默认 inactivity TTL 为 30 天。
- 删除 turn 时级联删除对应 conversation images。
- 成功 answer 后在一个事务中保存 user turn、assistant turn 内容和当前图片。
- 模型生成失败不保存半个 turn。
- persistence 失败不能伪装成已保存；Web done event 明确报告 conversation_saved=false。

这些表与 ingest storage 完全独立。

## 7. 公共接口

### 7.1 REST、MCP、Python SDK

Retrieve 和 Answer 公共 contract 删除：

- conversation_history
- session_id
- referenced_image_ids

公共接口只接受当前独立请求：

- query
- 当前请求的 query images / direct multimodal retrieval input
- workspaces
- top_k / chunk_top_k
- filters
- 其他现有无状态参数

公共 query images 永远表示当前请求图片，不产生 durable image ID。

### 7.2 Web route

/web/answer 接受：

- query
- images
- conversation_id
- workspaces

它不接受 conversation_history 或 principal_id。

/web/history 的读取与删除同样使用服务端 principal + conversation_id。

### 7.3 内部 answer input

Core 内部可以接收一个 request-local PreparedAnswerTurn：

- current_query
- retrieval_query
- text_history
- materialized_query_images

它不包含 principal_id、conversation_id、历史图片 UUID 或 persistence handle。公共 RAGService.aanswer 不暴露 text_history 参数；公共调用构造空 history 的 PreparedAnswerTurn。

## 8. ConversationTurnResolver

### 8.1 职责

ConversationTurnResolver 是 Web-only 的一次 structured LLM completion，不是 agent。

它负责：

1. 结合服务端历史把当前 follow-up 改写成独立 retrieval query。
2. 从当前 principal/conversation 的图片 catalog 中选择相关历史图片。

它没有：

- tools
- loop
- autonomous memory
- framework dependency
- 关键词字典
- 正则引用规则
- 固定中文/英文 ordinal 解析器

### 8.2 输入

- 当前 query。
- 当前图片的稳定顺序与 VLM description；description 失败时保留 “current image N” 占位信息。
- token-bounded text history。
- 最近 100 个持久 turn 内的 scoped image catalog：
  - image_id
  - turn_number
  - ordinal
  - concise vlm_description
- allowed_history_image_count。

Resolver 不接收其他 principal 或 conversation 的 catalog。

### 8.3 输出

- retrieval_query
- selected_history_image_ids，按相关性排序

输出数量不能超过 allowed_history_image_count。

### 8.4 后端验证

LLM 输出不是授权：

- 每个 ID 必须重新按 principal_id + conversation_id 查询。
- 重复 ID 去重。
- 超出 allowed count 的尾部结果拒绝或截断，并记录 trace。
- 不存在或越 scope 的 ID 不会 materialize。

### 8.5 降级

Resolver 调用失败时：

- 使用当前 query 作为 retrieval query。
- 不猜测历史图片 ID。
- 保留服务端 text history 和历史图片 caption 作为 answer 文本上下文。
- 继续使用当前 turn 图片。
- Web/trace 标记 history_image_resolution=degraded。

## 9. 当前图片优先

当前 turn 图片不是 resolver 候选，而是显式用户输入。

固定数据流：

~~~text
1. 验证并规范化当前图片
2. 为当前图片生成 VLM description
3. 当前图片先预留 answer transport slots
4. remaining = effective_max_images - current_image_count
5. Resolver 最多选择 remaining 张历史图片
6. current + selected history 一起参与 query enhancement/direct visual retrieval
7. Retrieval/rerank 完成后，RAG 原图只填充剩余 slots
~~~

Resolver 不能删除、替换或降级当前图片。

如果当前图片数量超过 active answer model 的有效能力，Web/API 在检索和流式输出前返回明确验证错误。前端有效上传上限会提前防止正常用户进入该状态。

如果当前图片无法在既定最小质量下进入 transport byte budget，请求失败并指出对应图片；不能静默忽略显式用户输入。

## 10. Retrieval 与 VLM description

### 10.1 Retrieval 不受 answer 图片上限约束

RAGService.aretrieve 和 LightRAG aquery_data 继续只受正常 RAG 限制：

- top_k
- chunk_top_k
- direct_visual_top_k
- metadata filters
- LightRAG token budgets
- rerank

它们不知道 answer.max_images。

/retrieve 返回 client-safe chunk content 及 image_url/thumbnail_url，不发送 inline base64，因此不需要 answer transport count cap。

### 10.2 Visual description 保真

当前 LightRAG ingest 只有在 VLM analysis 成功且 description 非空时才建立 multimodal chunk，并把组合文本写入 chunk.content。

Answer 的不变量：

- 每个进入 answer context 的 visual chunk 都发送完整的已存储 content。
- 原图是否进入 prompt 不影响 description 是否保留。
- “完整”指已入库 content；极端 VLM 输出可能已在 ingest token budget 下截断。
- 普通 answer context_top_k 和模型文本 context window 仍然适用。

异常或旧数据若出现 image_data 非空但 content 为空：

- 有 raw slot 时可发送原图。
- 无 raw slot 时丢弃该 chunk，避免 citation 指向模型从未看到的内容。
- 记录 malformed_visual_chunk trace。

本设计不为这种异常数据增加 runtime VLM 重分析系统。

## 11. 图片预算

三个预算必须独立命名和实现。

### 11.1 当前请求输入安全

query_images.max_current_images：

- 默认 3。
- Web、REST、MCP 和 Python 公共验证使用同一配置。
- 前端不再硬编码 MAX_IMAGES=3。

每图输入 bytes、mime 和 decode 验证继续独立配置。

### 11.2 Retrieval 候选

继续使用现有 top_k/chunk_top_k/direct_visual_top_k，不新增图片 count cap。

### 11.3 Answer image transport

answer.max_images：

- 默认 10。
- 表示最终 answer 模型 messages 中全部 image blocks 的产品 ceiling。
- current、selected history、RAG 共用一个计数器。
- 不是 gallery 限制，也不是 retrieval 限制。

answer.image_max_bytes、answer.image_max_total_bytes、max/min pixels、quality floor 继续控制 bytes 和单图质量。

有效 count：

~~~text
effective_max_images =
  min(answer.max_images, probed_answer_model_max_images)
~~~

分配顺序：

1. 当前 turn 图片。
2. Resolver 选中的历史图片。
3. 现有 rerank 顺序中的 RAG visual chunks。

历史和 RAG 原图未获 slot 时都有文字 description fallback。当前图片没有静默 fallback。

### 11.4 配置清理

删除：

- answer.max_user_images
- query_images.max_described_images
- query_images.session_max_images
- query_images.session_max_sessions
- checkpoint/session-image 专属配置

新增或重定义：

- answer.max_images：默认 10，全部最终原图 blocks。
- query_images.max_current_images：默认 3；当前图片都应被描述。
- web_conversations.max_turns：默认 100。
- web_conversations.ttl_days：默认 30。

不保留旧配置 alias。

## 12. Eager answer vision capability

### 12.1 探测对象

探测 AnswerEngine 实际使用的 model_for_role(config, "query")，而不是无条件探测 llm.default。

VLM role、rerank model 和 multimodal embedding 有各自职责，不能用 answer capability 代替。

### 12.2 能力状态

Manager 保存一个 request-independent AnswerImageCapability：

- status：supported / unsupported / unknown
- configured_ceiling
- effective_max_images
- provider
- base_url
- model
- failure_kind

这是真正的三态：

- supported：模型成功接受 image_url blocks。
- unsupported：provider 明确返回该模型不支持图片。
- unknown：timeout、401、429、5xx、服务未启动或不可分类错误。

Probe 成功标准是 transport 接受图片请求，不强制模型回复包含特定单词。

### 12.3 启动流程

~~~text
answer.max_images == 0
  -> 不调用模型；status=unsupported；failure_kind=config_disabled；effective=0

发送 1 张极小图片
  -> 明确图片能力错误：unsupported，effective=0，停止
  -> transient/unclassified：unknown，effective=0，停止
  -> 成功：继续 count probe

发送 configured ceiling 张极小图片
  -> 成功：effective=configured ceiling
  -> provider 明确返回最大 N：effective=min(N, ceiling)
  -> 确定性 count 拒绝但没有 N：在 [1, ceiling] 有界二分
  -> transient/unclassified：保持 supported，effective=1，
     failure_kind=count_probe_unknown
~~~

默认 ceiling=10 时最坏约 5 次极小 startup request，通常 2 次。

能力不写入持久表；每个进程启动重新验证，避免 model/endpoint 变化后留下 stale capability。

### 12.4 UI 与后端

Web 当前轮上传上限：

~~~text
effective_current_upload_limit =
  min(query_images.max_current_images, capability.effective_max_images)
~~~

- supported：启用加号、file input、paste 和 drag/drop-to-composer。
- unsupported：禁用这些入口，解释 active answer model 不支持图片。
- unknown：暂时禁用，解释能力检查失败；不能谎称模型不支持。

Gate 必须位于统一 addImage 入口，不能只隐藏加号。

后端始终执行同样校验；前端 capability 只是服务端事实的 UX 投影。

unsupported/unknown 不阻塞纯文本 answer。此时 AnswerPromptAssembler 使用
effective=0，不发送 RAG 原图，但继续发送已存储 visual descriptions。

禁用 Web answer 上传不影响：

- /retrieve 的 image query 能力
- ingest-time VLM
- multimodal embedding
- 已入库 visual chunk descriptions
- gallery

### 12.5 运行时防漂移

Eager probe 是主要能力来源。若 provider routing 在进程运行期间变化，answer 请求仍保留一次防御性处理：

- 仅在零可见 token 时接受明确 capacity error。
- 更新当前进程 capability。
- 重新组装 prompt 并重试一次。
- 无法分类或再次失败则返回结构化错误。

它是 drift guard，不是延迟到首个用户请求才学习能力。

## 13. Answer prompt 编排

AnswerPromptAssembler 是唯一创建最终 image blocks 的组件。

输入：

- text-only history messages
- current query
- current images
- selected history images
- reranked RAG contexts
- AnswerImageCapability

规则：

- 持久历史不会恢复为带 image_url 的旧 user messages。
- selected history 原图只在最终 current user content 中出现一次，并带 turn/ordinal label。
- 所有 image blocks 通过同一个 ImagePayloadBudget 和现有 bounded image compression utility。
- Trace 的 total image count 必须等于 provider 实际接收的 block 数。
- Citation index 只基于最终模型实际看到的 text description 或 raw image。

不新增 provider-specific prompt planner。

## 14. Streaming 与错误

Web SSE 在发送任何 token、preview 或最终 answer 前：

1. 完成 conversation resolution。
2. 完成 retrieval/rerank。
3. 完成最终 prompt 编排。
4. 打开 upstream stream。
5. 预取第一个可见 token。

只有在第一个可见 token 前允许 capacity drift retry。

错误分类：

- CURRENT_IMAGES_UNSUPPORTED：active answer model 明确不支持。
- CURRENT_IMAGE_LIMIT_EXCEEDED：当前图片超过有效上限。
- CURRENT_IMAGE_PAYLOAD_INVALID：decode/mime/bytes/quality 失败。
- ANSWER_IMAGE_CAPABILITY_UNKNOWN：能力探测未完成或失败。
- HISTORY_IMAGE_RESOLUTION_DEGRADED：resolver 失败但可 text-only 继续。
- ANSWER_PROVIDER_CAPACITY_CHANGED：运行时能力漂移且重试失败。

成功 answer 才保存 turn。错误响应不创建半个 checkpoint。

## 15. Gallery

Gallery 来自 final cited sources：

- 对 cited visual chunks 去重。
- 使用 authenticated image_url/thumbnail_url。
- 不按 answer.max_images 截断。
- 不要求原图曾直接发送给 answer model；模型看到该 visual chunk 的已存储 VLM description 即可合法引用。
- Gallery 的虚拟滚动、分页或缩略图缓存属于性能实现，不是内容数量限制。

## 16. 删除范围

直接删除：

- dlightrag_checkpoints 表及 PGCheckpointStore。
- SessionImageStore。
- dlightrag-image:// 内部引用协议。
- RAGServiceManager checkpoint/session-image CRUD。
- 公共 request 中的 conversation_history/session_id/referenced_image_ids。
- Web request 中的 conversation_history。
- frontend conversationStore 作为 prompt 事实源的逻辑。
- QueryPlanner 输出 referenced_image_ids 的 schema/prompt。
- 旧 checkpoint 配置、events、tests 和文档。
- answer user/RAG 双预算与 answer.max_user_images。

保留或改造：

- frontend history store 仅作为服务端 history 的 UI projection；不参与请求 prompt。
- QueryImageEnhancer 的 current-image description 能力。
- 现有 image compression utility。
- 现有 retrieval、rerank、citation 和 visual asset URL 能力。

## 17. 保护区

本功能不得改变以下产品语义：

- RAGService.aretrieve。
- core/retrieval/**。
- LightRAGBackend 对 aquery_data/QueryParam 的使用。
- BM25/RRF/rerank 算法。
- ingestion pipeline。
- LightRAG visual chunk content。
- workspace ingest storage。
- public retrieve response shape，除删除会话型 request 字段外。

若实现过程中发现必须修改这些边界，停止并重新 review 设计，不能顺手扩大 refactor。

## 18. Observability

每次 Web answer 至少记录：

- principal/conversation 使用不可逆 hash，不记录原始值。
- history_turns_loaded。
- history_image_catalog_count。
- history_images_selected。
- history_image_resolution_status。
- answer_image_capability_status。
- answer_image_configured_ceiling。
- answer_image_effective_limit。
- answer_images_current/history/rag。
- answer_images_total。
- answer_image_bytes_total。
- rag_visual_descriptions_included。
- rag_raw_images_skipped。
- capacity_probe/retry outcome。

answer_images_total 必须从最终 messages 计算或与最终 messages 交叉验证，不能继续只统计 RAG budget。

## 19. 验收测试

### 19.1 公共 contract

- REST/MCP 对 conversation_history、session_id、referenced_image_ids 返回 extra-field validation error。
- Python 公共 answer/retrieve signature 不含这些参数。
- 当前 request query images 仍可用于无状态 retrieve/answer。

### 19.2 会话 scope

- 同 principal + conversation 恢复历史。
- 相同 conversation UUID、不同 principal 互不可见。
- 切换 workspace 后仍是同一 conversation。
- 每轮 workspace 仍单独鉴权并记录。
- Clear History 删除旧 conversation 并生成新的 conversation_id。

### 19.3 retention

- 保存并恢复最近 100 个完整 turn。
- 第 101 个 turn 写入后删除最旧完整 turn 及其图片。
- TTL prune 只删除 Web conversation tables。
- destructive migration 不影响任何 ingest 表和记录。

### 19.4 resolver

- 输入 catalog 只含当前 principal/conversation 图片。
- 当前图片不属于 selected_history_image_ids。
- selected history 数量不超过 remaining slots。
- 越 scope、伪造和重复 ID 不 materialize。
- Resolver failure 使用 current query/current images/text history 继续并标 degraded。

### 19.5 retrieval 与 description

- answer.max_images=0 不改变 /retrieve 的 visual result count。
- RAG 原图无 slot 时完整已存储 content 仍进入 answer prompt。
- malformed empty-content visual chunk 无 slot 时被丢弃且不可引用。

### 19.6 transport budget

- current + history + RAG 的最终 image block 总数不超过 effective limit。
- 默认 product ceiling 为 10。
- 默认 current turn 上传为 3，且 Web/backend 同源。
- 当前图片先占 slot；history 次之；RAG 填余量。
- 当前图片不能压缩到最低质量时请求明确失败。

### 19.7 capability

- Probe 使用 query role override 而不是错误的 default role。
- 一张图明确 unsupported 后不执行 count probe。
- 10 张成功时 effective=10。
- 明确返回 max=5 时 effective=5。
- 无数字的确定性 count rejection 通过有界二分得到最大值。
- timeout/401/429/5xx 产生 unknown，不产生 unsupported。
- 单图成功但 count probe 瞬时失败时保持 supported，并安全降级为 effective=1。
- supported/unsupported/unknown 正确驱动统一 addImage gate 和后端校验。

### 19.8 streaming

- capacity error 在首个 visible token 前可重试一次。
- 首 token 后不重试。
- 失败请求不保存 turn。
- 成功请求保存完整 turn 和图片。

## 20. 被否决的替代方案

### 20.1 调用方自带 history

否决。它让 REST/MCP/SDK 携带更长、可伪造且与服务端状态冲突的 prompt，并把 Web 产品职责泄漏给所有调用方。

### 20.2 Conversation-aware RAG core

否决。Retrieval 不需要 session、principal、history image UUID 或 persistence。把这些加入 core 会侵入所有上游产品路径。

### 20.3 通用图片 agent 或 batch tree

否决。当前痛点只需要 Web history reference resolution 和最终 transport budgeting。RAG visual chunks 已有 VLM description，没有证据支持开发复杂多阶段视觉 agent。

### 20.4 固定语言规则

否决。图片指代属于 LLM-driven language understanding；后端只做 scope、数量和 schema 验证。

### 20.5 把 gallery 限制为模型图片上限

否决。Gallery 是 cited-source UX，answer transport 是 provider constraint，两者不是同一预算。

## 21. 高层实施顺序

1. 写 contract tests，锁定公共无状态接口和保护区。
2. 引入新 Web conversation tables/store/service。
3. 切换 Web history GET/DELETE 和 answer request。
4. 引入 ConversationTurnResolver 和 scoped image materialization。
5. 将 answer 双预算替换为一个 AnswerPromptAssembler。
6. 升级 eager query-role capability probe 和前端 gate。
7. 增加 stream first-token handshake 和 capacity drift guard。
8. 删除旧 checkpoint/session/history 路径、配置、测试和文档。
9. 执行 destructive conversation reset。
10. 完整验证 public contracts、scope、retrieval invariants、streaming 和 UI。

该顺序不是 implementation plan；用户完成书面 spec review 后，再单独编写逐文件、逐测试的实施计划。
