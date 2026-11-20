# ADR 0075 — Workspace names the corpus, and Agentspace names the agent's files

**Summary / 摘要**

Two different things were called "Workspace" on reader-facing surfaces: the corpus scope a reader
administers (工作区; internal **Corpus Workspace**) and the model-visible files the agent works in
(formerly **Agent Workspace**). This ADR separates them for good. **Workspace** names the corpus
scope and nothing else. **Agentspace** — one word, capital A — names the agent's durable file space
for one conversation: the model-visible files its Runs work in, carried from one Run to the next,
with each settled Run leaving a restorable snapshot of it. The reader surface renames with the word:
the topbar button and the Inspector panel's title and copy become Agentspace, the agent-side
"Reset workspace" becomes "Reset Agentspace", and both locales keep the English word, as they
already do for Run. The code keeps `workspace` in this change — PostgreSQL tables and columns, HTTP
routes, Python and TypeScript names, and the model-facing prompt text — under the mapping recorded
below; the prompt text is deferred rather than mapped, because prompt prefixes are
cache-anchored (ADR 0015).  
两个此前都叫 "Workspace" 的东西是分开的：读者管理的语料范围（工作区；内部称 **Corpus Workspace**），
和 agent 工作的模型可见文件（旧称 **Agent Workspace**）。本 ADR 将它们彻底分开：**Workspace** 只指语料
范围；**Agentspace**（一个词，大写 A）指 agent 在一个会话中的持久文件空间——它的 Run 所工作的模型可见
文件，跨 Run 延续，每个已结算 Run 留下可恢复的快照。读者界面随词而改：顶栏按钮、Inspector 面板标题与
说明改为 Agentspace，agent 侧的 "Reset workspace" 改为 "Reset Agentspace"；两种语言都保留英文词，与
Run 的做法一致。代码在本变更中保留 `workspace`——PostgreSQL 表与列、HTTP 路由、Python 与 TypeScript
命名、以及面向模型的提示词——映射记录在下；其中提示词因前缀受缓存锚定（ADR 0015）而暂缓。

## Status

Accepted 2026-10-25; the reader surface lands in this change, in both locales. The docs rename ships
with it: Domain Language's entries, CONTEXT.md, and the living docs use Agentspace for the agent's
files and leave Workspace to the corpus. Code identifiers and model-facing prompt text keep the old
word under the mapping below.

## Context

The product had one word for two things, and both were reader-facing.

**The corpus scope.** "Workspace" (工作区) is the named, authorized unit of documents a deployment
administers: the Files inspector resets and deletes workspaces, Workspace administration creates them,
Access Rules map JWT claims to workspace patterns, and an accepted Retrieval or Answer Run pins the
authorized Workspace set it ran on. Domain Language names it **Corpus Workspace** and defines it as
one corpus runtime's named, authorized scope.

**The agent's files.** The model-visible files one conversation's agents write with rooted tools
were "Agent Workspace" — and in the Chinese locale also 工作区, because there was no second word
either. The Web surfaced them under the corpus's label: the topbar button said Workspace and opened
an Inspector panel titled Workspace, and that panel's "Reset workspace" sat one step from the Files
panel's "Delete workspace". A reader had to infer from the panel which thing a control acted on, and
the two actions have different blast radii: resetting the agent's files keeps the conversation and
its answers; deleting a workspace also deletes its corpus data.

**The agent's files are no longer a scratch tree.** Each settled Run commits a snapshot of the
space, the Inspector lists those snapshots and their branches, and a later Run, Follow-Up or Fork
continues from them. "Workspace" described a temporary working area; this is the conversation's
carried file space, and "Agent Workspace" shortened to "workspace" in every sentence that used it.

**The same word reached the data-protection documents.** The retention schedule said the Run clock
bounds the Workspace Inventory and that a tree is removed seven days after its Run finishes; the data
map called the disk plane the Agent Workspace. A DPA drafted from those pages would use one word for
the corpus and the agent's files.

**The cost grows with every surface.** The agent's snapshots belong to the conversation Inspector;
the corpus belongs to Administration and Access Rules; both keep growing. Every new label that says
"workspace" has to be read in context and can be read wrong. The owner approved the split on
2026-10-25 and named the agent's space **Agentspace**.

## Decision

**1. Workspace names the corpus scope, always.** On every reader-facing surface and in every living
document, "Workspace" (工作区) means the authorized corpus scope — Domain Language's **Corpus
Workspace** — and nothing else. Corpus-side copy keeps its word unchanged: Workspace actions, Delete
workspace, Corpus Reset, Workspace Delete, Workspace Creator, and the authorized Workspace set.

**2. Agentspace names the agent's durable file space.** **Agentspace** is the model-visible files one
conversation's agent works in, carried from one Run to the next; each settled Run leaves a restorable
snapshot of it. It is one word, capital A, in every position of a sentence, including Chinese copy,
where the English word is kept as Run already is; only a code identifier such as `app.agentspace` is
lowercased by code convention. It is never written "Agent Workspace", "Agent workspace", "agent
workspace", "Agent Space" or "agent-space".

**3. The reader surface renames with the word.**

| Surface | Before | After |
|---|---|---|
| Topbar button | Workspace / 工作区 | Agentspace |
| Inspector panel title, tab and copy | Workspace / 工作区 ("its workspace") | Agentspace ("its Agentspace") |
| Agent-side reset action | Reset workspace / 重置工作区 | Reset Agentspace |
| Web component and module that own the surface | `inspector-workspace.ts`, `DlInspectorWorkspace`, `app.workspace` | `inspector-agentspace.ts`, `DlInspectorAgentspace`, `app.agentspace` |
| Corpus-side copy (Files inspector, Workspace administration) | Workspace / 工作区 | unchanged |

The Web copies both locales from this map, and the Chinese locale keeps the English word, so the
product names the space the same way in both.

**4. The code keeps `workspace` in this change, recorded so the name is never ambiguous.**

- **PostgreSQL tables and columns**: `dlightrag_session_workspace_heads`,
  `dlightrag_session_workspace_snapshots`, `dlightrag_answer_workspace_inventory`, and the
  `answer.agent.workspace_root` configuration key. Table and column names do not change.
- **HTTP routes**: `/web/api/conversations/{conversation_id}/workspace`,
  `/workspace/snapshots/{run_id}`, `/workspace/files` and `/workspace/reset`, with the wire fields
  the Inspector already reads (`heads`, `snapshots`, `files`, `versions`, `change`, `changes`).
- **Python and TypeScript class and module names**: `ConversationWorkspace`, `WorkspaceSnapshot`,
  `WorkspaceFile`, `WorkspaceFileVersion`, `resetConversationWorkspace`, `workspace-store.ts`,
  `engine/answer/workspace.py`, and their peers. The reader-surface component and module are the
  exception in 3: they name the surface, so they rename with it.
- **Error and health identifiers**: `agent_workspace`, `workspace_integrity_error`,
  `workspace_not_found`. A reader-facing message may say Agentspace while the identifier that carries
  it keeps the code word.
- **Model-facing prompt text**: the prompts and Tool descriptions the model reads say workspace (and
  "working directory") for these files; this ADR does not rewrite them. Prompt prefixes are
  cache-anchored, so a rename changes the cached prefix of every Run and belongs to a prompt-version
  change (ADR 0015), not a naming one.

A reader who meets `workspace` in code, configuration, a route or a prompt applies this mapping: with
the corpus it is the corpus scope; with a Run, a conversation, an epoch or a snapshot it is the
Agentspace.

**5. The avoid list.**

- For the agent's files, avoid **Agent Workspace**, **agent workspace**, **working directory**,
  **working dir**, **sandbox**, and bare **workspace** on any reader-facing surface.
- For the corpus, never write **Agentspace** or **agentspace**.
- "Agentspace" is one word with a capital A; "Agent Space" and "agent-space" are always wrong.

## Alternatives considered

- **Keep "Agent Workspace", qualified with a capital.** It still contains the overloaded noun, so
  every shortened sentence ("its workspace", "the workspace quota") keeps the old ambiguity, and two
  words with a space read as a description rather than a product name.
- **"Run files".** Wrong grain: the files are carried from Run to Run and a settled Run leaves a
  snapshot of the conversation's space, so "Run files" says the opposite of where they live and
  suggests they are products of one Run rather than the space it worked in.
- **"Sandbox".** The product already has a security boundary word — the Execution Environment's
  `trust` mode is not called a sandbox (ADR 0024) — and a sandbox names isolation, not a durable,
  restorable place where the agent's work lives. Reusing it would claim a guarantee the space
  does not make.
- **"Agent folder".** Understates a filesystem with branches, snapshots and file history, and
  collides with the Files inspector's folder vocabulary, where "folder" already means a corpus
  directory.
- **Rename the corpus instead, and give the agent "Workspace".** Rejected: corpus Workspace is
  embedded in Access Rules, the public REST surface, the Files inspector, and every translated
  reader string, and it is the accurate word for a named scope of indexed data. Renaming that side
  would be a client-visible migration to fix a collision the agent side can leave by taking a new
  name.
- **Do nothing; let readers read the panel title.** The status quo, and the owner's answer was no:
  the two labels occurred in adjacent panels with different destructive actions, and the ambiguity
  already reached the trust documents.

## Consequences

- A reader can tell the two apart by the word alone: the topbar and the conversation Inspector say
  Agentspace; the Files inspector and every corpus control keep Workspace/工作区. The agent-side reset
  and the corpus-side delete no longer read as variants of one action.
- Both locales say Agentspace; Chinese keeps the English word, so the space is named identically to
  Run and does not acquire a Chinese coinage the product would have to keep consistent.
- The living docs and the Domain Language use the new term: **Agentspace**, **Agentspace Epoch** and
  **Agentspace Inventory** replace Agent Workspace, Workspace Epoch and Workspace Inventory. The
  retention schedule, data map, security, configuration and runtime pages are updated with them.
- Historical ADRs (0019, 0022, 0024, 0033, 0038, 0049, 0051, 0052 and others) keep "Agent
  Workspace": an ADR is a historical record and is not rewritten by a later name. This ADR and
  Domain Language are the current names; a search for the current term has to search the old one too
  to find the history.
- Code, configuration keys, HTTP routes and wire fields do not change, so no client, deployment or
  database migration is needed; a reader of code applies the mapping in Decision 4.
- The model still sees "workspace" in its instructions, so the model's vocabulary and the reader's
  vocabulary differ until the deferred prompt-version change. The mapping is recorded so that
  difference is deliberate and not a second meaning.
- An error code or health check named `agent_workspace` may be shown to an operator; operator copy
  may say Agentspace, but the identifier stays.
- "Agentspace" is now the term a new document must use; a new living document that says "Agent
  Workspace" for the space is stale, and `docs/domain-language.md` is where the naming authority
  lives.

## Deferred

- **Model-facing prompt text.** The system and plan prompts and the Tool descriptions the model reads
  still say workspace and "working directory"; renaming them changes the cached prompt prefix of every
  Run and is deferred to a prompt-version change under ADR 0015. Until then, the model's word for the
  space stays `workspace`, and Decision 4 is the mapping.
- **Code identifiers.** Table, column, route, wire-field and most class/module names keep `workspace`;
  renaming them is a schema and client-visible migration with no reader benefit, and would need its
  own ADR if the product ever wants the code and the reader to share one word.
- **Error and health identifiers.** `agent_workspace`, `workspace_integrity_error` and
  `workspace_not_found` stay as identifiers; translating them for reader copy is a per-surface choice,
  not a rename.
- **Historical records.** Existing ADRs and research notes keep the old
  term. They are not living documentation, and rewriting them would falsify what was decided when.
- **The snapshot's own product name.** This ADR names the space and states that each settled Run
  leaves a restorable snapshot, but does not name the snapshot or its change vocabulary
  (added/changed/removed). The Inspector's per-Run diff timeline is the change that makes those
  reader-facing labels real, and their naming belongs there.
