# ADR 0076 — the Agentspace's language layer says Agentspace, and only the protocol keeps workspace

**Summary / 摘要**

[ADR 0075](0075-workspace-names-the-corpus-and-agentspace-names-the-agents-files.md) split the reader's
word: **Workspace** (工作区) is the corpus scope, **Agentspace** is the agent's files. It kept the code
word `workspace` for PostgreSQL names, HTTP routes, wire fields, prompts — and, in the same paragraph,
for Python and TypeScript class and module names. That last item does not belong with the others: a
table name is expensive to change, a class name is free. The residue showed up at once: two
`WorkspaceChange` types lived side by side under `frontend/api/` with different meanings, corpus and
agentspace stores differed by one word in the same adapter layer, and a reader inside
`application/web_conversations` had to remember that `WorkspaceSnapshot` was not the corpus.

This ADR renames the **language layer** only. Python classes, dataclasses, type aliases, functions,
methods and module file names in agentspace-owned code say **Agentspace**; corpus/RAG code keeps bare
**Workspace** (it is the corpus's word). Everything a client, deployment or model can see stays exactly
as ADR 0075 left it: tables and columns, `/web/api/conversations/{id}/workspace/...` routes, wire JSON
keys, config keys, error and health identifiers, the operator CLI, and the model-facing prompt text. A
structural test on each side (Python AST, Web source scan) keeps the two vocabularies apart.

ADR 0075 把读者的词分开了：**Workspace**（工作区）指语料范围，**Agentspace** 指 agent 的文件。它同时把
`workspace` 保留给 PostgreSQL 名、HTTP 路由、wire 字段、prompt——以及同一段里的 Python/TypeScript 类名与
模块名。最后这一项和其余不同：表名改动昂贵，类名改动免费。残留立刻显形：`frontend/api/` 下两个
`WorkspaceChange` 同名异义，同一适配层里语料与 agentspace 的 store 只差一个词，`application/web_conversations`
的读者得记住 `WorkspaceSnapshot` 不是语料。本 ADR 只改**语言层**：agentspace 一侧的类、dataclass、类型别名、
函数、方法、模块文件名说 **Agentspace**；语料/RAG 代码保留裸 **Workspace**（那本来就是语料的词）。客户端、
部署与模型能看到的一切都不动：表与列、`/web/api/conversations/{id}/workspace/...` 路由、wire JSON 键、配置键、
错误与健康标识、运维 CLI、以及面向模型的 prompt 文本。两侧各一条结构测试（Python AST、Web 源码扫描）把两套
词汇隔开。

## Status

Accepted 2026-10-25; implemented in this change. It amends ADR 0075 Decision 4 (which had listed class
and module names under "the code keeps workspace") and supersedes ADR 0075's "Code identifiers"
deferral for the language layer only. ADR 0075's protocol mapping, its prompt deferral and its
reader-surface decision stand unchanged.

## Context

ADR 0075 decided the reader's vocabulary and mapped the code word onto it: with the corpus it is the
corpus scope; with a Run, a conversation, an epoch or a snapshot it is the Agentspace. That mapping is
enough for names that cannot move (a table, a route, a wire field, a prompt line), but it is a poor
answer for names that cost nothing to change.

Three residues made the cost concrete:

- **One name, two meanings, one directory.** `frontend/api/workspaces.ts` exported `WorkspaceChange` as
  the corpus mutation kinds (`ingest`, `replace`, `delete`, `retry`, `reset`, `delete_workspace`) while
  `frontend/api/conversations.ts` exported `WorkspaceChange` as the per-file change of a saved copy
  (`added`, `changed`, `unchanged`). Only the import path told them apart, and the second one was added
  by the Inspector diff work that ADR 0075's own surface produced.
- **One word, two stores, one layer.** `PGWorkspaceStore` (the Agent's files) sat in
  `adapters/postgres/answer/workspace.py` beside `PGWorkspaceRegistry` (the corpus) in
  `adapters/postgres/corpus/workspaces.py`; `_compose.py` wires both, so a reader meets them on one
  page.
- **One grep, two meanings, one operator.** `workspace_integrity_error` is an agentspace Run error and
  `workspace_not_found` is a corpus mutation outcome; a log search for `workspace` finds both panels'
  failures. The error ids are protocol and stay; the surrounding code should not add to the pile.

None of this ever confused the machine: modules, tables and routes are distinct namespaces. It confused
people — the only namespace that matters for a product whose ADR records are the vocabulary authority.
And it is a class of ambiguity that grows with every new module, since nothing stops a new file from
choosing either word.

The protocol names, by contrast, do have costs that make the mapping the right answer for them: a table
or column rename is a migration, a route or wire-field rename is a client-visible change, and prompt
text is prefix-cached (ADR 0015). Keeping `workspace` there is a decision with a price and a benefit;
keeping it in a class name was a default nobody priced.

## Decision

**1. The language layer of agentspace-owned code says Agentspace.** A declared class, dataclass, type
alias, function, method, or module file name that means the agent's files uses the product noun:
`Agentspace*`, `*agentspace*`. Examples of the rename (not exhaustive):

| Old | New |
|---|---|
| `ConversationWorkspace` (application model, view and Web type) | `ConversationAgentspace` |
| `WorkspaceHead`, `WorkspaceSnapshot`, `WorkspaceSnapshotChanges`, `WorkspaceFile`, `WorkspaceFileVersion` | `AgentspaceHead`, `AgentspaceSnapshot`, `AgentspaceSnapshotChanges`, `AgentspaceFile`, `AgentspaceFileVersion` |
| `WorkspaceChange` (the saved copy's `added`/`changed`/`unchanged`) | `AgentspaceChange` |
| `PGWorkspaceStore` (agentspace) | `PGAgentspaceStore` |
| `WorkspaceUnavailable`, `WorkspaceIntegrityError`, `RunWorkspace`, `WorkspaceSeed`, `StagedWorkspaceHead` | `AgentspaceUnavailable`, `AgentspaceIntegrityError`, `RunAgentspace`, `AgentspaceSeed`, `StagedAgentspaceHead` |
| `WorkspaceAccess`, `WorkspaceConfinement`, `WorkspaceQuotaExceeded`, `WorkspacePathFact`, `WorkspaceInventoryFacts`, `encode_workspace_text`/`decode_workspace_text`, `replace_workspace_file` (the Execution Environment's own vocabulary) | `AgentspaceAccess`, `AgentspaceConfinement`, `AgentspaceQuotaExceeded`, `AgentspacePathFact`, `AgentspaceInventoryFacts`, `encode_agentspace_text`/`decode_agentspace_text`, `replace_agentspace_file` |
| `workspace()`, `workspace_snapshot()`, `reset_workspace()` on the conversation service | `agentspace()`, `agentspace_snapshot()`, `reset_agentspace()` |
| `readConversationWorkspace()`, `readWorkspaceSnapshot()`, `readWorkspaceFileHistory()`, `resetConversationWorkspace()` | `readConversationAgentspace()`, `readAgentspaceSnapshot()`, `readAgentspaceFileHistory()`, `resetConversationAgentspace()` |
| Module `engine/answer/workspace.py`, `engine/runtime/workspace.py`, `adapters/postgres/answer/workspace.py` | `engine/answer/agentspace.py`, `engine/runtime/agentspace.py`, `adapters/postgres/answer/agentspace.py` |

The renames are declarations only. Tool results, prompt text, Skill documents, configuration keys, log
prose that quotes a protocol id, and constants named after a configuration key are text and protocol,
not declarations: they keep the spelling ADR 0075 mapped.

The Execution Environment's rooted-directory vocabulary belongs to this layer too: its confinement,
access, quota, inventory-facts and text-codec classes name the Agentspace, so they rename with it. What
they configure — `WORKSPACE_MAX_BYTES`, environment and configuration keys — keeps the protocol's
spelling, and the difference is the mapping, not an inconsistency.

**2. Corpus and RAG code keeps bare Workspace.** It is the corpus's own word (ADR 0075 Decision 1):
`application/corpus_admin/**`, `adapters/postgres/corpus/**`, `adapters/http/rest/**` workspaces,
`adapters/mcp/**` workspace selection, `application/access/workspaces.py`, `engine/rag/workspace/**`,
`adapters/http/browser/workspace_models.py`, `adapters/http/browser/routes/workspaces.py`, and the Web's
`api/workspaces.ts`, `stores/workspace-store.ts`, `ui/workspace-scope.ts`, `ui/workspace-create.ts`.
Its `WorkspaceChange` (mutation kinds), `WorkspaceCatalog*`, `PGWorkspaceRegistry` and `WorkspaceNotFoundError`
are correct names and do not move.

**3. The protocol does not move.** PostgreSQL tables and columns (`dlightrag_session_workspace_*`,
`agent_workspace_epoch`, `dlightrag_answer_workspace_inventory`), HTTP routes
(`/web/api/conversations/{id}/workspace`, `/workspace/snapshots/{run_id}`, `/workspace/files`,
`/workspace/reset`), wire JSON keys (`heads`, `snapshots`, `files`, `change`, `changes`), configuration
keys (`answer.agent.workspace_root`, the workspace bounds), error and health identifiers
(`agent_workspace`, `workspace_integrity_error`, `workspace_not_found`), the operator CLI
`dlightrag-workspace-audit`, and the model-facing prompt/Skill text keep `workspace` under ADR 0075's
mapping. No migration, no client-visible change, no prompt-cache change.

**4. Code that is neither declaration nor protocol may keep either word, but must not blur them.** Local
variables, dataclass fields that mirror a wire or column name, SQL string literals, JSON keys, log
messages that quote a protocol id, and comments that quote a historical ADR stay as written. A file
that names both concepts gives the corpus one its qualifier (or lives in a corpus module).

**5. One structural test per side keeps the vocabularies apart.**
`tests/unit/test_agentspace_vocabulary.py` parses the Python sources, classifies modules by path and
fails when an agentspace-owned module declares a `Workspace*` name, when a corpus-owned module declares
an `Agentspace*` name, or when a renamed module file reappears under its old path.
`frontend/agentspace.structure.test.ts` scans the Web sources and fails when the agentspace modules
(`api/conversations.ts`, `ui/inspector-agentspace.ts`) declare a `Workspace*` name, when the corpus
modules declare an `Agentspace*` name, or when two modules under `frontend/api/` export the same name —
the collision class that started this.

## Alternatives considered

- **Keep ADR 0075's deferral (status quo).** Cheapest today; every review and every new module pays the
  mapping instead, and the residue already produced a duplicate type and two same-prefix stores.
- **Rename the corpus side to `CorpusWorkspace*` instead.** The corpus's reader word is Workspace, its
  identifiers are embedded in Access Rules, REST and MCP contracts, and it is the older and larger
  surface; the newer, smaller concept is the one that should take the new noun — the same argument ADR
  0075 used to leave the corpus alone.
- **Rename the protocol too, so code and wire share one word.** A schema migration, a client-visible
  route/wire change and a prompt-cache change for a naming benefit no reader sees. If the product ever
  wants it, it is its own ADR and its own release.
- **Suffix instead of prefix (`*Agentspace`).** A suffix reads as a description (`RunWorkspaceAgentspace`)
  and scatters related names alphabetically; the noun-first form matches the product and sorts the
  space's own names together.
- **Only add the naming tests, without the rename.** A test cannot separate two vocabularies that are
  both spelled the same; it would only lock whatever each file already says.
- **Rename local variables, comments and mirrored field names too.** Large, unreviewable, and it would
  hide the protocol mapping the code actually follows; the declaration layer is where a reader picks
  the word and where the ambiguity was.

## Consequences

- A reader can tell the two spaces from any declaration alone: `Agentspace*` is the agent's files,
  bare `Workspace*` is the corpus. The ADR 0075 mapping is now needed only for protocol names and for
  historical records.
- Zero migration, zero wire change, zero client change, and the model still reads `workspace` in its
  instructions (ADR 0075's prompt deferral stands). The model's vocabulary and the reader's remain
  deliberately different until that prompt-version change.
- The renames are mechanical, so the diff is large but the behavior is not: test suites, type checks,
  lint, import contracts and the deployed smoke checks are the proof.
- Two structural tests now own the rule, so the next module cannot quietly reintroduce the collision;
  a contributor who needs the corpus word in an agentspace module must move the code to a corpus module
  instead.
- Test file names are kept for history even where they say workspace (`test_answer_workspace.py`); they
  name a file, not a declaration.
- The Execution Environment's classes say Agentspace while its configuration constants keep
  `WORKSPACE_*`; a reader who wants the space reads the class, and a reader who wants the setting reads
  the key.
- Historical ADRs keep "Agent Workspace"; a search for the current term still has to search the old one.
  `docs/trust/retention.md`'s pointer to the tree-retention constant follows the module rename.

## Deferred

- **Model-facing prompt and Skill text.** Still says workspace; renaming changes every Run's cached
  prompt prefix and belongs to a prompt-version change (ADR 0015). The protocol mapping in ADR 0075
  Decision 4 is the reader's guide until then.
- **Database, routes, wire fields, config keys, error/health ids, operator CLI.** Client- and
  deployment-visible; any rename needs a migration plan and its own ADR.
- **Local variables, comments, mirrored dataclass fields and log prose.** They may keep `workspace`
  where they quote a protocol name or a historical record; a later cosmetic pass is optional and is not
  enforced by the structural tests.
