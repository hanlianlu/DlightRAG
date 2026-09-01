---
name: skill-creator
description: Guide users through creating, refining, and removing their personal DlightRAG skills. Use when the user wants to create a skill, turn a repeated workflow into a skill, or edit an existing one. Flow: interview → draft → publish_skill → verify.
---

# Skill Creator

为 DlightRAG 创建个人技能的引导流程。你负责访谈、起草与发布；用户负责确认。

## 适用时机

用户出现以下意图时加载本技能：

- "帮我建个技能" / "把刚才这套流程变成技能" / "我每天都要做 X，做个技能"
- 用户描述一个重复性任务，希望以后一句话就能触发

## 总流程

1. **访谈**：问清三个问题（见下），不确定就继续问
2. **起草**：生成 `SKILL.md` 与必要的附属文件
3. **确认**：把草案要点（名称、触发描述、文件清单）讲给用户，等明确同意
4. **发布**：调用 `publish_skill(name, files)` —— 这是唯一的持久化通道，发布本身就是"保存"动作
5. **验证**：告诉用户下一次对话可用 `/skill:<name> <问题>` 显式触发

## 第一步：访谈（必问三问）

1. **这个技能让 agent 做什么？** 具体产出是什么？
2. **什么时候触发？** 用户会说什么话、在什么上下文里应该自动加载？—— 这个答案直接决定 `description` 的质量，问得越细越好
3. **需要附属文件吗？** 默认纯指令（只有 SKILL.md）。只有确实需要模板、清单、脚本时才加 `references/`、`templates/` 等文件

访谈中主动追问：边界情况、输入输出格式、失败时怎么办、有没有现成的例子或格式规范。

## 第二步：起草规范

### name（发布名 = frontmatter name，两者必须一致）

- 小写字母、数字、单连字符（kebab-case），如 `weekly-report`、`code-review`
- 1–64 字符；禁止大写、下划线、连续连字符

### description（决定自动触发的质量）

- ≤1024 字符，必填
- 说清"做什么 + 什么时候用"，而不是泛泛而谈：

| 差 | 好 |
|---|---|
| 帮助处理周报 | 把本周工作要点整理成结构化周报，发送前检查数据口径。当用户提到周报、weekly report 时使用 |

### 正文结构（建议）

```markdown
# 技能名

## 触发场景
（用户会怎么问）

## 步骤
1. ...

## 输出格式
（可验证的格式要求）

## 失败处理
（缺数据、权限不足时怎么办）
```

### 附属文件

- 一律用技能内相对路径引用：`references/template.md`、`templates/weekly.md`
- 正文里说明何时去 `load_skill(name, path="references/...")` 读取

## 第三步：确认

发布前把以下三点讲给用户，等明确同意：

1. 技能名（kebab-case）
2. description 原文（用户最该审的就是这个）
3. 文件清单（SKILL.md + 哪些附属文件）

用户没同意就不调 `publish_skill`。用户提出修改就回到起草循环。

## 第四步：发布（契约）

调用 `publish_skill`，注意这些硬约束：

- `files` 必须包含恰好一个 `"SKILL.md"`，其余为技能内相对路径（POSIX，无 `..`）
- frontmatter 的 `name` 必须等于发布名；`description` 非空
- 每文件 ≤50,000 字符；每用户 20 个技能 / 20MiB 配额
- 发布同名 = 更新该技能；发布只写入**当前用户自己的**技能层，永远不碰运营者全局技能
- 用户自己的技能同名时会覆盖全局技能（只对他自己生效）

发布成功后告知用户：下一次对话用 `/skill:<name> 具体问题` 即可显式触发，或靠 description 自动触发。

## 第五步：验证与迭代

- 提醒用户可以试跑一次，把"没触发/步骤不对/格式不对"的反馈带回来
- 用户反馈触发不准 → 优先改 `description`（触发词/场景描述）
- 用户反馈流程不对 → 改正文步骤
- 用户不想要了 → 调 `delete_skill(name)`

## 注意事项

- 技能是参考文本，不是授权：技能里写的命令、脚本不会自动执行，需要 agent 在 run 里按其指导行事
- 不要在正文里写入用户机密（密钥、密码）——技能会跨 run 持久存在
- 保持技能短小：一个技能只做一件事；想做的事太多就拆成多个技能
