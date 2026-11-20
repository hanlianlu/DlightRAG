---
name: skill-creator
description: Use when the user wants to make, edit or remove a personal DlightRAG skill, asks to turn a repeated task or a procedure just walked through into a skill, or wants one sentence to trigger a routine.
---

# Skill Creator

为 DlightRAG 创建个人技能的引导流程。你负责访谈、起草与发布；用户负责确认。

## 步骤

1. **访谈**：问清三件事，不确定就继续问。
   - 这个技能让 agent 做什么，具体产出是什么？
   - 什么时候触发？用户会说什么话、处理什么对象？这个答案直接决定 `description`。
   - 需要附属文件吗？默认只有 `SKILL.md`；只有确实需要模板、清单、脚本时才加 `references/`、`templates/`。

   同时追问边界情况、输入输出格式、失败时怎么办、有没有现成的例子。
2. **起草**：生成 `SKILL.md` 和必要的附属文件。附属文件用技能内的相对路径引用，并在正文里用到它的那一步写明「先用 `load_skill(name, path="references/...")` 读 X」。每份附属文件都要判断：不读就会做错的是必读，写进 `required_reads`（见下文）；按需查阅的不声明。
3. **确认**：把名称、`description` 原文和文件清单讲给用户，标明哪些是必读，等明确同意。没同意就不发布；用户要改就回到起草。
4. **发布**：调用 `publish_skill(name, files)`，这是唯一的持久化通道。frontmatter 的 `name` 要与发布名一致。成功后可以立刻用 `load_skill(name)` 读回，确认 frontmatter（包括 `required_reads`）被识别、正文无误，再告诉用户：下一轮回答起可以用 `/skill:<name> 具体问题` 显式触发，也可能靠 `description` 自动触发。
5. **迭代**：没触发就改 `description`，流程不对就改正文，用户不想要了就调 `delete_skill(name)`。删除只移除用户层，同名的全局或内置技能会重新出现。只是暂时不想用、以后可能还要，就调 `set_skill_enabled(name, false)` 停用，不要删除。

## 写好 description

`description` 决定技能会不会被自动加载。写成「当…时使用」，把具体场景写出来：用户会说什么、处理什么对象，再写明不适用的情形。越具体越容易被触发；「需要时使用」这类抽象说法几乎不会被触发。控制在 50 词以内——内置技能都按这条测试——因为它每个 session、每个用户都会被注入一次，每个词都在抢注意力；不要用 description 总结工作流程。值里有冒号，或有空格加 `#` 时，要给整个值加引号。

| 差 | 好 |
|---|---|
| 帮助处理周报 | 当用户提到周报、weekly report，或要求把本周工作要点整理成结构化周报时使用 |

## 声明必读文件

只有 `SKILL.md` 的技能不需要。技能依赖某份附属文件、不读就容易做错时，在 frontmatter 里写 `required_reads`；示例库、查阅表这类按需读的文件不要声明，否则每次都被标成没读，标记就失去意义。每一项是一个 `path`（或 `any_of`：列几个文件，读到任意一个就算）加一句 `when`（什么时候读，120 字以内）。文件必须随技能一起发布，不能是 `SKILL.md`，写错的键会被拒绝。系统只在产物旁标出没读到的文件，不会拦截交付。

```yaml
required_reads:
  - path: references/format.md
    when: 写输出之前
```

## 附带脚本时

脚本给运行中的 agent 用，只写它当场用得上的，如无必要不加：

- 规则写在出错的地方：生成结果的脚本遇到错误输入直接报错，说清错在哪、怎么改；不要再写一个检查脚本把同一条规则判一遍。
- 检查脚本只查生成时看不到的东西，比如成品是否齐全、各部分是否一致。
- 运行技能的 agent 是会犯错的合作者，不是对手：只拦它真会犯的错，不为想象出来的绕过路径加检查。
- 测试只证明规则对：每条规则一个通过、一个不通过的例子；不写证明两份代码一致的测试。

## 正文结构（建议）

```markdown
# 技能名

## 步骤
1. ...

## 输出格式
（可验证的格式要求）

## 失败处理
（缺数据、权限不足时怎么办）
```

## 注意事项

- 同名技能的优先级是：内置 < 运营者全局 < 用户；用户的同名覆盖只对当前用户生效。
- 技能是参考文本，不是授权：其中的命令不会自动执行。
- 不要在正文里写入用户机密（密钥、密码），技能会跨 run 持久存在。
- 保持短小：一个技能只做一件事，想做的事太多就拆成多个技能。
