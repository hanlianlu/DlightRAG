# VS Code Auto-Approve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all new VS Code agent sessions default to Bypass Approvals across every workspace.

**Architecture:** Configure the documented `chat.permissions.default` setting at VS Code user scope. Leave repository settings and existing fine-grained approval rules intact so they remain available whenever a session uses Default Approvals.

**Tech Stack:** VS Code user settings (JSON), macOS `plutil`, ripgrep

---

### Task 1: Set the user-level default permission

**Files:**
- Modify: `/Users/hanlianlyu/Library/Application Support/Code/User/settings.json`
- Reference: `docs/superpowers/specs/2026-07-26-vscode-auto-approve-design.md`

- [ ] **Step 1: Verify the user-level setting is currently absent**

Run:

```bash
rg -n '"chat\.permissions\.default"' "/Users/hanlianlyu/Library/Application Support/Code/User/settings.json"
```

Expected: no output and exit status `1`. A workspace-scoped copy may exist in
`/Users/hanlianlyu/Github/DlightRAG/.vscode/settings.json`, but that does not
satisfy the requested all-workspace scope.

- [ ] **Step 2: Add the user-level setting**

Change the final settings block from:

```json
    "chat.tools.terminal.ignoreDefaultAutoApproveRules": true,
    "kimi.yoloMode": true,
    "github.copilot.chat.claudeAgent.allowAutoPermissions": true
}
```

to:

```json
    "chat.tools.terminal.ignoreDefaultAutoApproveRules": true,
    "kimi.yoloMode": true,
    "github.copilot.chat.claudeAgent.allowAutoPermissions": true,
    "chat.permissions.default": "autoApprove"
}
```

- [ ] **Step 3: Validate the settings file syntax**

Run:

```bash
/usr/bin/plutil -lint "/Users/hanlianlyu/Library/Application Support/Code/User/settings.json"
```

Expected:

```text
/Users/hanlianlyu/Library/Application Support/Code/User/settings.json: OK
```

- [ ] **Step 4: Verify the configured value**

Run:

```bash
rg -n '"chat\.permissions\.default": "autoApprove"' "/Users/hanlianlyu/Library/Application Support/Code/User/settings.json"
```

Expected: exactly one matching line containing:

```text
"chat.permissions.default": "autoApprove"
```

- [ ] **Step 5: Confirm repository runtime files remain unchanged**

Run:

```bash
git -C /Users/hanlianlyu/Github/DlightRAG status --short
```

Expected: no runtime, dependency, or workspace configuration changes. The user
setting is outside the repository and must not be committed.

- [ ] **Step 6: Activate the default in a new session**

Start a new VS Code agent session. If VS Code displays the one-time elevated
permission warning, confirm it. Verify the session permission selector shows
**Bypass Approvals**. Existing sessions are not expected to change.
