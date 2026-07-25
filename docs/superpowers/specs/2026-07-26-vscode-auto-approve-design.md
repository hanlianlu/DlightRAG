# VS Code Auto-Approve Design

## Goal

Make every new VS Code agent session default to **Bypass Approvals** across all
workspaces, avoiding repeated manual approval for workspace access and other
tool calls.

## Approach

Add the following user-level VS Code setting:

```json
"chat.permissions.default": "autoApprove"
```

The setting belongs in the user's VS Code `settings.json`, not in this
repository. User scope is required because the behavior should apply to every
workspace.

## Behavior

- New agent sessions start with the **Bypass Approvals** permission level.
- Existing sessions retain their current permission level. The user must start
  a new session or switch the session permission selector manually.
- VS Code may show a one-time safety confirmation the first time the elevated
  permission level is used.
- Existing terminal, URL, and tool-specific approval rules remain unchanged.
  They continue to apply whenever a session uses **Default Approvals**.

## Safety

Bypass Approvals permits file edits, terminal commands, URL access, and external
tool calls without per-action confirmation. This is an intentional global user
preference requested by the user.

## Validation

1. Parse the updated user `settings.json` as JSON.
2. Confirm `chat.permissions.default` resolves to `autoApprove`.
3. Confirm no repository runtime or dependency files were changed.
