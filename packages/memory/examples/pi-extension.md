# Pi extension example (dlightrag-memory)

The Memory package runs as a stdio MCP server; the host owns the process and
the database. This is the reference shape for a Pi extension.

```text
1. Database (one command, extension-repo compose file):
   docker compose -f packages/memory/compose.yaml up -d

2. Extension spawns the server (subject bound at launch, never a tool arg):
   dlightrag-memory-mcp \
     --dsn postgres://dlightrag:dlightrag@127.0.0.1:5432/dlightrag \
     --subject <pi-user-id>

3. The model gets three tools:
   memory_recall(query)          — low-authority context, never citable
   memory_remember(kind, body, confidence, idempotency_key, supersedes_id?)
   memory_forget(memory_id | body)
```

Pi specifics:

- Spawn in an extension-owned subprocess; stdio transport is the only
  official protocol. Do not run the server as a daemon or over HTTP.
- The subject is the Pi user identity (e.g. the OS user or a configured
  stable id); the extension passes it at launch.
- Injected recall is model-controlled: the agent calls `memory_recall`
  itself. The package provides no automatic context injection hook.
- Lifecycle: terminate the subprocess in `session_shutdown`; restart it in
  `session_start` if it died.
