# ADR 0054 — MinerU becomes a workspace package, and the component tiers are written down

## Status

Accepted. Implementation in progress: the package and the workspace gate move together (one
slice), then the cutover, the documentation, and an audit pass over `packages/observability`.
This ADR states a boundary that `packages/` has enforced in code since it existed — the import
contracts in the root [`pyproject.toml`](../../pyproject.toml) and
[`verify_workspace_wheels.py`](../../scripts/verify_workspace_wheels.py) — but never wrote down.

## Context

### The question that exposed the gap

A review of the MinerU component — `scripts/mineru/` then, `sidecars/mineru/` after the first
slice — asked whether it belongs under `packages/`. The repository had no written rule to answer
with, so the two candidates were compared on the wrong axis. MinerU and ClickStack are **both
sidecars**: a sidecar is a topology — a process that runs beside the application and is reached
through a local interface. The OTel collector is the canonical example, so
`packages/observability` carries a sidecar too. Topology decides nothing about where *our* code
lives.

The axis that does decide it is narrower: **is our part of this component a distribution we own,
install, version and depend on — and is it host-neutral?** "Installed into the application's
environment" is the *consequence* of that, not the test:

| Component | Third-party runtime | Our part is | Installed into the app env? |
|---|---|---|---|
| `dlightrag-observability` | ClickStack containers | a host-neutral library the app imports (`_compose.py:76,113,205`) plus a console script the Makefile runs | **yes** |
| `dlightrag-memory` | PostgreSQL | a host-neutral library plus its own MCP server | **yes** |
| `chart-render/` | Node + ECharts | a subprocess the agent's Python calls (`subprocess.run(["echarts-render", …])`, `charts/SKILL.md:12`), even though the image links it onto `PATH` | no |
| `postgres/` | pgvector image | a Dockerfile and `init.sql` | no |
| MinerU (today) | the `mineru` package in its own venv | host shell; its interpreter-startup shim is injected into the *sidecar's* interpreter | no |

MinerU does **not** qualify under that test as it stands today: nothing of ours is importable and
nothing of ours is installed. The decision below is therefore explicit rather than deduced — we
choose to make the sidecar manager a first-party product surface, which is what turns it into a
package. If we chose not to, `sidecars/mineru/` would remain a correct home.

### What MinerU actually is, measured

`sidecars/mineru/` is 1015 lines of script and module, plus a 100-line README. By purpose:

- **270** platform supervision (`launch_agent.sh` 123, `systemd_service.sh` 123, `service.sh` 24),
  where the two backends are structurally parallel — one specification written out twice;
- **256** interpreter-startup guards for five upstream gaps (`sitecustomize.py`: image-pixel
  ceiling, title-prompt repair, bounded title-level requests, hybrid parse effort, HTTP keep-alive);
- **183** install and version-range derivation (`install.sh` 106, `lightrag_mineru_api.py` 77),
  which reads the pinned LightRAG client to decide which MinerU major this repository can drive;
- **179** title-aided LLM wiring (`title_aided.sh` 71, `_write_mineru_json.py` 108);
- **76** launcher and environment glue (`api.sh` 44, `env.sh` 32);
- **51** the Gradio WebUI launcher (`gradio.sh`).

Only one step is product knowledge — resolving the endpoint's reasoning dialect from DlightRAG's
model catalogue (`_write_mineru_json.py:21-41`). Everything else is host-neutral in nature; the
coupling is in how it is written (bash, paths relative to a checkout, `uv run --project`), not in
what it is.

### The Gradio WebUI

`gradio.sh` and its agent/unit exist to preview how a document parses. It is a MinerU diagnostic
tool, not a DlightRAG surface: it has no authentication, it reuses the API's single backend, and
it doubles the number of services every lifecycle command manages.

## Decision

### 1. The component tiers, stated

- `packages/<name>` — a component **we own** as a distribution the application depends on (by
  import or by console script), which stays **host-neutral**: no import of `dlightrag`,
  `lightrag`, or the application's other dependencies. It builds a wheel and carries a version in
  the release.
- `src/dlightrag/` — product code that is not reusable outside this product, and may import
  LightRAG and everything else. `frontend/` builds into it as static assets
  (`Dockerfile:49` → `static/app/`), so the browser shell is product code too.
- root component directories — what the application reaches **out of process**, in either of two
  shapes: a third-party runtime plus the glue that deploys or drives it (`postgres/`,
  `agent-browser/`), or a first-party tool whose interface to the product is an executable on
  `PATH` and whose toolchain is not Python (`chart-render/`, invoked as `echarts-render` and
  `html-report` by the built-in Skills). A root component carries no import surface, so it sits
  outside the Python lock and the release contract — and it therefore **owes a gate** for whatever
  version it pins: `platform-check` holds the Playwright pin across the client, the lock, the pool
  image and Compose; `install-chart-tools.sh` reads `RESVG_VERSION` out of the `Dockerfile`
  instead of repeating it; `echarts` appears once, in `chart-render/package.json`, with its lock.
- `scripts/`, `tests/`, `docs/`, `prompts/`, `.github/` — repository support: tooling, tests,
  documentation, prompt assets, CI.
- `blobs/`, `dlightrag_storage/`, `frontend/node_modules/` — runtime state and build output, not
  source.

"Sidecar" names a topology, never a tier.

### 2. MinerU becomes `packages/mineru`, as a development dependency

`dlightrag-mineru`, stdlib-only, host-neutral, with one console script:

```text
dlightrag-mineru install                     # service venv, MinerU range, shim, launcher
dlightrag-mineru api [args…]                 # foreground parse backend
dlightrag-mineru service {install|start|stop|status|logs|uninstall}
dlightrag-mineru purge [--yes] [--dry-run]   # removes the service venv it installed
dlightrag-mineru title-aided …               # writes ~/mineru.json from explicit fields
```

It joins the workspace as a **development dependency group member**, not a runtime dependency. The
manager has nothing to do inside the application container — MinerU runs on the host, and the image
has no launchd, systemd or `uv` — so a runtime dependency would ship dead weight and blur the
deployment topology ADR 0006 separates. `uv run` (and therefore every `make mineru-*`) includes the
default groups, while the image's `uv sync --frozen --no-dev` excludes it.

This revises the wheel gate's current invariant ("every workspace member is a dependency of
`dlightrag`") to "every member is a dependency of `dlightrag`, a development dependency, or
both" — a deliberate change, recorded here, because the old invariant cannot express a host tool.

`make mineru-*` becomes `uv run dlightrag-mineru …`, the same shape as `make hyperdx-bootstrap` →
`uv run dlightrag-observability bootstrap` (`Makefile:48-49`).

### 3. The package and the gate are one slice

`packages/mineru` cannot land alone: `verify_workspace_wheels.py` hard-codes the member list
(`:71`), the distribution table (`:29-33`), the artifact count (`:712`), the source-include
allow-list (`:649-653`), and the external-prohibition table (`:93-97,166`);
`verify_release_contract.py:72-85` and `release.py:27-37` hard-code the three version manifests;
`pyproject.toml:135-139` names the members. All of it moves in the same change, or `uv lock
--check`, `make ci` and `make workspace-wheels` go red on a package nobody can install.

### 4. The service runtime never depends on the application environment

`install` materializes everything the service needs **inside the service venv**:

- the interpreter-startup guards (the shell's `sitecustomize.py`, moved byte-for-byte and renamed
  `_dlightrag_mineru_shim.py`) into the venv's site-packages, behind a `dlightrag-mineru.pth`
  that imports them — see the Decisions section for why this replaced `PYTHONPATH`;
- the package's env reader, copied as `_env.py` so the two can never drift on what `.env.mineru`
  means;
- a launcher that loads `.env.mineru` (environment first, file second, as `env.sh` does today) and
  `exec`s the MinerU binary.

launchd and systemd point at absolute paths inside that venv (`<venv>/bin/python
<venv>/dlightrag-mineru/launcher.py`). The manager runs in the application environment and only
authors and drives the units; **no wrapper process sits on the supervised path**, so `KeepAlive` /
`Restart=on-failure` keep acting on MinerU itself.

**One anchor.** The project root — the nearest ancestor holding a `pyproject.toml`, else the working
directory — resolves both the service venv (when `MINERU_SERVICE_VENV` is relative) and the env
file (when `MINERU_ENV_FILE` is relative). `install` resolves once and the unit is written from that
resolution, so a later `service install` from another directory cannot point the service at a
different venv. `service install` refuses with exit 127 and "run `dlightrag-mineru install`
first" when the launcher is missing, which is the hint `api.sh:36` gives today.

### 5. Behaviour parity is an acceptance criterion

Nothing about macOS or Linux background serving changes except the removal of Gradio. The ported
renderers produce the same artifacts and drive the same commands:

| macOS (launchd) | Linux / WSL2 (systemd --user) |
|---|---|
| label `com.hanlianlyu.dlightrag.mineru-api`, domain `gui/$UID` | unit `dlightrag-mineru-api.service` |
| plist `RunAtLoad=true`, `KeepAlive=true`, `WorkingDirectory`=repository root, `EnvironmentVariables` (`MINERU_ENV_FILE`), `StandardOutPath`/`StandardErrorPath` in `~/Library/Logs/dlightrag/` | `Type=simple`, `After=network.target`, `Restart=on-failure`, `RestartSec=3`, `WantedBy=default.target`, `Environment=MINERU_ENV_FILE=…` |
| `install` = bootout the label → write plist → bootstrap | `install` = `disable --now` + remove → write unit → `daemon-reload` → `enable --now` |
| `bootstrap` retried 5 times with 0.5 s backoff (quietly), then once more with the error visible — `bootout` is asynchronous (`launch_agent.sh:66-78`) | the same write → reload → enable ordering, whose `daemon-reload` failure aborts the install |
| `start` = bootstrap; `stop` = bootout; `uninstall` = bootout + remove plist; `status` = `launchctl print`, propagating its exit code | `start` = `systemctl --user start`; `stop` = `systemctl --user stop`; `uninstall` = `disable --now` + remove + `daemon-reload`; `status` = `systemctl --user status` |
| `logs` = `tail -n 200 -f` over both log files | `logs` = `journalctl --user -n 200 -f` |
| unsupported OS → 64 (`service.sh:19-23`; the message names the supported platforms rather than a foreground escape) | no `systemctl` → 127 with the foreground hint (`systemd_service.sh:31-37`) |

Exit codes are part of parity: 64 for an unsupported OS, an unknown service action, a pin outside
the derived range (`install.sh:86`), or a `purge` awaiting `--yes`; 127 when `uv`
(`install.sh:100`), `mineru-api` (`api.sh:36`) or `systemctl` is missing; 1 for a title-aided
refusal (`title_aided.sh:53,59,65`), an env file that exists but cannot be read, and a `purge`
that refuses a directory which is not a service venv.

Two tests hold this: a **render-parity** test (the artifacts these renderers produce, against
artifacts captured from the shell manager before it was deleted) and a **command-parity** test (the
same fake `launchctl`/`systemctl` argv sequences, including retry counts and `daemon-reload`
ordering). systemd had no test of its own before this; it gets both. The launcher, the env reader and the
guards run under the service venv's Python (3.13 today, `install.sh:15`), not this project's 3.14,
so a test parses all three with 3.13's grammar — a formatter-friendly 3.14 spelling in the copied
`env.py` once stopped the service dead — and compiles them with that interpreter when a service
venv is present. Real launchd/systemd behaviour
is smoke-tested by hand on macOS.

Known limitation, deliberately unchanged: no `loginctl enable-linger`, so a Linux user service ends
with the session. Changing that is a separate decision.

### 6. The Gradio surface is deleted

`gradio.sh`, `MINERU_GRADIO_*`, its launchd agent and systemd unit, and every test that asserts
them. `install`, `stop` and `uninstall` cleaned up a pre-existing Gradio agent or unit for one
release; that release has passed and they no longer touch either.

### 7. The title-aided seam splits along the same axis

Product knowledge stays in the application: `src/dlightrag/engine/rag/corpus/mineru_title_aided.py`
resolves the endpoint's reasoning dialect from DlightRAG's model catalogue and calls the package's
writer, exposed as the console script `dlightrag-mineru-title-aided` and wired to
`make mineru-title-aided`. The package writes `~/mineru.json` from explicit fields (`api_key`,
`base_url`, `model`, `extra_body`), so any host can use it and it imports nothing of ours. The file
holds a credential, so it is written atomically and with mode 0600.

### 8. `.env.mineru.example` moves into the package

It becomes `packages/mineru/.env.example`, shipping in the source distribution beside
the other members' assets (uv_build's `source-include` reaches the sdist, not the wheel:
the template documents a checkout, and `install` needs no file to run). The **live**
file keeps its location and resolution rules (`.env.mineru` at the project root,
`MINERU_ENV_FILE` override), so an existing install is unaffected — and `.env.mineru`
remains a dotenv file holding the title-aided credential, as ADR 0006 requires of a
credential: the service's own setting, never `config.yaml`.

**Where a live config file belongs follows the material it configures.** This repository
ships as a repository, not as a package: nothing is published, and `make workspace-wheels`
is a local check that distributes nothing. A checkout-resident deployable keeps its config
inside its package — `packages/observability/stack/.env` sits with the compose stack that
reads it, excluded from artifacts by an explicit `source-exclude`. Host state keeps its
config beside that state: MinerU's service venv, launchd/systemd units and logs live
outside this package by design (§4), so `.env.mineru` stays at the project root where
those live, next to the application's own `.env`, and a credential never ends up inside
the tracked source tree. The template is code and travels with the package; the live file
is host state and does not.

## Consequences

- The release contract gains a fourth manifest (`make release` writes it in lockstep), and the wheel
  gate verifies one more distribution in isolation.
- The wheel gate's member-dependency invariant is revised: a member may be a development dependency
  instead of a runtime one.
- The MinerU compatibility range and the "delete this when upstream fixes it" notes for the five
  guards gain a single owner.
- The two platform backends become one specification with two renderers, and the shell's
  `shellcheck` surface for MinerU is replaced by `pytest`.
- A host that only wants the sidecar manager installs it without cloning the repository.
- Recorded, not decided: `packages/observability` has the weakest reuse case of the three members —
  its OTLP transport is upstream's, and its stack is a passenger of the code that drives it. The
  alternative (dissolve the exporter into `src/dlightrag/` and move the stack to a root component
  directory) was considered and declined here: one owner for the seam and the stack it feeds, with
  host-neutrality enforced by contract, is worth one distribution.
- `sidecars/` disappears; `docs/configuration.md`, `docs/operations.md`, the README, and
  `prerequisite_setup.py` point at the package.

## Decisions taken after the first implementation

Each of these was an open question in the first draft of this ADR; the reasoning is kept
because the alternatives were real.

- **The guards ride a `.pth` in the service venv's site-packages**, not `PYTHONPATH` set by
  the launcher. `PYTHONPATH` only reaches interpreters the launcher starts, so a direct
  `mineru-api` run (debugging, or another supervisor) silently lost all five upstream
  guards. `install` removes the `.pth` before upgrading MinerU and writes it afterwards,
  because `uv pip install` runs that interpreter and `site` would import guards against a
  half-installed package.
- **`purge` exists**, with guards: it removes a directory only when it holds `pyvenv.cfg`
  and our launcher marker, it refuses a root, a home, and the project, it prints what it
  would remove under `--dry-run`, and it wants `--yes` otherwise. The venv is 1.6 GB on a
  working host; leaving removal to `rm -rf` invites the mistake the guards exist to stop.
- **An unreadable env file is not an absent one.** `env.EnvFileError` reaches the CLI (exit
  1) and the launcher (exit 1, so the supervisor reports it) instead of falling back to
  host/port/effort defaults — a sidecar running on the wrong port looks like it is working.
- **`status` propagates the manager's exit code on both platforms.** The shell's macOS
  `launchctl print … || true` made a status check that cannot fail; systemd always
  propagated. This is a deliberate deviation from §5's parity table, recorded here.
- **The tool is POSIX-only** — macOS, Linux, WSL2 — and says so (exit 64). `bin/python`,
  `os.getuid()`, launchd and systemd are all absent or different on Windows, so "foreground
  only" was an inaccurate promise rather than a smaller feature.
- **The copied modules are pinned to the service interpreter's grammar.** `env.py` is copied
  verbatim into a 3.13 venv while this project targets 3.14, and the formatter rewrote a tuple
  `except` into 3.14-only syntax; the suite now parses every copied module with
  `feature_version=(3, 13)`.
- **systemd's unit is validated by `systemd-analyze verify`** in the test suite, which runs
  on Linux CI and skips on macOS; a missing validator fails under `CI` rather than skipping
  silently, the same rule the chart-render tests follow.

## Still open

- The systemd unit is verified statically in CI, but no real Linux host has yet run
  `make mineru-install && make mineru-service-install && make mineru-service-status/logs/stop`.
  That one-off smoke is still owed; the artifacts and command sequences are pinned meanwhile.
- Purge removes the venv only. `service uninstall` removes the units, and the order between them
  is the operator's; the command hints at it, and nothing enforces it.
