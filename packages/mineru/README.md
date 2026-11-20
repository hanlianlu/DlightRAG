# MinerU sidecar manager

The host-side MinerU service that DlightRAG's corpus ingestion parses through.
This distribution owns the host half: the dedicated service virtualenv, the MinerU
release range, the interpreter-startup guards, and the launchd/systemd units that keep
it running. DlightRAG never imports anything of MinerU's client from here — it
reaches the running service over HTTP at the endpoint `corpus.sidecars.mineru.*` names
in [`config.yaml`](../../config.yaml)
([Parser Sidecars](../../docs/configuration.md#parser-sidecars)).

## Why this is a package

[ADR 0054](../../docs/adr/0054-mineru-is-a-workspace-package.md) writes the tier rule
down: `packages/<name>` is a component **we own** as a distribution the application
depends on — by import or by console script — and that stays host-neutral. MinerU
qualifies because we chose to ship the manager as a product surface; before that it was
a component directory, and that was a defensible home too. "Sidecar" names a topology,
not a tier, which is why `packages/observability` also carries a sidecar (the ClickStack
containers) while `postgres/` does not live here at all.

Two consequences of the rule are visible in this directory:

- **It is a development dependency, not a runtime one.** The application container has
  no launchd, no systemd and no `uv`, and MinerU runs on the host, so the image's
  `uv sync --no-dev` leaves this manager out. Every `make mineru-*` and `uv run` still
  sees it.
- **The service runtime never depends on the application environment.** `install`
  writes everything the service needs *inside the service venv*, and the units point at
  absolute paths there. The manager authors and drives the units; it is never on the
  path the supervisor watches, so `KeepAlive` / `Restart=on-failure` keep acting on
  MinerU itself.

## Commands

`make mineru-*` are the documented entry points; each is a thin wrapper around the
console script.

| Command | Make target | What it does |
|---|---|---|
| `dlightrag-mineru install` | `make mineru-install` | Creates the service venv, installs MinerU into it, places the shim and launcher |
| `dlightrag-mineru api` | `make mineru-api` | Runs the parsing backend in the foreground (`:8210`) |
| `dlightrag-mineru service install` | `make mineru-service-install` | Writes the unit and starts it |
| `dlightrag-mineru service {start,stop,status,logs,uninstall}` | `make mineru-service-*` | The rest of the lifecycle |
| `dlightrag-mineru title-aided` | `make mineru-title-aided` | Writes the title-aided LLM config to `~/mineru.json` |
| `dlightrag-mineru purge [--yes] [--dry-run]` | — | Removes the service venv, after checking it is one of ours |

`install` places the service's whole runtime inside the service venv:

| File | Where | Role |
|---|---|---|
| `launcher.py` | `<venv>/dlightrag-mineru/` | Loads the env file and `exec`s MinerU |
| `_env.py` | `<venv>/dlightrag-mineru/` | A copy of this package's `env.py`, so the launcher and the manager can never disagree about what `.env.mineru` means |
| `_dlightrag_mineru_shim.py` | `<venv>/…/site-packages/` | Interpreter-startup guards for five upstream gaps: the image-pixel ceiling, title-prompt repair, bounded title-level requests, hybrid parse effort, and the HTTP keep-alive the client's polling needs |
| `dlightrag-mineru.pth` | `<venv>/…/site-packages/` | `import _dlightrag_mineru_shim`, so the guards load for **every** start of that interpreter — including a direct `mineru-api` run that bypasses the launcher. `install` drops the `.pth` before upgrading MinerU and writes it back after, so the guards never load against a half-installed package |

The manager runs on **macOS, Linux, and WSL2**; a host without launchd or systemd --user is
refused, because a background-service command that silently does nothing is worse than an
error. `dlightrag-mineru service status` reports through its exit code, as `systemctl`
status does.

## Configuration

Copy the template to the **project root** as `.env.mineru`:

```bash
cp packages/mineru/.env.example .env.mineru
```

The live file stays at the project root because it is a dotenv file beside `.env`, and
an existing install must keep working; `MINERU_ENV_FILE` moves it. The template itself
ships in this directory (and in the source distribution), not in the wheel: it is
documentation for a checkout, and `install` needs no file to run.

`corpus.sidecars.mineru` in `config.yaml` is the **application** side: the endpoint,
language, backend, and polling window DlightRAG and LightRAG read. The two halves stay
separate — a helper setting here never becomes product config. The title-aided
credential lives in `.env.mineru`, never in `config.yaml`.

The title-aided model endpoint, key, and reasoning dialect come from DlightRAG's own
model catalogue through `make mineru-title-aided`
([`mineru_title_aided.py`](../../src/dlightrag/engine/rag/corpus/mineru_title_aided.py)):
the application resolves which request fields turn reasoning off for *this* deployment's
endpoint, and this package writes them into `~/mineru.json` — atomically, mode 0600.

## MinerU version range

Which MinerU major this repository can drive is a property of the LightRAG client it
pins, not a number someone remembers to raise. MinerU 3.x serves the local task API
(`POST /tasks`) that today's client calls; 4.x serves `/v1/parse/jobs`. `install` reads
the client's source in the project environment and installs the matching range, so
upgrading LightRAG is the whole of a MinerU 4.x upgrade here. `MINERU_MIN_VERSION` and
`MINERU_MAX_VERSION` override either end, and an empty `MINERU_MAX_VERSION` lifts the
ceiling rather than falling back to a default.

## Install extras, and the one deliberate asymmetry

`MINERU_INSTALL_EXTRAS` follows the same rule. Unset, it is derived: `core,mlx` on Apple
Silicon for MinerU 3.x, `core` everywhere else, and nothing for 4.x, which dropped the
extras. The derivation reads the platform and the architecture (`Darwin` **and**
`arm64`/`aarch64`), not the operating system's name alone, so an Intel Mac gets `core`
too. Set the key in `.env.mineru` — or in the environment, which wins — to choose a GPU
variant instead.

**The manager detects the platform; it does not detect a GPU.** The setup wizard does:
`prerequisite_setup.has_nvidia_gpu()` looks for `nvidia-smi` and then writes
`core,vllm`. That asymmetry is deliberate — a wrong GPU guess produces an environment
whose wheels do not resolve, and the GPU variants (`vllm`, `lmdeploy`, and the driver
requirements behind them) are a choice only the operator can make. Both halves are
pinned: `tests/unit/test_mineru_package.py` for the derivation,
`tests/unit/test_prerequisite_setup.py` for the wizard's probe.

## Background service

One service, dispatched per OS: launchd (`com.hanlianlyu.dlightrag.mineru-api`) on
macOS, `systemd --user` (`dlightrag-mineru-api.service`) on Linux and WSL2. The
Gradio WebUI that used to run beside it is gone (ADR 0054 §5); for one release the
lifecycle commands still remove an agent or unit an older install left behind. macOS
logs land in `~/Library/Logs/dlightrag/`; Linux logs go to the journal.
`make mineru-service-logs` follows whichever applies.

Real launchd/systemd behaviour is smoke-tested by hand; the artifacts and the command
sequences are covered by tests, including a render-parity check against artifacts
captured from the shell manager this replaced.

## Tests

[`tests/unit/test_mineru_package.py`](../../tests/unit/test_mineru_package.py) covers
the env reader, the version range, the installer, the CLI, both renderers and the
launcher, with the captured artifacts in
[`tests/unit/fixtures/mineru/`](../../tests/unit/fixtures/mineru). The launcher and the
shim run under the *service* venv's Python (3.13, MinerU's supported ceiling), not this
project's 3.14, so a test compiles both with that interpreter when it is present.
