#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = ["questionary>=2", "rich>=13", "ruamel.yaml>=0.18"]
# ///
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG one-command onboarding wizard.

Run with:  uv run prerequisite_setup.py

Module-level imports stay limited to the stdlib so the pure logic is importable
under pytest; ruamel.yaml is imported lazily in the config writer, and
questionary/rich are imported lazily inside the interactive functions.
"""

from __future__ import annotations

import datetime as _dt
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
MINERU_ENV_PATH = REPO_ROOT / ".env.mineru"
MINERU_ENV_EXAMPLE_PATH = REPO_ROOT / ".env.mineru.example"
API_HEALTH_URL = "http://localhost:8100/health"
WEB_URL = "http://localhost:8100/web/"


# ---------------------------------------------------------------------------
# Provider registry and role resolvers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderSpec:
    """Maps a novice-friendly provider name to DlightRAG config."""

    provider: str  # DlightRAG provider enum value
    base_url: str | None  # canonical default; None => native (no base_url)
    requires_url: bool = False  # user MUST supply (Azure / Other, tenant-specific)


# LLM roles: openai-compatible => provider "openai" + base_url; else native.
PROVIDERS_LLM: dict[str, ProviderSpec] = {
    "OpenAI": ProviderSpec("openai", "https://api.openai.com/v1"),
    "DeepSeek": ProviderSpec("openai", "https://api.deepseek.com"),
    "OpenRouter": ProviderSpec("openai", "https://openrouter.ai/api/v1"),
    "Anthropic": ProviderSpec("anthropic", None),
    "Gemini": ProviderSpec("gemini", None),
    "Azure OpenAI": ProviderSpec("openai", None, requires_url=True),
    "Other (OpenAI-compatible)": ProviderSpec("openai", None, requires_url=True),
}

# Embedding providers.
PROVIDERS_EMBED: dict[str, ProviderSpec] = {
    "Voyage": ProviderSpec("voyage", "https://api.voyageai.com/v1"),
    "OpenAI": ProviderSpec("openai_compatible", "https://api.openai.com/v1"),
    "Gemini": ProviderSpec("gemini", None),
    "Jina": ProviderSpec("jina", "https://api.jina.ai/v1"),
    "Azure OpenAI": ProviderSpec("openai_compatible", None, requires_url=True),
    "Ollama (local)": ProviderSpec("ollama", "http://localhost:11434"),
    "Other (OpenAI-compatible)": ProviderSpec("openai_compatible", None, requires_url=True),
}

# Known embedding model -> vector dim (pre-fill; asked otherwise).
EMBED_DIMS: dict[str, int] = {
    "voyage-multimodal-3.5": 1024,
    "voyage-3.5": 1024,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "jina-embeddings-v3": 1024,
}

# Rerank menu label -> (strategy, needs its own API key).
RERANK_CHOICES: dict[str, tuple[str, bool]] = {
    "Reuse my LLM": ("chat_llm_reranker", False),
    "Voyage": ("voyage_reranker", True),
    "Jina": ("jina_reranker", True),
    "Cohere": ("cohere_reranker", True),
    "Azure Cohere": ("azure_cohere", True),
}


def _model_block(spec: ProviderSpec, model: str, base_url: str | None) -> dict:
    block: dict = {"provider": spec.provider, "model": model}
    resolved = base_url or spec.base_url
    if resolved is not None:
        block["base_url"] = resolved
    return block


def resolve_llm_choice(provider_name: str, *, model: str, base_url: str | None) -> tuple[dict, str]:
    spec = PROVIDERS_LLM[provider_name]
    return _model_block(spec, model, base_url), "DLIGHTRAG_LLM__DEFAULT__API_KEY"


def resolve_embedding_choice(
    provider_name: str, *, model: str, base_url: str | None
) -> tuple[dict, str]:
    spec = PROVIDERS_EMBED[provider_name]
    block = _model_block(spec, model, base_url)
    block["dim"] = EMBED_DIMS.get(model, 0)  # 0 => caller must prompt
    return block, "DLIGHTRAG_EMBEDDING__API_KEY"


def resolve_rerank_choice(choice: str) -> tuple[dict, str | None]:
    strategy, needs_key = RERANK_CHOICES[choice]
    return {"strategy": strategy}, ("DLIGHTRAG_RERANK__API_KEY" if needs_key else None)


# ---------------------------------------------------------------------------
# config.yaml writer (comment-preserving) and .env upsert
# ---------------------------------------------------------------------------
def _yaml():
    from ruamel.yaml import YAML  # lazy: PEP 723 runtime dep / dev-test dep

    y = YAML()  # round-trip mode by default: preserves comments
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _apply_model_block(node, block: dict) -> None:
    """Set provider/model/base_url/dim on a mapping; drop base_url when absent."""
    for key in ("provider", "model", "base_url", "dim"):
        if key in block:
            node[key] = block[key]
        elif key == "base_url" and key in node:
            del node[key]  # native provider: remove stale base_url


def write_config_yaml(
    path: Path,
    *,
    llm_default: dict | None = None,
    llm_roles: dict[str, dict] | None = None,
    embedding: dict | None = None,
    rerank: dict | None = None,
    mineru_api_mode: str | None = None,
) -> None:
    yaml = _yaml()
    data = yaml.load(path)
    if llm_default is not None:
        _apply_model_block(data["llm"]["default"], llm_default)
    if llm_roles is not None:
        if not llm_roles:
            data["llm"].pop("roles", None)
        else:
            roles = data["llm"].setdefault("roles", {})
            roles.clear()
            for role, block in llm_roles.items():
                roles.setdefault(role, {})
                _apply_model_block(roles[role], block)
    if embedding is not None:
        _apply_model_block(data["embedding"], embedding)
    if rerank is not None:
        for key, value in rerank.items():
            data["rerank"][key] = value
    if mineru_api_mode is not None:
        data["parser_sidecars"]["mineru"]["api_mode"] = mineru_api_mode
    yaml.dump(data, path)


def upsert_env(path: Path, values: dict[str, str]) -> None:
    """Insert/replace KEY=value lines; preserve all other lines and order."""
    remaining = dict(values)
    lines: list[str] = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            key = raw.split("=", 1)[0].strip() if "=" in raw else ""
            if key in remaining:
                lines.append(f"{key}={remaining.pop(key)}")
            else:
                lines.append(raw)
    for key, value in remaining.items():
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Backup + validation
# ---------------------------------------------------------------------------
MAX_BACKUPS = 1  # keep only the most recent <file>.bak-<timestamp> per file


def _prune_backups(path: Path, *, keep: int) -> None:
    """Delete all but the newest ``keep`` ``<name>.bak-*`` siblings of ``path``."""
    if keep < 0:
        return
    backups = sorted(path.parent.glob(f"{path.name}.bak-*"), key=lambda p: p.name)
    for old in backups[: len(backups) - keep]:
        old.unlink(missing_ok=True)


def backup_file(path: Path, *, keep: int = MAX_BACKUPS) -> Path | None:
    if not path.exists():
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    backup.write_bytes(path.read_bytes())
    _prune_backups(path, keep=keep)
    return backup


def validate_config() -> None:
    """Load config via DlightRAG to validate what we wrote. Raises on invalid.

    Imported lazily so the pure logic stays importable without the runtime
    package. Wired into the orchestrator (Plan 3) where the full config exists.
    """
    from dlightrag.config import load_config, reset_config

    reset_config()
    load_config()


# ---------------------------------------------------------------------------
# Platform / GPU / WSL2 detection and preflight
# ---------------------------------------------------------------------------
_PROC_VERSION = Path("/proc/version")


@dataclass(frozen=True)
class PlatformInfo:
    os: str  # "macos" | "linux" | "windows"
    arch: str  # "arm64" | "x86_64" | ...
    is_wsl: bool


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    hint: str = ""


def detect_platform() -> PlatformInfo:
    system = platform.system()
    os_name = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
        system, system.lower()
    )
    is_wsl = False
    if os_name == "linux" and _PROC_VERSION.exists():
        is_wsl = "microsoft" in _PROC_VERSION.read_text(encoding="utf-8", errors="ignore").lower()
    return PlatformInfo(os=os_name, arch=platform.machine().lower(), is_wsl=is_wsl)


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def run_preflight() -> list[Check]:
    info = detect_platform()
    checks: list[Check] = []
    for tool, hint in (
        ("uv", "https://docs.astral.sh/uv/"),
        ("docker", "https://docs.docker.com/get-docker/"),
        ("make", "macOS: xcode-select --install | Debian: apt-get install make"),
    ):
        checks.append(Check(tool, shutil.which(tool) is not None, hint))
    if info.os == "windows" and not info.is_wsl:
        checks.append(
            Check(
                "wsl2",
                False,
                "Run this wizard inside WSL2 (Docker Desktop WSL2 backend).",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# MinerU extras + hybrid service-model resolution (consumed by Plan 2)
# ---------------------------------------------------------------------------
def select_mineru_extras(info: PlatformInfo, *, has_gpu: bool) -> str:
    if info.os == "macos" and info.arch in ("arm64", "aarch64"):
        return "core,mlx"
    if has_gpu:
        return "core,vllm"
    return "core"


def resolve_service_model(info: PlatformInfo, *, systemd_available: bool) -> str:
    """Hybrid: background where a first-class mechanism exists, else foreground."""
    if info.os == "macos":
        return "launchd"
    if info.os == "linux" and systemd_available:
        return "systemd-user"
    return "foreground"


def systemd_user_available() -> bool:
    if shutil.which("systemctl") is None:
        return False
    result = subprocess.run(
        ["systemctl", "--user", "is-system-running"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 or "running" in result.stdout or "degraded" in result.stdout


# ---------------------------------------------------------------------------
# MinerU parser: official cloud vs local install + hybrid background service
# ---------------------------------------------------------------------------
def configure_mineru_official(token: str) -> None:
    write_config_yaml(CONFIG_PATH, mineru_api_mode="official")
    upsert_env(ENV_PATH, {"DLIGHTRAG_PARSER_SIDECARS__MINERU__API_TOKEN": token})


def configure_mineru_local_env(extras: str, *, title_aided: dict | None = None) -> None:
    if not MINERU_ENV_PATH.exists() and MINERU_ENV_EXAMPLE_PATH.exists():
        MINERU_ENV_PATH.write_bytes(MINERU_ENV_EXAMPLE_PATH.read_bytes())
    values = {"MINERU_INSTALL_EXTRAS": extras}
    if title_aided:
        values["MINERU_TITLE_AIDED_ENABLE"] = "true"
        values["MINERU_TITLE_AIDED_API_KEY"] = title_aided["api_key"]
        values["MINERU_TITLE_AIDED_BASE_URL"] = title_aided["base_url"]
        values["MINERU_TITLE_AIDED_MODEL"] = title_aided["model"]
    upsert_env(MINERU_ENV_PATH, values)
    write_config_yaml(CONFIG_PATH, mineru_api_mode="local")


def build_mineru_local_commands(service_model: str, *, title_aided: bool) -> list[list[str]]:
    cmds: list[list[str]] = [["make", "mineru-install"]]
    if title_aided:
        cmds.append(["make", "mineru-title-aided"])
    if service_model in ("launchd", "systemd-user"):
        cmds.append(["make", "mineru-service-install"])
    return cmds


def _default_runner(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _note_foreground_mineru() -> None:
    from rich.console import Console

    Console().print(
        "[yellow]No background service available.[/yellow] Start MinerU in another "
        "terminal with:  make mineru-api"
    )


def run_mineru_step(
    prompter: Prompter,
    info: PlatformInfo,
    *,
    has_gpu: bool,
    llm_title_aided: dict | None = None,
    runner=_default_runner,
    require_confirm: bool = False,
) -> bool:
    choice = prompter.select(
        "Document parser (MinerU)",
        ["Local (recommended)", "Official cloud API"],
    )
    if choice == "Official cloud API":
        token = prompter.password("MinerU API token")
        if require_confirm and not prompter.confirm(MINERU_OVERWRITE_CONFIRM, default=False):
            return False
        configure_mineru_official(token)
        return True

    extras = select_mineru_extras(info, has_gpu=has_gpu)
    title_aided = None
    if (
        llm_title_aided
        and llm_title_aided.get("base_url")
        and prompter.confirm("Improve heading detection with your LLM (title-aided)?", default=True)
    ):
        title_aided = llm_title_aided
    # Confirm AFTER collecting choices, right before overwriting existing settings.
    if require_confirm and not prompter.confirm(MINERU_OVERWRITE_CONFIRM, default=False):
        return False
    configure_mineru_local_env(extras, title_aided=title_aided)

    service_model = resolve_service_model(info, systemd_available=systemd_user_available())
    for cmd in build_mineru_local_commands(service_model, title_aided=bool(title_aided)):
        runner(cmd)
    if service_model == "foreground":
        _note_foreground_mineru()
    return True


# ---------------------------------------------------------------------------
# Interactive Models step (thin shell over the pure logic above)
# ---------------------------------------------------------------------------
class Prompter:
    """Minimal prompt surface; the questionary impl is created lazily in main()."""

    def select(self, message: str, choices: list[str]) -> str:
        raise NotImplementedError

    def text(self, message: str, default: str = "") -> str:
        raise NotImplementedError

    def password(self, message: str) -> str:
        raise NotImplementedError

    def confirm(self, message: str, default: bool = False) -> bool:
        raise NotImplementedError


def _ask_model(
    prompter: Prompter, providers: dict[str, ProviderSpec], role_label: str
) -> tuple[str, str, str | None, str]:
    name = prompter.select(f"{role_label} provider", list(providers))
    spec = providers[name]
    model = prompter.text(f"{role_label} model name", default="")
    if spec.requires_url:
        base_url = prompter.text(f"{role_label} base URL (required for {name})", default="")
    else:
        base_url = prompter.text(f"{role_label} base URL", default=spec.base_url or "")
    key = prompter.password(f"{role_label} API key")
    return name, model, (base_url or None), key


def run_models_step(prompter: Prompter, *, require_confirm: bool = False) -> dict | None:
    env_values: dict[str, str] = {}

    from rich.console import Console

    Console().print(
        "[dim]Provider = API protocol (openai / anthropic / gemini). Pick your vendor below — "
        "DeepSeek, OpenRouter, Azure, etc. map to the OpenAI-compatible protocol automatically.[/dim]"
    )

    name, model, base_url, key = _ask_model(prompter, PROVIDERS_LLM, "LLM")
    llm_block, llm_env = resolve_llm_choice(name, model=model, base_url=base_url)
    env_values[llm_env] = key

    llm_roles: dict[str, dict] = {}
    if prompter.confirm("Use a cheaper model for extraction/keyword? (advanced)", default=False):
        for role, env_key in (
            ("extract", "DLIGHTRAG_LLM__ROLES__EXTRACT__API_KEY"),
            ("keyword", "DLIGHTRAG_LLM__ROLES__KEYWORD__API_KEY"),
        ):
            rn, rm, rurl, rk = _ask_model(prompter, PROVIDERS_LLM, f"{role} LLM")
            block, _ = resolve_llm_choice(rn, model=rm, base_url=rurl)
            llm_roles[role] = block
            env_values[env_key] = rk

    ename, emodel, ebase, ekey = _ask_model(prompter, PROVIDERS_EMBED, "Embedding")
    embed_block, embed_env = resolve_embedding_choice(ename, model=emodel, base_url=ebase)
    if embed_block["dim"] == 0:
        embed_block["dim"] = int(prompter.text("Embedding dimension (dim)", default="1024"))
    env_values[embed_env] = ekey

    rerank_choice = prompter.select("Reranker", list(RERANK_CHOICES))
    rerank_block, rerank_env = resolve_rerank_choice(rerank_choice)
    if rerank_env is not None:
        env_values[rerank_env] = prompter.password("Reranker API key")

    # Confirm AFTER collecting answers, right before overwriting existing settings.
    if require_confirm and not prompter.confirm(MODELS_OVERWRITE_CONFIRM, default=False):
        return None

    # Back up config and any pre-existing .env only now — right before writing —
    # so aborted or declined runs never leave a stray .bak behind.
    config_backup = backup_file(CONFIG_PATH)
    if ENV_PATH.exists():
        backup_file(ENV_PATH)
    elif ENV_EXAMPLE_PATH.exists():
        ENV_PATH.write_bytes(ENV_EXAMPLE_PATH.read_bytes())

    write_config_yaml(
        CONFIG_PATH,
        llm_default=llm_block,
        llm_roles=llm_roles,
        embedding=embed_block,
        rerank=rerank_block,
    )
    upsert_env(ENV_PATH, env_values)
    return {
        "llm": {"api_key": key, "base_url": llm_block.get("base_url"), "model": model},
        "config_backup": config_backup,
    }


class SetupCancelled(Exception):
    """Raised when the user picks the in-menu Quit option (a clean, non-error exit)."""


QUIT_CHOICE = "✕ Quit · 退出"


# Home menu — shown only when DlightRAG is already configured (see is_configured).
MENU_START = "Start DlightRAG · 启动"
MENU_CHANGE = "Change settings · 修改设置"
MENU_SHOW = "Show settings · 查看设置"
MENU_RESET = "Start over · 重新配置"
HOME_CHOICES = [MENU_START, MENU_CHANGE, MENU_SHOW, MENU_RESET]
HOME_PROMPT = "DlightRAG is already set up — what next? · DlightRAG 已配置，接下来做什么？"

# "Change settings" sub-menu (section-level, per the design).
SEC_MODELS = "Models & API keys · 模型与密钥"
SEC_MINERU = "Document parsing (MinerU) · 文档解析"
SEC_ALL = "Everything · 全部"
SEC_BACK = "← Back · 返回"
CHANGE_CHOICES = [SEC_MODELS, SEC_MINERU, SEC_ALL, SEC_BACK]
CHANGE_PROMPT = "What do you want to change? · 你想修改什么？"

MODELS_OVERWRITE_CONFIRM = (
    "Overwrite your current model settings and API keys with these answers? · "
    "用这些答案覆盖当前的模型设置与密钥？"
)
MINERU_OVERWRITE_CONFIRM = (
    "Overwrite your current document-parsing (MinerU) settings? · 覆盖当前的文档解析设置？"
)
RESET_WIPE_CONFIRM = (
    "Delete ALL documents you've already added (vectors, graph)? This cannot be undone. · "
    "删除所有已导入的文档（向量、图谱）？此操作不可恢复"
)

REQUIRED_ENV_KEYS = (
    "DLIGHTRAG_LLM__DEFAULT__API_KEY",
    "DLIGHTRAG_EMBEDDING__API_KEY",
)


def with_quit(choices: list[str]) -> list[str]:
    """Append the Quit sentinel so every menu offers a no-Ctrl+C way out."""
    return [*choices, QUIT_CHOICE]


def check_quit(answer: str) -> str:
    """Turn a Quit selection into a clean SetupCancelled; pass everything else through."""
    if answer == QUIT_CHOICE:
        raise SetupCancelled
    return answer


def _questionary_prompter() -> Prompter:
    import questionary

    # unsafe_ask(): let Ctrl+C / Ctrl+D propagate as KeyboardInterrupt / EOFError
    # (caught once in main) instead of silently returning None and crashing later.
    class _Q(Prompter):
        def select(self, message: str, choices: list[str]) -> str:
            return check_quit(questionary.select(message, choices=with_quit(choices)).unsafe_ask())

        def text(self, message: str, default: str = "") -> str:
            return questionary.text(message, default=default).unsafe_ask()

        def password(self, message: str) -> str:
            return questionary.password(message).unsafe_ask()

        def confirm(self, message: str, default: bool = False) -> bool:
            return questionary.confirm(message, default=default).unsafe_ask()

    return _Q()


# ---------------------------------------------------------------------------
# Docker bring-up + health poll
# ---------------------------------------------------------------------------
def docker_up_command() -> list[str]:
    return ["docker", "compose", "up", "-d"]


def probe_health(url: str, *, opener=None) -> bool:
    import urllib.request

    opener = opener or urllib.request.urlopen
    try:
        with opener(url, timeout=5) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def wait_for_health(url: str, *, attempts=60, delay=2.0, probe=probe_health, sleep=None) -> bool:
    sleep = sleep or time.sleep
    for i in range(attempts):
        if probe(url):
            return True
        if i < attempts - 1:
            sleep(delay)
    return False


# ---------------------------------------------------------------------------
# Re-run menu: view / change / start over (shown only when already configured)
# ---------------------------------------------------------------------------
def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if "=" in raw and not raw.lstrip().startswith("#"):
                key, value = raw.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def is_configured(env_path: Path | None = None) -> bool:
    """True when .env already holds the required API keys (a prior successful setup)."""
    env = _read_env(env_path or ENV_PATH)
    return all(env.get(key) for key in REQUIRED_ENV_KEYS)


def read_config_summary(config_path: Path, env_path: Path) -> dict:
    """Build a display-ready, masked summary of the current config (never secrets)."""
    data = _yaml().load(config_path)
    env = _read_env(env_path)
    llm = data.get("llm", {}) or {}
    default = llm.get("default", {}) or {}
    roles = llm.get("roles", {}) or {}
    embedding = data.get("embedding", {}) or {}
    rerank = data.get("rerank", {}) or {}
    mineru = (data.get("parser_sidecars", {}) or {}).get("mineru", {}) or {}
    return {
        "llm_default": {
            "provider": default.get("provider", "?"),
            "model": default.get("model", "?"),
            "base_url": default.get("base_url"),
        },
        "llm_roles": {
            role: {
                "provider": (block or {}).get("provider", "?"),
                "model": (block or {}).get("model", "?"),
                "base_url": (block or {}).get("base_url"),
            }
            for role, block in roles.items()
        },
        "embedding": {
            "provider": embedding.get("provider", "?"),
            "model": embedding.get("model", "?"),
            "dim": embedding.get("dim", "?"),
            "base_url": embedding.get("base_url"),
        },
        "rerank": {
            "strategy": rerank.get("strategy", "?"),
            "enabled": bool(rerank.get("enabled", False)),
            "model": rerank.get("model"),
            "base_url": rerank.get("base_url"),
        },
        "mineru_mode": mineru.get("api_mode", "?"),
        "workspace": data.get("workspace", "default"),
        "keys_set": {
            "LLM": bool(env.get("DLIGHTRAG_LLM__DEFAULT__API_KEY")),
            "Embedding": bool(env.get("DLIGHTRAG_EMBEDDING__API_KEY")),
            "Rerank": bool(env.get("DLIGHTRAG_RERANK__API_KEY")),
        },
    }


def render_summary(console, summary: dict) -> None:
    from rich.table import Table

    table = Table(title="Current settings · 当前配置", show_header=False)
    default = summary["llm_default"]
    table.add_row("LLM", f"{default['provider']} · {default['model']}")
    if default.get("base_url"):
        table.add_row("", f"[dim]{default['base_url']}[/dim]")
    for role, block in summary["llm_roles"].items():
        table.add_row(f"  • {role}", f"{block['provider']} · {block['model']}")
        if block.get("base_url"):
            table.add_row("", f"[dim]{block['base_url']}[/dim]")
    embedding = summary["embedding"]
    table.add_row(
        "Embedding", f"{embedding['provider']} · {embedding['model']} (dim {embedding['dim']})"
    )
    if embedding.get("base_url"):
        table.add_row("", f"[dim]{embedding['base_url']}[/dim]")
    rerank = summary["rerank"]
    rerank_model = f" · {rerank['model']}" if rerank.get("model") else ""
    rerank_state = "on" if rerank["enabled"] else "off"
    table.add_row("Rerank", f"{rerank['strategy']}{rerank_model} ({rerank_state})")
    if rerank.get("base_url"):
        table.add_row("", f"[dim]{rerank['base_url']}[/dim]")
    table.add_row("MinerU", summary["mineru_mode"])
    table.add_row("Workspace", summary["workspace"])
    table.add_row(
        "API keys",
        "   ".join(
            f"{name}: {'set ✓' if ok else 'missing ✗'}" for name, ok in summary["keys_set"].items()
        ),
    )
    console.print(table)


def _bring_up_stack(console) -> int:
    console.print("Starting DlightRAG + PostgreSQL… · 正在启动…")
    try:
        _default_runner(docker_up_command())
    except Exception as exc:
        console.print(f"[red]docker compose up failed:[/red] {exc}")
        return 1
    if wait_for_health(API_HEALTH_URL):
        console.print(f"[green]Ready![/green] Open [link={WEB_URL}]{WEB_URL}[/link] · 已就绪")
    else:
        console.print(
            f"[yellow]Not healthy yet — check `docker compose ps`, then open[/yellow] {WEB_URL}"
        )
    return 0


def _apply_and_validate(console, result: dict) -> bool:
    """Validate the freshly written config; restore backup and report on failure."""
    try:
        validate_config()
        return True
    except Exception as exc:
        backup = result.get("config_backup")
        if backup is not None:
            CONFIG_PATH.write_bytes(backup.read_bytes())
        console.print(f"[red]Config invalid; restored backup:[/red] {exc}")
        return False


def run_first_time_setup(
    console, prompter: Prompter, info: PlatformInfo, *, require_confirm: bool = False
) -> int | None:
    result = run_models_step(prompter, require_confirm=require_confirm)
    if result is None:
        console.print("No changes made. · 未做任何更改")
        return None
    if not _apply_and_validate(console, result):
        return 1
    run_mineru_step(
        prompter,
        info,
        has_gpu=has_nvidia_gpu(),
        llm_title_aided=result["llm"],
        require_confirm=require_confirm,
    )
    return _bring_up_stack(console)


def run_change_settings(console, prompter: Prompter, info: PlatformInfo) -> None:
    section = prompter.select(CHANGE_PROMPT, CHANGE_CHOICES)
    if section == SEC_BACK:
        return
    changed = False
    result = None
    if section in (SEC_MODELS, SEC_ALL):
        result = run_models_step(prompter, require_confirm=True)
        if result is None:
            console.print("No changes made. · 未做任何更改")
            return
        if not _apply_and_validate(console, result):
            return
        changed = True
    if section in (SEC_MINERU, SEC_ALL):
        if run_mineru_step(
            prompter,
            info,
            has_gpu=has_nvidia_gpu(),
            llm_title_aided=result["llm"] if result else None,
            require_confirm=True,
        ):
            changed = True
    if changed:
        console.print(
            "[green]Saved.[/green] Pick 'Start DlightRAG' to (re)launch. · 已保存，选择“启动”重新运行"
        )
    else:
        console.print("No changes made. · 未做任何更改")


def _wipe_data(console, *, runner=_default_runner) -> None:
    console.print("Erasing ingested data… · 正在清除已导入数据…")
    try:
        runner(["uv", "run", "scripts/reset.py", "--all", "-y"])
    except Exception as exc:
        console.print(
            f"[yellow]Couldn't erase data automatically ({exc}); "
            f"run `uv run scripts/reset.py --all` yourself.[/yellow]"
        )


def run_start_over(console, prompter: Prompter, info: PlatformInfo) -> int | None:
    console.print(
        "[bold]Start over[/bold] — re-enter settings; nothing changes until you confirm. · "
        "[bold]重新配置[/bold]：重新输入设置，确认前不会改动"
    )
    rc = run_first_time_setup(console, prompter, info, require_confirm=True)
    if rc != 0:  # None (declined the overwrite) or 1 (invalid config)
        return rc
    # Confirm the irreversible data wipe immediately before doing it.
    if prompter.confirm(RESET_WIPE_CONFIRM, default=False):
        _wipe_data(console)
    return rc


def run_home(console, prompter: Prompter, info: PlatformInfo) -> int:
    while True:
        choice = prompter.select(HOME_PROMPT, HOME_CHOICES)
        if choice == MENU_START:
            return _bring_up_stack(console)
        if choice == MENU_RESET:
            rc = run_start_over(console, prompter, info)
            if rc is not None:
                return rc
        elif choice == MENU_CHANGE:
            run_change_settings(console, prompter, info)
        elif choice == MENU_SHOW:
            render_summary(console, read_config_summary(CONFIG_PATH, ENV_PATH))


def main(prompter: Prompter | None = None) -> int:
    from rich.console import Console

    console = Console()
    console.rule("[bold]DlightRAG setup")
    console.print("[dim]Pick '✕ Quit · 退出' in any menu, or press Ctrl+C, to cancel.[/dim]")
    failed = [c for c in run_preflight() if not c.ok]
    for c in failed:
        console.print(f"[red]missing[/red] {c.name} — {c.hint}")
    if failed:
        return 1

    prompter = prompter or _questionary_prompter()
    info = detect_platform()
    try:
        if is_configured():
            return run_home(console, prompter, info)
        rc = run_first_time_setup(console, prompter, info)
        return 0 if rc is None else rc
    except KeyboardInterrupt, EOFError, SetupCancelled:
        console.print(
            "\n[yellow]Setup cancelled.[/yellow] Re-run any time with "
            "[bold]uv run prerequisite_setup.py[/bold]"
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
