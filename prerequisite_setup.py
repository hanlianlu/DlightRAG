#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
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
    if llm_roles:
        roles = data["llm"].setdefault("roles", {})
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
def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    backup.write_bytes(path.read_bytes())
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
) -> None:
    choice = prompter.select(
        "Document parser (MinerU)",
        ["Local (recommended)", "Official cloud API"],
    )
    if choice == "Official cloud API":
        token = prompter.password("MinerU API token")
        configure_mineru_official(token)
        return

    extras = select_mineru_extras(info, has_gpu=has_gpu)
    title_aided = None
    if (
        llm_title_aided
        and llm_title_aided.get("base_url")
        and prompter.confirm("Improve heading detection with your LLM (title-aided)?", default=True)
    ):
        title_aided = llm_title_aided
    configure_mineru_local_env(extras, title_aided=title_aided)

    service_model = resolve_service_model(info, systemd_available=systemd_user_available())
    for cmd in build_mineru_local_commands(service_model, title_aided=bool(title_aided)):
        runner(cmd)
    if service_model == "foreground":
        _note_foreground_mineru()


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


def run_models_step(prompter: Prompter) -> dict:
    config_backup = backup_file(CONFIG_PATH)
    backup_file(ENV_PATH)
    if not ENV_PATH.exists() and ENV_EXAMPLE_PATH.exists():
        ENV_PATH.write_bytes(ENV_EXAMPLE_PATH.read_bytes())

    env_values: dict[str, str] = {}

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

    write_config_yaml(
        CONFIG_PATH,
        llm_default=llm_block,
        llm_roles=llm_roles or None,
        embedding=embed_block,
        rerank=rerank_block,
    )
    upsert_env(ENV_PATH, env_values)
    return {
        "llm": {"api_key": key, "base_url": llm_block.get("base_url"), "model": model},
        "config_backup": config_backup,
    }


def _questionary_prompter() -> Prompter:
    import questionary

    class _Q(Prompter):
        def select(self, message: str, choices: list[str]) -> str:
            return questionary.select(message, choices=choices).ask()

        def text(self, message: str, default: str = "") -> str:
            return questionary.text(message, default=default).ask()

        def password(self, message: str) -> str:
            return questionary.password(message).ask()

        def confirm(self, message: str, default: bool = False) -> bool:
            return questionary.confirm(message, default=default).ask()

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


def main() -> int:
    from rich.console import Console

    console = Console()
    console.rule("[bold]DlightRAG setup")
    failed = [c for c in run_preflight() if not c.ok]
    for c in failed:
        console.print(f"[red]missing[/red] {c.name} — {c.hint}")
    if failed:
        return 1

    prompter = _questionary_prompter()
    info = detect_platform()
    result = run_models_step(prompter)
    try:
        validate_config()
    except Exception as exc:
        backup = result.get("config_backup")
        if backup is not None:
            CONFIG_PATH.write_bytes(backup.read_bytes())
        console.print(f"[red]Config invalid; restored backup:[/red] {exc}")
        return 1

    run_mineru_step(prompter, info, has_gpu=has_nvidia_gpu(), llm_title_aided=result["llm"])

    console.print("Starting DlightRAG + PostgreSQL…")
    try:
        _default_runner(docker_up_command())
    except Exception as exc:
        console.print(f"[red]docker compose up failed:[/red] {exc}")
        return 1

    if wait_for_health(API_HEALTH_URL):
        console.print(f"[green]Ready![/green] Open [link={WEB_URL}]{WEB_URL}[/link]")
    else:
        console.print(
            f"[yellow]Not healthy yet — check `docker compose ps`, then open[/yellow] {WEB_URL}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
