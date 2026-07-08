# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the prerequisite_setup.py onboarding wizard (Plan 1)."""

import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def wiz():
    spec = importlib.util.spec_from_file_location(
        "prerequisite_setup", _ROOT / "prerequisite_setup.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/typing annotation resolution can find it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ScriptedPrompter:
    """Feeds pre-scripted answers to the Models step without a TTY."""

    def __init__(self, answers):
        self._a = list(answers)

    def select(self, message, choices):
        return self._a.pop(0)

    def text(self, message, default=""):
        value = self._a.pop(0)
        return value if value != "" else default

    def password(self, message):
        return self._a.pop(0)

    def confirm(self, message, default=False):
        return self._a.pop(0)


def test_module_imports(wiz):
    assert wiz.CONFIG_PATH.name == "config.yaml"
    assert wiz.ENV_PATH.name == ".env"


# --- Task 2: provider registry / resolvers --------------------------------
def test_llm_openai_compatible_mapping(wiz):
    block, env_key = wiz.resolve_llm_choice("DeepSeek", model="deepseek-v4-flash", base_url=None)
    assert block["provider"] == "openai"
    assert block["base_url"] == "https://api.deepseek.com"
    assert block["model"] == "deepseek-v4-flash"
    assert env_key == "DLIGHTRAG_LLM__DEFAULT__API_KEY"


def test_llm_native_provider_has_no_base_url(wiz):
    block, _ = wiz.resolve_llm_choice("Anthropic", model="claude-4", base_url=None)
    assert block["provider"] == "anthropic"
    assert "base_url" not in block


def test_llm_azure_requires_user_base_url(wiz):
    assert wiz.PROVIDERS_LLM["Azure OpenAI"].requires_url is True
    block, _ = wiz.resolve_llm_choice(
        "Azure OpenAI", model="gpt-4o", base_url="https://x.openai.azure.com/v1"
    )
    assert block["base_url"] == "https://x.openai.azure.com/v1"


def test_embedding_mapping_prefills_dim(wiz):
    block, env_key = wiz.resolve_embedding_choice(
        "Voyage", model="voyage-multimodal-3.5", base_url=None
    )
    assert block["provider"] == "voyage"
    assert block["dim"] == 1024
    assert env_key == "DLIGHTRAG_EMBEDDING__API_KEY"


def test_rerank_reuse_llm_needs_no_key(wiz):
    block, env_key = wiz.resolve_rerank_choice("Reuse my LLM")
    assert block["strategy"] == "chat_llm_reranker"
    assert env_key is None


# --- Task 3: config.yaml writer -------------------------------------------
def test_write_config_preserves_comments_and_updates(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "# curated header comment\n"
        "llm:\n"
        "  default:\n"
        "    provider: openai  # inline note\n"
        "    model: old-model\n"
        "    base_url: https://old\n"
        "embedding:\n"
        "  provider: voyage\n"
        "  model: old-embed\n"
        "  dim: 1024\n",
        encoding="utf-8",
    )
    wiz.write_config_yaml(
        src,
        llm_default={
            "provider": "openai",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
        },
        embedding={
            "provider": "voyage",
            "model": "voyage-multimodal-3.5",
            "base_url": "https://api.voyageai.com/v1",
            "dim": 1024,
        },
    )
    text = src.read_text(encoding="utf-8")
    assert "# curated header comment" in text
    assert "# inline note" in text
    assert "deepseek-v4-flash" in text
    assert "old-model" not in text


# --- Task 4: .env upsert ---------------------------------------------------
def test_upsert_env_preserves_and_updates(wiz, tmp_path):
    env = tmp_path / ".env"
    env.write_text("EXISTING=keep\nDLIGHTRAG_LLM__DEFAULT__API_KEY=old\n", encoding="utf-8")
    wiz.upsert_env(
        env,
        {"DLIGHTRAG_LLM__DEFAULT__API_KEY": "new", "DLIGHTRAG_EMBEDDING__API_KEY": "e"},
    )
    lines = env.read_text(encoding="utf-8").splitlines()
    assert "EXISTING=keep" in lines
    assert "DLIGHTRAG_LLM__DEFAULT__API_KEY=new" in lines
    assert "DLIGHTRAG_EMBEDDING__API_KEY=e" in lines
    assert sum(line.startswith("DLIGHTRAG_LLM__DEFAULT__API_KEY=") for line in lines) == 1


def test_upsert_env_creates_from_missing(wiz, tmp_path):
    env = tmp_path / ".env"
    wiz.upsert_env(env, {"K": "v"})
    assert env.read_text(encoding="utf-8").strip() == "K=v"


# --- Task 5: backup --------------------------------------------------------
def test_backup_file_creates_timestamped_copy(wiz, tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("x: 1\n", encoding="utf-8")
    backup = wiz.backup_file(f)
    assert backup is not None
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "x: 1\n"
    assert backup.name.startswith("config.yaml.bak-")


def test_backup_file_missing_returns_none(wiz, tmp_path):
    assert wiz.backup_file(tmp_path / "nope.yaml") is None


def test_backup_file_keeps_only_latest(wiz, tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("current\n", encoding="utf-8")
    # Two pre-existing older backups (older timestamps sort first).
    (tmp_path / "config.yaml.bak-20200101000000").write_text("old0\n", encoding="utf-8")
    (tmp_path / "config.yaml.bak-20200101000001").write_text("old1\n", encoding="utf-8")
    backup = wiz.backup_file(f)
    assert backup is not None
    remaining = sorted(p.name for p in tmp_path.glob("config.yaml.bak-*"))
    assert remaining == [backup.name]  # only the freshly-created backup survives
    assert backup.read_text(encoding="utf-8") == "current\n"


# --- Task 6: detection + preflight ----------------------------------------
def test_detect_platform_apple_silicon(wiz, monkeypatch):
    monkeypatch.setattr(wiz.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(wiz.platform, "machine", lambda: "arm64")
    info = wiz.detect_platform()
    assert info.os == "macos"
    assert info.arch == "arm64"
    assert info.is_wsl is False


def test_detect_platform_linux_wsl(wiz, monkeypatch, tmp_path):
    monkeypatch.setattr(wiz.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wiz.platform, "machine", lambda: "x86_64")
    proc = tmp_path / "version"
    proc.write_text("Linux version 5.x microsoft-standard-WSL2", encoding="utf-8")
    monkeypatch.setattr(wiz, "_PROC_VERSION", proc)
    info = wiz.detect_platform()
    assert info.os == "linux"
    assert info.is_wsl is True


def test_has_nvidia_gpu(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    assert wiz.has_nvidia_gpu() is True


def test_preflight_flags_missing_tool(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz.shutil, "which", lambda name: None if name == "docker" else "/usr/bin/" + name
    )
    checks = wiz.run_preflight()
    docker = next(c for c in checks if c.name == "docker")
    assert docker.ok is False


# --- Task 7: MinerU extras + service model --------------------------------
@pytest.mark.parametrize(
    "os_name,arch,gpu,expected",
    [
        ("macos", "arm64", False, "core,mlx"),
        ("linux", "x86_64", True, "core,vllm"),
        ("linux", "x86_64", False, "core"),
    ],
)
def test_select_mineru_extras(wiz, os_name, arch, gpu, expected):
    info = wiz.PlatformInfo(os=os_name, arch=arch, is_wsl=False)
    assert wiz.select_mineru_extras(info, has_gpu=gpu) == expected


@pytest.mark.parametrize(
    "os_name,is_wsl,systemd,expected",
    [
        ("macos", False, False, "launchd"),
        ("linux", False, True, "systemd-user"),
        ("linux", True, True, "systemd-user"),
        ("linux", False, False, "foreground"),
    ],
)
def test_resolve_service_model(wiz, os_name, is_wsl, systemd, expected):
    info = wiz.PlatformInfo(os=os_name, arch="x86_64", is_wsl=is_wsl)
    assert wiz.resolve_service_model(info, systemd_available=systemd) == expected


# --- Task 8: interactive Models step --------------------------------------
def test_models_step_writes_config_and_env(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n  default:\n    provider: openai\n    model: x\n    base_url: https://x\n"
        "embedding:\n  provider: voyage\n  model: x\n  dim: 1024\n"
        "rerank:\n  strategy: voyage_reranker\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing.env.example")
    prompter = _ScriptedPrompter(
        [
            "DeepSeek",
            "deepseek-v4-flash",
            "",
            "sk-llm",  # LLM: provider, model, base_url(default), key
            False,  # advanced roles? no
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",  # embedding: provider, model, base_url(default), key
            "Reuse my LLM",  # rerank
        ]
    )
    wiz.run_models_step(prompter)
    text = cfg.read_text(encoding="utf-8")
    assert "deepseek-v4-flash" in text
    assert "api.deepseek.com" in text
    assert "chat_llm_reranker" in text
    env_text = env.read_text(encoding="utf-8")
    assert "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-llm" in env_text
    assert "DLIGHTRAG_EMBEDDING__API_KEY=sk-embed" in env_text


# --- Plan 2 Task 1: MinerU config helpers ---------------------------------
def test_configure_mineru_official(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("parser_sidecars:\n  mineru:\n    api_mode: local\n", encoding="utf-8")
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    wiz.configure_mineru_official("tok-123")
    assert "api_mode: official" in cfg.read_text(encoding="utf-8")
    assert "DLIGHTRAG_PARSER_SIDECARS__MINERU__API_TOKEN=tok-123" in env.read_text(encoding="utf-8")


def test_configure_mineru_local_env_writes_extras_and_title_aided(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("parser_sidecars:\n  mineru:\n    api_mode: official\n", encoding="utf-8")
    mineru_env = tmp_path / ".env.mineru"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", mineru_env)
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    wiz.configure_mineru_local_env(
        "core,mlx",
        title_aided={
            "api_key": "sk",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
        },
    )
    text = mineru_env.read_text(encoding="utf-8")
    assert "MINERU_INSTALL_EXTRAS=core,mlx" in text
    assert "MINERU_TITLE_AIDED_ENABLE=true" in text
    assert "MINERU_TITLE_AIDED_MODEL=deepseek-v4-flash" in text
    assert "api_mode: local" in cfg.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "service_model,title_aided,expected",
    [
        ("launchd", False, [["make", "mineru-install"], ["make", "mineru-service-install"]]),
        (
            "systemd-user",
            True,
            [
                ["make", "mineru-install"],
                ["make", "mineru-title-aided"],
                ["make", "mineru-service-install"],
            ],
        ),
        ("foreground", False, [["make", "mineru-install"]]),
    ],
)
def test_build_mineru_local_commands(wiz, service_model, title_aided, expected):
    assert wiz.build_mineru_local_commands(service_model, title_aided=title_aided) == expected


# --- Plan 2 Task 2: run_mineru_step fork ----------------------------------
def test_run_mineru_step_official(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("parser_sidecars:\n  mineru:\n    api_mode: local\n", encoding="utf-8")
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    ran: list = []
    info = wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)
    prompter = _ScriptedPrompter(["Official cloud API", "tok-xyz"])
    wiz.run_mineru_step(prompter, info, has_gpu=False, runner=lambda cmd: ran.append(cmd))
    assert "api_mode: official" in cfg.read_text(encoding="utf-8")
    assert ran == []


def test_run_mineru_step_local_runs_commands(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("parser_sidecars:\n  mineru:\n    api_mode: official\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", tmp_path / ".env.mineru")
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    monkeypatch.setattr(wiz, "systemd_user_available", lambda: False)
    ran: list = []
    info = wiz.PlatformInfo(os="macos", arch="arm64", is_wsl=False)
    prompter = _ScriptedPrompter(["Local (recommended)", False])
    wiz.run_mineru_step(prompter, info, has_gpu=False, runner=lambda cmd: ran.append(cmd))
    assert ["make", "mineru-install"] in ran
    assert ["make", "mineru-service-install"] in ran
    assert "MINERU_INSTALL_EXTRAS=core,mlx" in (tmp_path / ".env.mineru").read_text(
        encoding="utf-8"
    )


# --- Plan 3: creds return, title-aided reuse, docker bring-up --------------
def test_models_step_returns_llm_creds(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n  default:\n    provider: openai\n    model: x\n    base_url: https://x\n"
        "embedding:\n  provider: voyage\n  model: x\n  dim: 1024\n"
        "rerank:\n  strategy: voyage_reranker\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    prompter = _ScriptedPrompter(
        [
            "DeepSeek",
            "deepseek-v4-flash",
            "",
            "sk-llm",
            False,
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",
            "Reuse my LLM",
        ]
    )
    result = wiz.run_models_step(prompter)
    assert result["llm"] == {
        "api_key": "sk-llm",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    }


def test_run_mineru_step_local_title_aided(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("parser_sidecars:\n  mineru:\n    api_mode: official\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", tmp_path / ".env.mineru")
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    monkeypatch.setattr(wiz, "systemd_user_available", lambda: True)
    ran: list = []
    info = wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)
    prompter = _ScriptedPrompter(["Local (recommended)", True])
    creds = {
        "api_key": "sk",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    }
    wiz.run_mineru_step(
        prompter, info, has_gpu=False, llm_title_aided=creds, runner=lambda cmd: ran.append(cmd)
    )
    assert ["make", "mineru-title-aided"] in ran
    assert "MINERU_TITLE_AIDED_MODEL=deepseek-v4-flash" in (tmp_path / ".env.mineru").read_text(
        encoding="utf-8"
    )


def test_docker_up_command(wiz):
    assert wiz.docker_up_command() == ["docker", "compose", "up", "-d"]


def test_wait_for_health_success(wiz):
    calls = {"n": 0}

    def probe(url):
        calls["n"] += 1
        return calls["n"] >= 2

    assert wiz.wait_for_health("u", attempts=5, delay=0, probe=probe, sleep=lambda _: None) is True


def test_wait_for_health_gives_up(wiz):
    assert (
        wiz.wait_for_health("u", attempts=3, delay=0, probe=lambda url: False, sleep=lambda _: None)
        is False
    )


def test_with_quit_appends_sentinel(wiz):
    assert wiz.with_quit(["A", "B"]) == ["A", "B", wiz.QUIT_CHOICE]


def test_check_quit_passes_through_normal_answer(wiz):
    assert wiz.check_quit("A") == "A"


def test_check_quit_raises_setup_cancelled(wiz):
    with pytest.raises(wiz.SetupCancelled):
        wiz.check_quit(wiz.QUIT_CHOICE)


@pytest.mark.parametrize("kind", ["ctrl_c", "ctrl_d", "menu_quit"])
def test_main_cancel_exits_cleanly(wiz, tmp_path, monkeypatch, kind):
    """Ctrl+C, Ctrl+D, or the in-menu Quit all exit 130 with no traceback."""
    monkeypatch.setattr(wiz, "run_preflight", lambda: [])  # pretend all tools present
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  default:\n    provider: openai\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    exc = {"ctrl_c": KeyboardInterrupt, "ctrl_d": EOFError, "menu_quit": wiz.SetupCancelled}[kind]

    class _Cancel(wiz.Prompter):
        def select(self, message, choices):
            raise exc

        def text(self, message, default=""):
            raise exc

        def password(self, message):
            raise exc

        def confirm(self, message, default=False):
            raise exc

    assert wiz.main(prompter=_Cancel()) == 130


# --- §11 re-run menu: detection / summary / dispatch ----------------------
class _NullConsole:
    def print(self, *args, **kwargs):
        pass

    def rule(self, *args, **kwargs):
        pass


def _info(wiz):
    return wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)


def test_is_configured_true_when_keys_present(wiz, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-a\nDLIGHTRAG_EMBEDDING__API_KEY=sk-b\n",
        encoding="utf-8",
    )
    assert wiz.is_configured(env) is True


def test_is_configured_false_when_missing_key(wiz, tmp_path):
    env = tmp_path / ".env"
    env.write_text("DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-a\n", encoding="utf-8")
    assert wiz.is_configured(env) is False


def test_is_configured_false_when_no_env(wiz, tmp_path):
    assert wiz.is_configured(tmp_path / "missing.env") is False


def test_read_config_summary_masks_secrets_and_extracts(wiz, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n"
        "  default:\n    provider: openai\n    model: gpt-x\n    base_url: https://api.x\n"
        "  roles:\n    extract:\n      provider: openai\n      model: cheap\n"
        "      base_url: https://api.deepseek.com\n"
        "embedding:\n  provider: voyage\n  model: voyage-x\n  dim: 1024\n"
        "  base_url: https://api.voyageai.com/v1\n"
        "rerank:\n  enabled: true\n  strategy: voyage_reranker\n"
        "parser_sidecars:\n  mineru:\n    api_mode: local\n"
        "workspace: default\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-a\nDLIGHTRAG_EMBEDDING__API_KEY=sk-b\n",
        encoding="utf-8",
    )
    s = wiz.read_config_summary(cfg, env)
    assert s["llm_default"] == {"provider": "openai", "model": "gpt-x", "base_url": "https://api.x"}
    assert s["llm_roles"]["extract"] == {
        "provider": "openai",
        "model": "cheap",
        "base_url": "https://api.deepseek.com",
    }
    assert s["embedding"]["dim"] == 1024
    assert s["embedding"]["base_url"] == "https://api.voyageai.com/v1"
    assert s["rerank"] == {"strategy": "voyage_reranker", "enabled": True}
    assert s["mineru_mode"] == "local"
    assert s["workspace"] == "default"
    assert s["keys_set"] == {"LLM": True, "Embedding": True, "Rerank": False}
    assert "sk-a" not in repr(s) and "sk-b" not in repr(s)


def test_home_start_brings_up_stack(wiz, monkeypatch):
    ups: list = []
    monkeypatch.setattr(wiz, "_default_runner", lambda cmd: ups.append(cmd))
    monkeypatch.setattr(wiz, "wait_for_health", lambda url, **k: True)
    prompter = _ScriptedPrompter([wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert ups == [["docker", "compose", "up", "-d"]]


def test_home_show_then_start(wiz, monkeypatch):
    shown: list = []
    monkeypatch.setattr(wiz, "read_config_summary", lambda c, e: {"ok": True})
    monkeypatch.setattr(wiz, "render_summary", lambda console, s: shown.append(s))
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: 0)
    prompter = _ScriptedPrompter([wiz.MENU_SHOW, wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert shown == [{"ok": True}]


def test_home_change_then_start(wiz, monkeypatch):
    changed: list = []
    monkeypatch.setattr(wiz, "run_change_settings", lambda console, p, info: changed.append(True))
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: 0)
    prompter = _ScriptedPrompter([wiz.MENU_CHANGE, wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert changed == [True]


def test_home_reset_without_wipe(wiz, monkeypatch):
    wiped: list = []
    monkeypatch.setattr(wiz, "run_first_time_setup", lambda console, p, info, **k: 0)
    monkeypatch.setattr(wiz, "_wipe_data", lambda console, **k: wiped.append(True))
    prompter = _ScriptedPrompter([wiz.MENU_RESET, False])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert wiped == []


def test_home_reset_with_wipe(wiz, monkeypatch):
    wiped: list = []
    monkeypatch.setattr(wiz, "run_first_time_setup", lambda console, p, info, **k: 0)
    monkeypatch.setattr(wiz, "_wipe_data", lambda console, **k: wiped.append(True))
    prompter = _ScriptedPrompter([wiz.MENU_RESET, True])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert wiped == [True]


def test_home_reset_declined_returns_to_menu(wiz, monkeypatch):
    monkeypatch.setattr(wiz, "run_first_time_setup", lambda console, p, info, **k: None)
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: 0)
    wiped: list = []
    monkeypatch.setattr(wiz, "_wipe_data", lambda console, **k: wiped.append(True))
    prompter = _ScriptedPrompter([wiz.MENU_RESET, wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert wiped == []


def test_change_models_only(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz, "run_models_step", lambda p, **k: {"llm": {"base_url": "u"}, "config_backup": None}
    )
    monkeypatch.setattr(wiz, "validate_config", lambda: None)
    mineru: list = []
    monkeypatch.setattr(wiz, "run_mineru_step", lambda *a, **k: mineru.append(True) or True)
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_MODELS]), _info(wiz))
    assert mineru == []


def test_change_mineru_only(wiz, monkeypatch):
    models: list = []
    monkeypatch.setattr(wiz, "run_models_step", lambda p, **k: models.append(True))
    mineru: list = []
    monkeypatch.setattr(wiz, "run_mineru_step", lambda *a, **k: mineru.append(True) or True)
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_MINERU]), _info(wiz))
    assert models == []
    assert mineru == [True]


def test_change_everything_runs_both(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz, "run_models_step", lambda p, **k: {"llm": {"base_url": "u"}, "config_backup": None}
    )
    monkeypatch.setattr(wiz, "validate_config", lambda: None)
    mineru: list = []
    monkeypatch.setattr(wiz, "run_mineru_step", lambda *a, **k: mineru.append(True) or True)
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_ALL]), _info(wiz))
    assert mineru == [True]


def test_change_back_does_nothing(wiz, monkeypatch):
    touched: list = []
    monkeypatch.setattr(wiz, "run_models_step", lambda p, **k: touched.append("models"))
    monkeypatch.setattr(wiz, "run_mineru_step", lambda *a, **k: touched.append("mineru"))
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_BACK]), _info(wiz))
    assert touched == []


def test_change_models_declined_makes_no_change(wiz, monkeypatch):
    monkeypatch.setattr(wiz, "run_models_step", lambda p, **k: None)
    mineru: list = []
    monkeypatch.setattr(wiz, "run_mineru_step", lambda *a, **k: mineru.append(True) or True)
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_MODELS]), _info(wiz))
    assert mineru == []


_MODELS_ANSWERS = [
    "DeepSeek",
    "deepseek-v4-flash",
    "",
    "sk-llm",
    False,  # no advanced extract/keyword roles
    "Voyage",
    "voyage-multimodal-3.5",
    "",
    "sk-embed",
    "Reuse my LLM",
]


def _models_cfg(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n  default:\n    provider: openai\n    model: old\n    base_url: https://old\n"
        "embedding:\n  provider: voyage\n  model: old\n  dim: 1024\n"
        "rerank:\n  strategy: voyage_reranker\n",
        encoding="utf-8",
    )
    return cfg


def test_models_step_confirm_declined_leaves_config(wiz, tmp_path, monkeypatch):
    cfg = _models_cfg(tmp_path)
    original = cfg.read_text(encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    prompter = _ScriptedPrompter([*_MODELS_ANSWERS, False])  # decline the overwrite
    assert wiz.run_models_step(prompter, require_confirm=True) is None
    assert cfg.read_text(encoding="utf-8") == original


def test_models_step_confirm_accepted_writes(wiz, tmp_path, monkeypatch):
    cfg = _models_cfg(tmp_path)
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    prompter = _ScriptedPrompter([*_MODELS_ANSWERS, True])  # accept the overwrite
    assert wiz.run_models_step(prompter, require_confirm=True) is not None
    assert "deepseek-v4-flash" in cfg.read_text(encoding="utf-8")


def test_mineru_step_confirm_declined_skips_write(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("parser_sidecars:\n  mineru:\n    api_mode: official\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", tmp_path / ".env.mineru")
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    ran: list = []
    info = wiz.PlatformInfo(os="macos", arch="arm64", is_wsl=False)
    prompter = _ScriptedPrompter(["Local (recommended)", False])  # decline overwrite
    applied = wiz.run_mineru_step(
        prompter, info, has_gpu=False, runner=lambda c: ran.append(c), require_confirm=True
    )
    assert applied is False
    assert ran == []
    assert "api_mode: official" in cfg.read_text(encoding="utf-8")
