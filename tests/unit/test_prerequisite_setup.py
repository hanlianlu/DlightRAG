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
