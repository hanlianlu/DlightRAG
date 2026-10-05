# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup validation for optional trusted local execution."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlightrag.application.config import (
    AgentBrowserConfig,
    AgentExecutionConfig,
    AgentMailboxConfig,
    AnswerSectionSettings,
    DlightragConfig,
    WebSourcesConfig,
)
from dlightrag.application.settings import agent_browser_settings
from dlightrag.engine.answer.execution_settings import (
    default_local_workspace_root,
    validate_agent_execution,
)


def test_execution_environment_defaults_to_trust() -> None:
    assert AgentExecutionConfig().execution_environment == "trust"


def test_child_concurrency_is_at_least_one() -> None:
    assert AgentExecutionConfig(child_concurrency=1).child_concurrency == 1
    with pytest.raises(ValidationError):
        AgentExecutionConfig(child_concurrency=0)


def test_disabled_ignores_workspace_root(tmp_path: Path) -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={
            "working_dir": str(tmp_path / "corpus"),
        },
        answer={
            "agent": AgentExecutionConfig(
                execution_environment="disabled", workspace_root=str(tmp_path / "ws")
            ),
        },
    )
    assert (
        validate_agent_execution(
            execution_environment=config.answer.agent.execution_environment,
            workspace_root=config.answer.agent.workspace_root,
            working_dir=config.deployment.working_dir,
        )
        is None
    )


def test_trust_without_root_uses_home_default(tmp_path: Path) -> None:
    resolved = validate_agent_execution(
        execution_environment="trust",
        workspace_root=None,
        working_dir=str(tmp_path / "corpus"),
    )
    assert resolved == default_local_workspace_root()
    assert resolved is not None
    assert resolved.is_dir()


def test_trust_rejects_a_relative_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        validate_agent_execution(
            execution_environment="trust",
            workspace_root="relative/workspaces",
            working_dir=str(tmp_path / "corpus"),
        )


def test_the_retired_sandbox_mode_is_refused_with_no_alias() -> None:
    """Stronger-than-the-kernel isolation left the application, so the name is gone.

    A configuration that still names it fails validation rather than selecting a mode
    that does nothing, and no alias accepts it in its place (ADR 0024).
    """
    with pytest.raises(ValidationError, match="execution_environment"):
        AgentExecutionConfig.model_validate({"execution_environment": "sandbox"})
    with pytest.raises(ValidationError, match="execution_environment"):
        AgentExecutionConfig.model_validate({"execution_environment": "isolated"})


def test_workspace_root_must_not_overlap_working_dir(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={
            "working_dir": str(corpus),
        },
        answer={
            "agent": AgentExecutionConfig(
                execution_environment="trust", workspace_root=str(corpus / "nested")
            ),
        },
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_agent_execution(
            execution_environment=config.answer.agent.execution_environment,
            workspace_root=config.answer.agent.workspace_root,
            working_dir=config.deployment.working_dir,
        )


@pytest.mark.parametrize(
    "servers",
    [[], [{"name": "docs", "transport": "stdio", "command": "unused", "tools": ["read"]}]],
)
def test_deployment_outbound_mcp_key_is_rejected(servers) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentExecutionConfig.model_validate({"outbound_mcp": servers})


def test_unknown_agent_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DlightragConfig.model_validate(
            {"agent": {"execution_environment": "disabled", "sandbox": True}}
        )


def test_skills_root_defaults_to_none_and_accepts_absolute() -> None:
    assert AgentExecutionConfig().skills_root is None
    config = AgentExecutionConfig.model_validate({"skills_root": "/opt/dlightrag/skills"})
    assert config.skills_root == "/opt/dlightrag/skills"


def test_skills_root_rejects_a_relative_path() -> None:
    with pytest.raises(ValidationError, match="skill roots must be absolute paths"):
        AgentExecutionConfig.model_validate({"skills_root": "skills"})


def test_owner_skills_root_defaults_to_none_and_accepts_absolute() -> None:
    assert AgentExecutionConfig().owner_skills_root is None
    config = AgentExecutionConfig.model_validate(
        {"owner_skills_root": "/opt/dlightrag/owner_skills"}
    )
    assert config.owner_skills_root == "/opt/dlightrag/owner_skills"


def test_owner_skills_root_rejects_a_relative_path() -> None:
    with pytest.raises(ValidationError, match="skill roots must be absolute paths"):
        AgentExecutionConfig.model_validate({"owner_skills_root": "owner_skills"})


def test_disabled_builtin_skills_defaults_empty_and_accepts_names() -> None:
    assert AgentExecutionConfig().disabled_builtin_skills == ()
    config = AgentExecutionConfig.model_validate(
        {"disabled_builtin_skills": ["skill-creator", "operator-disabled"]}
    )
    assert config.disabled_builtin_skills == ("skill-creator", "operator-disabled")


def test_search_tool_configuration_defaults_and_absolute_paths() -> None:
    default = AgentExecutionConfig()
    assert default.fd_path == "fd"
    assert default.ripgrep_path == "rg"
    assert default.search_tool_cache_root is None
    assert default.search_tool_auto_install is False

    configured = AgentExecutionConfig.model_validate(
        {
            "fd_path": "/opt/bin/fd",
            "ripgrep_path": "/opt/bin/rg",
            "search_tool_cache_root": "/var/cache/dlightrag-tools",
            "search_tool_auto_install": True,
        }
    )
    assert configured.search_tool_auto_install is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fd_path", "relative/bin/fd"),
        ("ripgrep_path", "relative/bin/rg"),
        ("search_tool_cache_root", "relative-cache"),
    ],
)
def test_search_tool_directories_must_be_absolute(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        AgentExecutionConfig.model_validate({field: value})


_POOL = {
    "endpoints": ("ws://agent-browser-1:3000/", "ws://agent-browser-2:3000/"),
    "egress_proxy": "http://agent-browser-egress:3128",
}


def test_the_agent_browser_pool_is_bound_through_the_environment_as_compose_binds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__ENDPOINTS",
        '["ws://agent-browser-1:3000/","ws://agent-browser-2:3000/"]',
    )
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__EGRESS_PROXY", "http://agent-browser-egress:3128"
    )

    browser = DlightragConfig().answer.agent.browser  # pyright: ignore[reportCallIssue]

    assert browser.endpoints == _POOL["endpoints"]
    assert browser.egress_proxy == _POOL["egress_proxy"]


@pytest.mark.parametrize(
    ("bound", "sandboxed"),
    [(None, True), ("true", True), ("false", False)],
    ids=["by-default", "on", "off"],
)
def test_chromiums_sandbox_is_on_unless_the_operator_turns_it_off_and_reaches_the_settings(
    monkeypatch: pytest.MonkeyPatch, bound: str | None, sandboxed: bool
) -> None:
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__ENDPOINTS", '["ws://agent-browser-1:3000/"]'
    )
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__EGRESS_PROXY", "http://agent-browser-egress:3128"
    )
    if bound is not None:
        monkeypatch.setenv("DLIGHTRAG_ANSWER__AGENT__BROWSER__CHROMIUM_SANDBOX", bound)

    config = DlightragConfig()  # pyright: ignore[reportCallIssue]
    settings = agent_browser_settings(config)

    assert config.answer.agent.browser.chromium_sandbox is sandboxed
    assert settings is not None and settings.chromium_sandbox is sandboxed


@pytest.mark.parametrize(
    ("bound", "registers"),
    [(None, True), ("true", True), ("false", False)],
    ids=["by-default", "on", "off"],
)
def test_the_agent_may_register_unless_the_operator_turns_registration_off(
    monkeypatch: pytest.MonkeyPatch, bound: str | None, registers: bool
) -> None:
    if bound is not None:
        monkeypatch.setenv("DLIGHTRAG_ANSWER__AGENT__BROWSER__ACCOUNT_REGISTRATION", bound)

    browser = DlightragConfig().answer.agent.browser  # pyright: ignore[reportCallIssue]

    assert browser.account_registration is registers


_MAILBOX = {
    "bucket": "agent-mail",
    "alias_domain": "orliantra.cc",
    "access_key_id": "fixture-key-id",
    "secret_access_key": "fixture-secret-key",
}


def test_without_a_bucket_there_is_no_agent_mailbox() -> None:
    mailbox = AgentMailboxConfig()

    assert (mailbox.enabled, mailbox.region, mailbox.prefix) == (False, None, "mail")
    assert AgentExecutionConfig().mailbox == mailbox


def test_a_bucket_comes_with_the_domain_and_the_keys_that_read_it() -> None:
    assert AgentMailboxConfig(**_MAILBOX).enabled is True
    for missing in ("alias_domain", "access_key_id", "secret_access_key"):
        with pytest.raises(
            ValidationError,
            match="answer.agent.mailbox.bucket requires alias_domain, access_key_id and "
            "secret_access_key",
        ):
            AgentMailboxConfig(**{k: v for k, v in _MAILBOX.items() if k != missing})


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("endpoint", "https://account.r2.cloudflarestorage.com"),
        ("region", "us-east-1"),
        ("alias_domain", "orliantra.cc"),
        ("access_key_id", "fixture-key-id"),
        ("secret_access_key", "fixture-secret-key"),
    ],
)
def test_mailbox_settings_without_a_bucket_are_refused(setting: str, value: str) -> None:
    with pytest.raises(ValidationError, match="answer.agent.mailbox settings require bucket"):
        AgentMailboxConfig(**{setting: value})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("bucket", "Agent-Mail"),
        ("bucket", "ab"),
        ("alias_domain", "localhost"),
        ("alias_domain", "Mail.Example.com"),
        ("prefix", "/mail"),
        ("prefix", "a//b"),
        ("region", "r" * 65),
        ("endpoint", "https://user:secret@account.r2.cloudflarestorage.com"),
    ],
)
def test_a_mailbox_value_outside_what_it_names_is_refused(setting: str, value: str) -> None:
    with pytest.raises(ValidationError):
        AgentMailboxConfig(**{**_MAILBOX, setting: value})  # pyright: ignore[reportArgumentType]


def test_a_blank_mailbox_variable_is_an_unset_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ENDPOINT",
        "REGION",
        "BUCKET",
        "ALIAS_DOMAIN",
        "ACCESS_KEY_ID",
        "SECRET_ACCESS_KEY",
    ):
        monkeypatch.setenv(f"DLIGHTRAG_ANSWER__AGENT__MAILBOX__{name}", "")

    mailbox = DlightragConfig().answer.agent.mailbox  # pyright: ignore[reportCallIssue]

    assert (mailbox.enabled, mailbox.endpoint, mailbox.region, mailbox.alias_domain) == (
        False,
        None,
        None,
        None,
    )


def test_the_agent_mailbox_is_bound_through_the_environment_and_its_keys_never_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = {
        **{name.upper(): value for name, value in _MAILBOX.items()},
        "ENDPOINT": "https://account.r2.cloudflarestorage.com",
        "REGION": "auto",
        "PREFIX": "inbound",
    }
    for name, value in bound.items():
        monkeypatch.setenv(f"DLIGHTRAG_ANSWER__AGENT__MAILBOX__{name}", value)

    config = DlightragConfig()  # pyright: ignore[reportCallIssue]
    mailbox = config.answer.agent.mailbox

    assert (
        mailbox.bucket,
        mailbox.alias_domain,
        mailbox.endpoint,
        mailbox.region,
        mailbox.prefix,
    ) == (
        "agent-mail",
        "orliantra.cc",
        "https://account.r2.cloudflarestorage.com",
        "auto",
        "inbound",
    )
    rendered = f"{config!r} {config} {config.model_dump()} {mailbox!r}"
    assert "fixture-key-id" not in rendered and "fixture-secret-key" not in rendered


@pytest.mark.parametrize(
    ("bound", "action_timeout", "depth"),
    [(None, 10.0, 12), ({"ACTION_TIMEOUT_SECONDS": "30", "SNAPSHOT_DEPTH": "20"}, 30.0, 20)],
    ids=["by-default", "bound"],
)
def test_the_browser_tools_bounds_reach_the_settings_with_downloads_under_the_attachment_limit(
    monkeypatch: pytest.MonkeyPatch,
    bound: dict[str, str] | None,
    action_timeout: float,
    depth: int,
) -> None:
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__ENDPOINTS", '["ws://agent-browser-1:3000/"]'
    )
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__EGRESS_PROXY", "http://agent-browser-egress:3128"
    )
    monkeypatch.setenv("DLIGHTRAG_ANSWER__GENERATION__MAX_ATTACHMENT_BYTES", "4096")
    for name, value in (bound or {}).items():
        monkeypatch.setenv(f"DLIGHTRAG_ANSWER__AGENT__BROWSER__{name}", value)

    settings = agent_browser_settings(DlightragConfig())  # pyright: ignore[reportCallIssue]

    assert settings is not None
    assert (settings.action_timeout_seconds, settings.snapshot_depth) == (action_timeout, depth)
    assert settings.max_download_bytes == 4096


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_timeout_seconds", 0),
        ("action_timeout_seconds", 121),
        ("snapshot_depth", 0),
        ("snapshot_depth", 65),
    ],
)
def test_a_browser_bound_outside_its_range_is_refused_naming_the_setting(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError, match=field):
        AgentBrowserConfig.model_validate({field: value})


def test_endpoints_without_an_egress_proxy_are_refused() -> None:
    with pytest.raises(
        ValidationError, match="answer.agent.browser.endpoints require egress_proxy"
    ):
        AgentBrowserConfig(endpoints=_POOL["endpoints"])


@pytest.mark.parametrize(
    ("field", "value", "problem"),
    [
        ("endpoints", ("http://secret-host:3000/",), "ws:// or wss:// URLs with a host"),
        ("endpoints", ("ws://",), "ws:// or wss:// URLs with a host"),
        ("endpoints", ("ws://secret-host:3000/?token=1",), "query or a fragment"),
        ("endpoints", ("ws://secret-host:3000/#x",), "query or a fragment"),
        ("endpoints", ("ws://secret-host:3000/", "WS://SECRET-HOST:3000"), "must be unique"),
        ("endpoints", ("ws://secret-host:99999999/",), "must be valid URLs"),
        ("egress_proxy", "https://secret-host:3128", "http:// URL with a host"),
        ("egress_proxy", "http://secret-host:3128/path", "a host and port only"),
        ("egress_proxy", "http://", "http:// URL with a host"),
    ],
)
def test_a_malformed_pool_address_is_refused_naming_the_setting_and_never_echoing_it(
    field: str, value: object, problem: str
) -> None:
    settings = {"egress_proxy": "http://agent-browser-egress:3128", field: value}
    if field == "egress_proxy":
        settings["endpoints"] = _POOL["endpoints"]

    with pytest.raises(ValidationError, match=problem) as caught:
        AgentBrowserConfig.model_validate(settings)

    assert field in str(caught.value)
    assert "secret-host" not in str(caught.value)


def test_naming_the_browser_without_a_configured_pool_is_refused() -> None:
    with pytest.raises(
        ValidationError,
        match="answer.web_sources.extract_providers names 'browser' but answer.agent.browser",
    ):
        AnswerSectionSettings(web_sources=WebSourcesConfig(extract_providers=("browser",)))


def test_the_browser_is_an_extract_provider_and_never_a_search_provider() -> None:
    with pytest.raises(ValidationError, match="search_providers"):
        WebSourcesConfig(search_providers=("browser",))  # pyright: ignore[reportArgumentType]
