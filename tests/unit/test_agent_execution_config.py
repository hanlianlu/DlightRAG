# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup validation for optional trusted local execution."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlightrag.application.config import (
    AgentBrowserConfig,
    AgentExecutionConfig,
    AnswerSectionSettings,
    DlightragConfig,
    WebSourceProviderConfig,
    WebSourcesConfig,
)
from dlightrag.engine.answer.execution_settings import (
    default_local_workspace_root,
    validate_agent_execution,
)


def test_execution_environment_defaults_to_trust() -> None:
    assert AgentExecutionConfig().execution_environment == "trust"


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


def _answer(
    web_sources: WebSourcesConfig | None = None, *, browser: bool = False
) -> AnswerSectionSettings:
    web_sources = web_sources or WebSourcesConfig()
    if not browser:
        return AnswerSectionSettings(web_sources=web_sources)
    agent = AgentExecutionConfig(browser=AgentBrowserConfig(**_POOL))
    return AnswerSectionSettings(web_sources=web_sources, agent=agent)


def _keyed() -> WebSourcesConfig:
    return WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-key"),
        tavily=WebSourceProviderConfig(api_key="tavily-key"),
    )


def test_the_agent_browser_is_off_until_endpoints_are_configured() -> None:
    assert AgentExecutionConfig().browser.enabled is False
    assert AgentBrowserConfig().endpoints == ()
    assert AgentBrowserConfig(**_POOL).enabled is True


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


def test_the_extract_chain_the_browser_joins_is_hosted_providers_first() -> None:
    assert _answer(_keyed(), browser=True).extract_chain() == ("exa", "tavily", "browser")


def test_without_a_browser_the_extract_chain_is_the_hosted_order() -> None:
    assert _answer(_keyed()).extract_chain() == ("exa", "tavily")
    assert _answer().extract_chain() == ()


def test_a_configured_browser_joins_an_explicit_list_at_its_end() -> None:
    chain = _answer(
        WebSourcesConfig(
            exa=WebSourceProviderConfig(api_key="exa-key"), extract_providers=("exa",)
        ),
        browser=True,
    ).extract_chain()

    assert chain == ("exa", "browser")


def test_naming_the_browser_in_an_explicit_list_only_positions_it() -> None:
    chain = _answer(
        WebSourcesConfig(
            exa=WebSourceProviderConfig(api_key="exa-key"),
            extract_providers=("browser", "exa"),
        ),
        browser=True,
    ).extract_chain()

    assert chain == ("browser", "exa")


def test_an_explicit_empty_extract_list_still_leaves_the_browser_its_chain() -> None:
    assert _answer(WebSourcesConfig(extract_providers=()), browser=True).extract_chain() == (
        "browser",
    )


def test_naming_the_browser_without_a_configured_pool_is_refused() -> None:
    with pytest.raises(
        ValidationError,
        match="answer.web_sources.extract_providers names 'browser' but answer.agent.browser",
    ):
        _answer(WebSourcesConfig(extract_providers=("browser",)))


def test_the_browser_is_an_extract_provider_and_never_a_search_provider() -> None:
    with pytest.raises(ValidationError, match="search_providers"):
        WebSourcesConfig(search_providers=("browser",))  # pyright: ignore[reportArgumentType]
