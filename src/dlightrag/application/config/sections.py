# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Self-contained configuration for dlightrag.

Configuration sources (highest → lowest precedence):
    1. Constructor arguments (when used as library)
    2. Environment variables (DLIGHTRAG_ prefix)
    3. .env file (secrets + deployment-only overrides)
    4. config.yaml (structured app settings)
    5. Default values

Configuration only holds validated settings. Turning them into connections, TLS
contexts, or LightRAG's environment variables belongs to the adapters that use
them, so constructing a configuration never mutates the process environment.
"""

import json
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, DotEnvSettingsSource, NoDecode, SettingsConfigDict

from dlightrag.application.config.yaml_source import Yaml12ConfigSettingsSource
from dlightrag.application.connections.policy import ConnectionPolicy
from dlightrag.engine.ai.settings import (
    FrozenSettings,
    ModelsSettings,
    ServiceUrl,
    freeze_settings_value,
    thaw_settings_value,
)
from dlightrag.engine.ai.telemetry import hides_secret_value, is_secret_key
from dlightrag.engine.rag.workspace.settings import CorpusSettings
from dlightrag.engine.rag.workspace.workspaces import (
    normalize_workspace,
    require_canonical_workspace_id,
)

type ServiceRole = Literal["writer", "reader"]

_YAML_FILE = "config.yaml"
_ENV_FILE = ".env"
_LOCAL_MCP_ALLOWED_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCAL_MCP_ALLOWED_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]
_LOCAL_API_HOSTS = {"127.0.0.1", "localhost", "::1"}
#: The HTTP client's names share the reserved namespace without being settings;
#: the process environment and a .env shared with clients may carry them. Test
#: suites and Compose-only inputs never join them: they use their own names.
_AUXILIARY_ENV_NAMES = frozenset(
    {
        "DLIGHTRAG_API_TOKEN",
        "DLIGHTRAG_API_URL",
        "DLIGHTRAG_CLIENT_TIMEOUT",
    }
)
#: Compose-only inputs that left the reserved namespace, named in the startup error.
_RENAMED_ENV_NAMES = MappingProxyType(
    {
        **{
            f"DLIGHTRAG_POSTGRES_{suffix}": f"COMPOSE_POSTGRES_{suffix}"
            for suffix in (
                "EFFECTIVE_CACHE_SIZE",
                "MAINTENANCE_WORK_MEM",
                "MAX_CONNECTIONS",
                "PG_TEXTSEARCH_FILTERED_SEED",
                "PG_TEXTSEARCH_FILTERED_SEED_MARGIN",
                "SHARED_BUFFERS",
                "SHM_SIZE",
                "WORK_MEM",
            )
        },
        **{
            f"DLIGHTRAG_MEMORY_POSTGRES_{suffix}": f"COMPOSE_MEMORY_POSTGRES_{suffix}"
            for suffix in ("DATABASE", "USER", "PASSWORD")
        },
        "DLIGHTRAG_SKILLS_DIR": "COMPOSE_GLOBAL_SKILLS_DIR",
    }
)


def _is_auxiliary_env_name(name: str) -> bool:
    return name.upper() in _AUXILIARY_ENV_NAMES


def _unknown_environment(names: Sequence[str]) -> ValueError:
    renamed = [
        f"{name} -> {_RENAMED_ENV_NAMES[name.upper()]}"
        for name in names
        if name.upper() in _RENAMED_ENV_NAMES
    ]
    hint = f"; renamed: {', '.join(renamed)}" if renamed else ""
    return ValueError(f"Unknown DlightRAG environment variables: {list(names)}{hint}")


def _drop_auxiliary_dotenv_names(source: DotEnvSettingsSource) -> None:
    """Let a .env shared with clients carry their names; they are not settings.

    Only this source is filtered: constructor values and config.yaml keep rejecting
    unknown keys. A renamed Compose input fails here with its replacement named.
    """
    renamed = sorted(name for name in source.env_vars if name.upper() in _RENAMED_ENV_NAMES)
    if renamed:
        raise _unknown_environment(renamed)
    source.env_vars = {
        name: value for name, value in source.env_vars.items() if not _is_auxiliary_env_name(name)
    }


PostgresSSLMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]


def _validate_oauth_endpoint_url(value: str, field_name: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not include query or fragment components")
    if parsed.scheme != "https" and parsed.hostname not in _LOCAL_API_HOSTS:
        raise ValueError(f"{field_name} must use HTTPS except on loopback")


def _find_env_file() -> Path | None:
    """Locate .env in the current working directory only."""
    candidate = Path(_ENV_FILE)
    return candidate if candidate.is_file() else None


def _find_yaml_config() -> Path | None:
    """Locate config.yaml in the current working directory."""
    cwd = Path(_YAML_FILE)
    if cwd.is_file():
        return cwd
    return None


class CitationHighlightConfig(BaseModel):
    """Optional semantic highlighting for cited source snippets."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: bool = True
    timeout: float = Field(default=10.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)
    max_input_chars: int = Field(default=4096, ge=1)
    cache_size: int = Field(default=500, ge=1)


class CitationsConfig(BaseModel):
    """Citation validation and UI enrichment configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    highlights: CitationHighlightConfig = Field(default_factory=CitationHighlightConfig)


class AnswerConfig(BaseModel):
    """Final answer generation controls."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_attachments: int = Field(
        default=6,
        ge=0,
        description="Maximum answer attachments admitted per request.",
    )
    max_attachment_bytes: int = Field(
        default=100 * 1024 * 1024,
        ge=1,
        description="Maximum bytes accepted for a single answer attachment (100 MiB).",
    )
    max_total_attachment_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=1,
        description="Maximum total bytes accepted across all answer attachments (128 MiB).",
    )
    max_images: int = Field(
        default=12,
        ge=0,
        description="Maximum current and retrieved image blocks sent to the answer LLM.",
    )
    lineage_adoption: bool = Field(
        default=True,
        description=(
            "Let a Run adopt a Resource an earlier Run on the same Agent Session "
            "registered, on first use, by reusing that Run's stored conversion view."
        ),
    )

    # Vision support is runtime Answer state, not config. Users do not set it
    # in config.yaml; the startup probe records it on Application health.
    image_max_bytes: int = Field(
        default=3_000_000,
        ge=1,
        description="Maximum compressed binary bytes per answer image.",
    )
    image_max_total_bytes: int = Field(
        default=24_000_000,
        ge=1,
        description="Maximum total compressed binary image bytes per answer request.",
    )
    image_max_px: int = Field(
        default=1536,
        ge=1,
        description="Maximum image long edge sent to the answer LLM.",
    )
    image_max_pixels: int = Field(
        default=40_000_000,
        ge=1,
        description="Maximum decoded source pixels accepted for answer and Web images.",
    )
    image_min_px: int = Field(
        default=1024,
        ge=1,
        description="Minimum long edge preserved before skipping oversized answer images.",
    )
    image_quality: int = Field(
        default=89,
        ge=1,
        le=95,
        description="Initial JPEG quality for answer LLM image previews.",
    )
    image_min_quality: int = Field(
        default=79,
        ge=1,
        le=95,
        description="Minimum JPEG quality before skipping oversized answer images.",
    )


class LaneRuntimeConfig(BaseModel):
    """Per-lane local workers and deployment-wide nonterminal admission limit."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    worker_concurrency: int = Field(ge=1)
    max_nonterminal_runs: int = Field(ge=1)


class RuntimeConfig(BaseModel):
    """Operation-neutral durable RunRuntime admission and retention."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    query: LaneRuntimeConfig = LaneRuntimeConfig(
        worker_concurrency=16,
        max_nonterminal_runs=30_000,
    )
    corpus_mutation: LaneRuntimeConfig = LaneRuntimeConfig(
        worker_concurrency=2,
        max_nonterminal_runs=1_000,
    )
    run_retention_days: int = Field(
        default=365,
        ge=1,
        description=(
            "Retention floor for terminal Answer Runs and their event logs, "
            "counted from finished_at; top-level Retrieval uses seven days. "
            "Also the retention clock for memory: superseded profile history "
            "is purged after the same span. A conversation whose last turn's "
            "Run ages out becomes empty and is then reclaimed."
        ),
    )


class ArtifactPublicationConfig(BaseModel):
    """Independent Agent workspace, publication, and browser-preview budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_artifacts: int = Field(default=20, ge=1)
    max_file_bytes: int = Field(default=30 * 1024 * 1024, ge=1)
    max_total_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    workspace_max_bytes: int = Field(
        default=1024 * 1024 * 1024,
        ge=1,
        le=5 * 1024 * 1024 * 1024,
    )
    preview_image_max_pixels: int = Field(default=16_000_000, ge=1)
    preview_image_max_edge: int = Field(default=4096, ge=1)
    original_image_max_pixels: int = Field(default=64_000_000, ge=1)
    original_image_max_edge: int = Field(default=8000, ge=1)
    active_html_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1)


class SessionNotesConfig(BaseModel):
    """Bounds on the durable memory one Agent Session keeps.

    Memory belongs to the Session (ADR 0022), so these bound the Session's notes
    plane rather than any Run's workspace: a note outlives the Run that wrote it, and
    the plane stays small enough to be memory rather than a second transcript. A note
    that does not fit is refused for that note alone — never truncated, never evicted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_count: int = Field(
        default=64,
        ge=1,
        le=1_024,
        description=(
            "How many notes one Agent Session's plane holds. The compaction summary "
            "names at most eight of them, so a larger plane keeps memory that later "
            "turns can still read with read(path=...) without spending prompt budget."
        ),
    )
    max_bytes: int = Field(
        default=256 * 1024,
        ge=1_024,
        le=16 * 1024 * 1024,
        description=(
            "Total bytes one Agent Session's notes may hold. A note larger than this "
            "whole budget can never be remembered: it is refused by name on the Run's "
            "trace and stays in the Run's own working copy."
        ),
    )


class AgentBrowserConfig(BaseModel):
    """The Agent Browser pool (ADR 0032): Research renders and drives pages in a browser it leases.

    No endpoint means no Agent Browser. The endpoints and the proxy are Compose Service
    names, so ``docker-compose.yml`` binds them (ADR 0006) and ``config.yaml`` leaves
    them unset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    endpoints: tuple[ServiceUrl, ...] = Field(
        default=(),
        description=(
            "One Playwright run-server WebSocket URL per pool container; each serves one "
            "Run at a time. Empty disables the Agent Browser."
        ),
    )
    egress_proxy: ServiceUrl | None = Field(
        default=None,
        description=(
            "The HTTP proxy every browser launch uses. It is the pool's only way out, so "
            "endpoints require it."
        ),
    )
    chromium_sandbox: bool = Field(
        default=True,
        description=(
            "Whether every browser launches inside Chromium's own process sandbox. Whether "
            "a pool host can start it is the operator's to state: a host whose user "
            "namespaces or seccomp profile forbid the sandbox cannot launch any browser "
            "while this is true, and renders fail as unreachable until the host is relaxed "
            "or this is set false. False runs Chromium with --no-sandbox, so the container "
            "and its network are the only isolation."
        ),
    )
    lease_wait_seconds: float = Field(
        default=10.0,
        ge=0,
        le=120,
        description="How long a render waits for a free browser before it reports the pool busy.",
    )
    connect_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        le=120,
        description="How long connecting to one browser may take.",
    )
    navigation_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        description=(
            "How long a page may take to load, and how long a wait for text, a screenshot, "
            "a capture, or the save of one downloaded file may take."
        ),
    )
    settle_timeout_seconds: float = Field(
        default=5.0,
        ge=0,
        le=60,
        description="How long a loaded page may take to go quiet; a page that never does is read.",
    )
    action_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=120,
        description=(
            "How long one element action, one snapshot, or one find may take on a page the "
            "browser tool drives."
        ),
    )
    snapshot_depth: int = Field(
        default=12,
        ge=1,
        le=64,
        description=(
            "How many levels of the page the browser tool's accessibility snapshot shows; "
            "deeper elements keep their refs, and find locates them."
        ),
    )
    idle_release_seconds: float = Field(
        default=30.0,
        ge=0,
        le=600,
        description=(
            "A Run gives its browser back once it has gone this long with no page open and no "
            "render in flight; its next page or render leases one again. 0 releases at once."
        ),
    )
    account_registration: bool = Field(
        default=True,
        description=(
            "Whether the Agent may register on third-party sites and sign in with Agent "
            "Accounts of its own (ADR 0034). False offers neither register nor login."
        ),
    )

    @field_validator("endpoints")
    @classmethod
    def _websocket_endpoints(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        identities = []
        for endpoint in value:
            try:
                parts = urlsplit(endpoint)
                port = parts.port
            except ValueError:
                raise ValueError("endpoints must be valid URLs") from None
            if parts.scheme not in {"ws", "wss"} or not parts.hostname:
                raise ValueError("endpoints must be ws:// or wss:// URLs with a host")
            if parts.query or parts.fragment:
                raise ValueError("endpoints must not carry a query or a fragment")
            identities.append((parts.scheme, parts.hostname.lower(), port, parts.path or "/"))
        if len(set(identities)) != len(identities):
            raise ValueError("endpoints must be unique")
        return value

    @field_validator("egress_proxy")
    @classmethod
    def _http_proxy(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parts = urlsplit(value)
            _ = parts.port
        except ValueError:
            raise ValueError("egress_proxy must be a valid URL") from None
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError("egress_proxy must be an http:// URL with a host")
        if parts.path not in {"", "/"} or parts.query or parts.fragment:
            raise ValueError("egress_proxy must name a host and port only")
        return value

    @model_validator(mode="after")
    def _endpoints_have_a_proxy(self) -> Self:
        if self.endpoints and self.egress_proxy is None:
            raise ValueError("answer.agent.browser.endpoints require egress_proxy")
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.endpoints)


class AgentMailboxConfig(BaseModel):
    """The Agent Mailbox (ADR 0034): mail to the Agent's mailbox aliases, read from an S3-compatible
    bucket the deployment fills. No bucket means no Agent Mailbox.

    The bucket's layout is the contract: each message whole, as one object, under
    ``<prefix>/<envelope recipient, lower case>/``. How mail gets there is the deployment's.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    endpoint: ServiceUrl | None = Field(
        default=None,
        description="The bucket's S3 endpoint. Unset, AWS S3's own endpoint for the region.",
    )
    region: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description=(
            "The bucket's region as its endpoint names it, such as us-east-1 on AWS S3 or auto "
            "on Cloudflare R2. Unset, the AWS SDK resolves it as it does for any S3 client."
        ),
    )
    bucket: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$",
        description="The bucket the deployment's mail routing writes to. Unset disables the mailbox.",
    )
    prefix: str = Field(
        default="mail",
        pattern=r"^([A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*)?$",
        description="The key prefix the routing writes under; empty puts alias folders at the root.",
    )
    alias_domain: str | None = Field(
        default=None,
        max_length=253,
        pattern=(r"^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$"),
        description="The domain the Agent's addresses are minted on, which the routing delivers.",
    )
    access_key_id: str | None = Field(default=None, repr=False)
    secret_access_key: str | None = Field(default=None, repr=False)

    @field_validator(
        "endpoint",
        "region",
        "bucket",
        "alias_domain",
        "access_key_id",
        "secret_access_key",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        """A blank variable in ``.env`` is an unset setting."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def _a_bucket_comes_with_what_reads_it(self) -> Self:
        if self.bucket is None:
            if any(
                (
                    self.endpoint,
                    self.region,
                    self.alias_domain,
                    self.access_key_id,
                    self.secret_access_key,
                )
            ):
                raise ValueError("answer.agent.mailbox settings require bucket")
        elif not (self.alias_domain and self.access_key_id and self.secret_access_key):
            raise ValueError(
                "answer.agent.mailbox.bucket requires alias_domain, access_key_id and "
                "secret_access_key"
            )
        return self

    @property
    def enabled(self) -> bool:
        return self.bucket is not None


class AgentExecutionConfig(BaseModel):
    """Optional Agent execution: no environment, or one confined to its workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    @field_validator("skills_root", "owner_skills_root")
    @classmethod
    def _validate_skills_roots(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).expanduser().is_absolute():
            raise ValueError("skill roots must be absolute paths when set")
        return value

    @field_validator("search_tool_cache_root")
    @classmethod
    def _validate_search_tool_cache_root(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).expanduser().is_absolute():
            raise ValueError("search tool cache root must be absolute when set")
        return value

    @field_validator("fd_path", "ripgrep_path")
    @classmethod
    def _validate_search_tool_path(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("search tool paths must be non-empty")
        candidate = Path(value).expanduser()
        if "/" in value and not candidate.is_absolute():
            raise ValueError("search tool paths containing directories must be absolute")
        return value

    execution_environment: Literal["disabled", "trust"] = Field(
        default="trust",
        description=(
            "disabled | trust. Default trust runs the rooted local adapter, which "
            "confines every Agent process to its Agent Workspace; set disabled to "
            "expose no path or Bash tools. Isolation stronger than the host kernel "
            "belongs to the environment the application is deployed in."
        ),
    )
    child_guidance_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=86_400,
        description="Default expiry for a durable ask_parent request.",
    )
    session_notes: SessionNotesConfig = Field(default_factory=SessionNotesConfig)
    workspace_root: str | None = Field(
        default=None,
        description=(
            "Absolute Agent Workspace root. When trust and unset, "
            "defaults to ~/.dlightrag/agent_workspaces. Multi-host deployments "
            "must set the same absolute path on every worker."
        ),
    )
    skills_root: str | None = Field(
        default=None,
        description=(
            "Absolute global Agent Skills root. When unset, defaults to "
            "~/.dlightrag/skills. Multi-host deployments must set the same "
            "shared absolute path on every worker."
        ),
    )
    owner_skills_root: str | None = Field(
        default=None,
        description=(
            "Absolute per-owner Agent Skills root. When unset, defaults to "
            "~/.dlightrag/owner_skills. Users publish their own skills here "
            "through the validated publish_skill tool; the global root stays "
            "operator-provisioned and read-only."
        ),
    )
    disabled_builtin_skills: tuple[str, ...] = Field(
        default=(),
        description=(
            "Names of packaged built-in Agent Skills to hide. Global and owner skills "
            "with the same names remain discoverable."
        ),
    )
    fd_path: str = Field(
        default="fd",
        description="Absolute fd executable path or PATH command name (minimum 10.5.0).",
    )
    ripgrep_path: str = Field(
        default="rg",
        description="Absolute ripgrep executable path or PATH command name (minimum 15.2.0).",
    )
    search_tool_cache_root: str | None = Field(
        default=None,
        description="Absolute managed fd/ripgrep cache; defaults to ~/.dlightrag/tools.",
    )
    search_tool_auto_install: bool = Field(
        default=False,
        description=(
            "Allow runtime download of the latest compatible fd/ripgrep GitHub releases "
            "with published SHA-256 verification. Disabled by default for services."
        ),
    )
    publication: ArtifactPublicationConfig = Field(default_factory=ArtifactPublicationConfig)
    connections: ConnectionPolicy = Field(default_factory=ConnectionPolicy)
    browser: AgentBrowserConfig = Field(default_factory=AgentBrowserConfig)
    mailbox: AgentMailboxConfig = Field(default_factory=AgentMailboxConfig)


class WebConversationsConfig(BaseModel):
    """Browser conversation surface; retention follows RuntimeConfig."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    active_html_preview_enabled: bool = Field(
        default=True,
        description="Allow explicit opaque-origin execution of self-contained HTML Artifacts.",
    )


class WebSourceProviderConfig(BaseModel):
    """Credential for one first-class Web source provider."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    api_key: str | None = Field(default=None, repr=False)

    @field_validator("api_key", mode="before")
    @classmethod
    def _normalize_api_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class WebSourcesConfig(BaseModel):
    """Independent ordered Search and Extract provider chains.

    ``None`` derives an order from configured keys (Exa, then Tavily). An
    explicit empty tuple disables that operation while retaining credentials.
    The Extract chain may also name ``browser``, the Agent Browser (ADR 0032), which
    is a configured deployment capability rather than a keyed provider.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    search_providers: tuple[Literal["exa", "tavily"], ...] | None = None
    extract_providers: tuple[Literal["exa", "tavily", "browser"], ...] | None = None
    exa: WebSourceProviderConfig = Field(default_factory=WebSourceProviderConfig)
    tavily: WebSourceProviderConfig = Field(default_factory=WebSourceProviderConfig)

    @field_validator("search_providers", "extract_providers")
    @classmethod
    def _unique_provider_order(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if len(set(value)) != len(value):
            raise ValueError("Web provider order cannot contain duplicates")
        return value

    def configured_providers(self) -> tuple[Literal["exa", "tavily"], ...]:
        return tuple(
            name
            for name, configured in (("exa", self.exa.api_key), ("tavily", self.tavily.api_key))
            if configured
        )  # type: ignore[return-value]

    def search_order(self) -> tuple[Literal["exa", "tavily"], ...]:
        return (
            self.search_providers
            if self.search_providers is not None
            else self.configured_providers()
        )

    def extract_order(self) -> tuple[Literal["exa", "tavily", "browser"], ...]:
        return (
            self.extract_providers
            if self.extract_providers is not None
            else self.configured_providers()
        )

    @model_validator(mode="after")
    def _orders_have_credentials(self) -> WebSourcesConfig:
        keys = {"exa": self.exa.api_key, "tavily": self.tavily.api_key}
        for operation, order in (
            ("search", self.search_order()),
            ("extract", self.extract_order()),
        ):
            # The Agent Browser holds no key; whether it is configured is the Answer
            # section's to check, since the pool is configured beside it.
            missing = [name for name in order if name != "browser" and not keys[name]]
            if missing:
                raise ValueError(f"Web {operation} provider(s) lack api_key: {', '.join(missing)}")
        return self


class AccessControlRuleConfig(BaseModel):
    """Map one verified JWT claim value to DlightRAG actions."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    claim: str
    value: str
    workspaces: list[str] = Field(default_factory=lambda: ["*"])
    actions: list[str]


class AccessControlConfig(BaseModel):
    """DlightRAG resource authorization settings.

    Without rules every authenticated caller holds everything; with rules, the
    rules (and each workspace's creator) decide.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    rules: list[AccessControlRuleConfig] = Field(default_factory=list)


def _redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact values whose keys name secrets (see ``is_secret_key``)."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if is_secret_key(key):
            result[key] = "***" if hides_secret_value(key, value) else value
        elif isinstance(value, dict):
            result[key] = _redact_dict(value)
        elif isinstance(value, list | tuple):
            result[key] = type(value)(
                _redact_dict(item) if isinstance(item, dict) else item for item in value
            )
        else:
            result[key] = value
    return result


def _mask_hidden_fields(model: BaseModel, data: dict[str, Any]) -> dict[str, Any]:
    """Mask every field a settings model declares ``repr=False`` in its dump."""
    for name, field in type(model).model_fields.items():
        if name not in data:
            continue
        value = getattr(model, name)
        if not field.repr:
            data[name] = "***" if data[name] else data[name]
        elif isinstance(value, BaseModel) and isinstance(data[name], dict):
            data[name] = _mask_hidden_fields(value, data[name])
        elif isinstance(value, tuple | list) and isinstance(data[name], list | tuple):
            data[name] = type(data[name])(
                _mask_hidden_fields(item, dumped)
                if isinstance(item, BaseModel) and isinstance(dumped, dict)
                else dumped
                for item, dumped in zip(value, data[name], strict=False)
            )
    return data


class WebIdentitySettings(BaseModel):
    """Edge-asserted identity source for the Web surface.

    The browser front door already authenticated the human (Cloudflare Access,
    Azure Easy Auth, or AWS Amplify/CloudFront); the Web surface verifies the
    edge credential per request and never renders a login page or issues a
    token of its own. The edge token is verified like an API bearer: each unset
    field below is the API verifier's own (``jwt_issuer``, ``jwt_audience``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    edge: Literal["cloudflare", "azure", "aws"] | None = Field(
        default=None,
        description=(
            "Edge identity provider. None keeps the existing pasted-token Web "
            "login (development/operator hatch)."
        ),
    )
    issuer: ServiceUrl | None = Field(
        default=None,
        description="Edge-token issuer when it differs from jwt_issuer.",
    )
    audience: Annotated[str | list[str] | None, NoDecode] = Field(
        default=None,
        description="Edge-token audience when it differs from jwt_audience (e.g. an AAD client id).",
    )
    jwks_url: ServiceUrl | None = Field(
        default=None,
        description=(
            "Edge-token keys when the issuer's OpenID discovery cannot name them. "
            "Unset, the edge issuer's discovery document does."
        ),
    )

    @field_validator("audience", mode="before")
    @classmethod
    def _normalize_audience(cls, value: Any) -> str | list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if not text.startswith("["):
                return text
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "web_identity.audience string must be a plain audience or a JSON array"
                ) from exc
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        raise ValueError("web_identity.audience must be a string or a list of strings")


class DeploymentSettings(FrozenSettings):
    service_role: ServiceRole = "writer"
    # The default workspace's display name; `workspace_id` is its canonical id.
    workspace: str = "default"
    working_dir: str = "./dlightrag_storage"

    @field_validator("workspace")
    @classmethod
    def _names_a_canonical_workspace(cls, value: str) -> str:
        require_canonical_workspace_id(normalize_workspace(value))
        return value

    @property
    def workspace_id(self) -> str:
        """The default workspace's canonical id, derived here once for every caller."""
        return normalize_workspace(self.workspace)

    @field_validator("working_dir")
    @classmethod
    def _absolute_working_dir(cls, value: str) -> str:
        return str(Path(value).resolve())

    @property
    def working_dir_path(self) -> Path:
        return Path(self.working_dir)


class PostgresSettings(FrozenSettings):
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "dlightrag"
    password: str = Field(default="dlightrag", repr=False)  # local development default
    database: str = "dlightrag"
    ssl_mode: PostgresSSLMode | None = None
    ssl_cert: str | None = None
    ssl_key: str | None = None
    ssl_root_cert: str | None = None
    ssl_crl: str | None = None
    pool_min_size: int = 2
    pool_max_size: int = 16
    command_timeout: float | None = Field(default=60.0, gt=0)
    acquire_timeout: float = Field(default=30.0, gt=0)
    lightrag_pool_max_size: int = Field(default=16, ge=1)
    session_settings: Mapping[str, str | int | float | bool] = Field(default_factory=dict)
    statement_cache_size: int | None = None
    connection_retries: int = Field(default=10, ge=1, le=100)
    connection_retry_backoff: float = Field(default=3.0, ge=0, le=300)
    connection_retry_backoff_max: float = Field(default=30.0, ge=0, le=600)
    pool_close_timeout: float = Field(default=5.0, ge=0, le=30)

    @field_validator("session_settings", mode="after")
    @classmethod
    def _freeze_session_settings(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_settings_value(value)

    @field_serializer("session_settings")
    def _serialize_session_settings(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_settings_value(value)


class LightRAGStorageSettings(FrozenSettings):
    """Deployment-static LightRAG storage names and narrow vector bindings."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    vector_index_type: Literal["HNSW", "HNSW_HALFVEC", "IVFFLAT", "VCHORDRQ"] = "HNSW_HALFVEC"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 256
    hnsw_ef_search: int = 256
    vector_storage: Literal["PGVectorStorage", "MilvusVectorDBStorage"] = "PGVectorStorage"
    graph_storage: Literal["PGTableGraphStorage"] = "PGTableGraphStorage"
    kv_storage: Literal["PGKVStorage"] = "PGKVStorage"
    doc_status_storage: Literal["PGDocStatusStorage"] = "PGDocStatusStorage"
    milvus_uri: ServiceUrl | None = Field(default=None, repr=False)
    milvus_token: str | None = Field(default=None, repr=False)
    milvus_db_name: str | None = None
    vector_db_kwargs: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("milvus_uri", "milvus_token", "milvus_db_name", mode="before")
    @classmethod
    def _normalize_optional_milvus_binding(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip() or None

    @field_validator("vector_db_kwargs", mode="after")
    @classmethod
    def _freeze_vector_kwargs(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if len(value) > 16:
            raise ValueError("vector_db_kwargs accepts at most 16 entries")
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key or len(key) > 64:
                raise ValueError("vector_db_kwargs keys must contain 1 to 64 characters")
            if any(marker in key.lower() for marker in ("password", "secret", "token", "uri")):
                raise ValueError("vector_db_kwargs must not contain credentials or endpoint URIs")
            if item is not None and not isinstance(item, str | int | float | bool):
                raise ValueError("vector_db_kwargs values must be scalar")
            if isinstance(item, str) and len(item) > 128:
                raise ValueError("vector_db_kwargs string values must not exceed 128 characters")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("vector_db_kwargs numeric values must be finite")
            normalized[key] = item
        if len(json.dumps(normalized, separators=(",", ":"), default=str).encode()) > 4096:
            raise ValueError("vector_db_kwargs exceeds the 4096-byte configuration bound")
        return freeze_settings_value(normalized)

    @model_validator(mode="after")
    def _validate_vector_kwargs(self) -> Self:
        if self.vector_storage != "MilvusVectorDBStorage":
            if any((self.milvus_uri, self.milvus_token, self.milvus_db_name)):
                raise ValueError(
                    "milvus_uri, milvus_token, and milvus_db_name require "
                    "vector_storage='MilvusVectorDBStorage'"
                )
            return self
        supported = {
            "cosine_better_than_threshold",
            "index_type",
            "metric_type",
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef",
            "sq_type",
            "sq_refine",
            "sq_refine_type",
            "sq_refine_k",
            "ivf_nlist",
            "ivf_nprobe",
        }
        unknown = sorted(set(self.vector_db_kwargs).difference(supported))
        if unknown:
            raise ValueError(
                "Milvus vector_db_kwargs contains unsupported keys: " + ", ".join(unknown)
            )
        return self

    @field_serializer("vector_db_kwargs")
    def _serialize_vector_kwargs(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_settings_value(value)


class StorageSettings(FrozenSettings):
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    lightrag: LightRAGStorageSettings = Field(default_factory=LightRAGStorageSettings)


class AnswerSectionSettings(FrozenSettings):
    generation: AnswerConfig = Field(default_factory=AnswerConfig)
    agent: AgentExecutionConfig = Field(default_factory=AgentExecutionConfig)
    citations: CitationsConfig = Field(default_factory=CitationsConfig)
    conversations: WebConversationsConfig = Field(default_factory=WebConversationsConfig)
    web_sources: WebSourcesConfig = Field(default_factory=WebSourcesConfig)

    @model_validator(mode="after")
    def _a_named_browser_is_configured(self) -> Self:
        if "browser" in self.web_sources.extract_order() and not self.agent.browser.enabled:
            raise ValueError(
                "answer.web_sources.extract_providers names 'browser' but "
                "answer.agent.browser has no endpoints"
            )
        return self

    def extract_chain(self) -> tuple[str, ...]:
        """The Extract chain a Run walks, in order: hosted providers, then the browser.

        A configured Agent Browser always joins the automatic chain, at its end unless the
        list names it elsewhere, so a deployment that adds the browser changes no list.
        """
        order: tuple[str, ...] = self.web_sources.extract_order()
        if self.agent.browser.enabled and "browser" not in order:
            return (*order, "browser")
        return order


type JwtAlgorithm = Literal["HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256"]


class AccessSectionSettings(FrozenSettings):
    auth_mode: Literal["none", "simple", "jwt"] = "none"
    api_token: str | None = Field(default=None, repr=False)
    allow_insecure_no_auth: bool = False
    jwt_verification_key: str | None = Field(default=None, repr=False)
    jwt_jwks_url: ServiceUrl | None = None
    jwt_issuer: ServiceUrl | None = None
    jwt_audience: Annotated[str | tuple[str, ...] | None, NoDecode] = None
    # Pins the signing algorithm. Unset, a published key names its own and
    # jwt_verification_key is HS256.
    jwt_algorithm: JwtAlgorithm | None = None
    # The Web is same-origin and REST/MCP clients are not browsers, so no origin
    # is allowed unless one is named.
    cors_allow_origins: tuple[str, ...] = ()
    web_identity: WebIdentitySettings = Field(default_factory=WebIdentitySettings)
    control: AccessControlConfig = Field(default_factory=AccessControlConfig)

    @field_validator("jwt_audience", mode="before")
    @classmethod
    def _normalize_audience(cls, value: Any) -> str | tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if not text.startswith("["):
                return text
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("jwt_audience must be a plain audience or JSON array") from exc
        if isinstance(value, (list, tuple)):
            items = tuple(str(item).strip() for item in value if str(item).strip())
            return items or None
        raise ValueError("jwt_audience must be a string or sequence of strings")


class ApiInterfaceSettings(FrozenSettings):
    host: str = "127.0.0.1"
    port: int = Field(default=8100, ge=1, le=65535)


class McpInterfaceSettings(FrozenSettings):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8101, ge=1, le=65535)
    allowed_hosts: tuple[str, ...] = tuple(_LOCAL_MCP_ALLOWED_HOSTS)
    allowed_origins: tuple[str, ...] = tuple(_LOCAL_MCP_ALLOWED_ORIGINS)
    resource_server_url: ServiceUrl | None = None


class InterfacesSettings(FrozenSettings):
    api: ApiInterfaceSettings = Field(default_factory=ApiInterfaceSettings)
    mcp: McpInterfaceSettings = Field(default_factory=McpInterfaceSettings)
    max_upload_size_mb: int = Field(default=512, ge=1)


class ObservabilitySettings(FrozenSettings):
    log_level: str = "info"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = Field(default=None, repr=False)
    langfuse_host: ServiceUrl = "https://cloud.langfuse.com"
    langfuse_export_external_spans: bool = False
    langfuse_trace_sensitive_data: bool = True
    langfuse_environment: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,39}$",
        description="Deployment label; Langfuse drops any other alphabet or length.",
    )
    langfuse_release: str | None = None
    langfuse_sample_rate: float = Field(default=1.0, ge=0, le=1)
    langfuse_timeout: int | None = Field(default=None, ge=1, le=300)
    langfuse_flush_at: int | None = Field(default=None, ge=1)
    langfuse_flush_interval: float | None = Field(default=None, ge=0.1, le=300)

    @field_validator("langfuse_environment")
    @classmethod
    def _reject_reserved_environment_prefix(cls, value: str | None) -> str | None:
        if value is not None and value.lower().startswith("langfuse"):
            raise ValueError("langfuse_environment must not start with 'langfuse'")
        return value


class DlightragConfig(BaseSettings):
    """The nine-section immutable DlightRAG configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DLIGHTRAG_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        dotenv_filtering="match_prefix",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    def __init__(self, **values: Any) -> None:
        allowed = {name.upper() for name in self.__class__.model_fields}
        # Settings read the environment case-insensitively, so the gate does too.
        unknown = sorted(
            key
            for key in os.environ
            if key.upper().startswith("DLIGHTRAG_")
            and not _is_auxiliary_env_name(key)
            and key.upper().removeprefix("DLIGHTRAG_").split("__", 1)[0] not in allowed
        )
        if unknown:
            raise _unknown_environment(unknown)
        super().__init__(**values)
        # BaseSettings serializes constructor-supplied nested models through its
        # source pipeline. Restore the already-validated canonical instances so
        # explicit-field provenance (notably keyless vs incomplete model roles)
        # and object identity survive composition.
        for field_name in self.__class__.model_fields:
            supplied = values.get(field_name)
            if isinstance(supplied, BaseModel):
                object.__setattr__(self, field_name, supplied)

    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    answer: AnswerSectionSettings = Field(default_factory=AnswerSectionSettings)
    runtime: RuntimeConfig = RuntimeConfig()
    access: AccessSectionSettings = Field(default_factory=AccessSectionSettings)
    interfaces: InterfacesSettings = Field(default_factory=InterfacesSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        if isinstance(dotenv_settings, DotEnvSettingsSource):
            _drop_auxiliary_dotenv_names(dotenv_settings)
        sources = [init_settings, env_settings, dotenv_settings]
        if (yaml_path := _find_yaml_config()) is not None:
            sources.append(Yaml12ConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        dumped = _mask_hidden_fields(self, super().model_dump(**kwargs))
        return _redact_dict(dumped)

    def model_dump_json(self, **kwargs: Any) -> str:
        indent = kwargs.pop("indent", None)
        return json.dumps(self.model_dump(**kwargs), default=str, indent=indent)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in self.model_dump().items())})"

    def __str__(self) -> str:
        return repr(self)

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        profiles = self.corpus.retrieval.bm25_profiles
        if len({profile.name for profile in profiles}) != len(profiles):
            raise ValueError("bm25_profiles names must be unique")
        if self.corpus.retrieval.bm25_enabled and not any(p.fallback for p in profiles):
            raise ValueError("bm25_profiles must include at least one fallback profile")
        self._validate_auth()
        self._validate_storage_composition()
        return self

    def _validate_storage_composition(self) -> None:
        lightrag = self.storage.lightrag
        if lightrag.vector_storage != "MilvusVectorDBStorage":
            return
        if self.is_reader:
            raise ValueError(
                "service_role='reader' does not support MilvusVectorDBStorage; "
                "use the writer role or PGVectorStorage"
            )
        promotion = self.corpus.promotion
        if promotion.doc_threshold is not None or promotion.chunk_threshold is not None:
            raise ValueError(
                "workspace promotion thresholds require PGVectorStorage and cannot be used "
                "with MilvusVectorDBStorage"
            )

    def _validate_auth(self) -> None:
        access, api, mcp = self.access, self.interfaces.api, self.interfaces.mcp
        if access.auth_mode == "none" and access.api_token:
            raise ValueError("api_token is set; configure auth_mode='simple' explicitly")
        if access.auth_mode == "simple" and not access.api_token:
            raise ValueError("auth_mode='simple' requires api_token")
        if access.auth_mode == "jwt":
            if not (access.jwt_verification_key or access.jwt_jwks_url or access.jwt_issuer):
                raise ValueError("auth_mode='jwt' requires jwt_issuer or jwt_verification_key")
            # Published keys verify any token the issuer signs, so the audience pins ours.
            published = access.jwt_jwks_url or not access.jwt_verification_key
            if published and not (access.jwt_issuer and access.jwt_audience):
                raise ValueError("verifying published keys requires jwt_issuer and jwt_audience")
        web = access.web_identity
        if web.edge:
            if access.auth_mode != "jwt":
                raise ValueError("web_identity.edge requires auth_mode='jwt'")
            if not ((web.issuer or access.jwt_issuer) and (web.audience or access.jwt_audience)):
                raise ValueError("web_identity.edge requires an issuer and an audience")
        if mcp.resource_server_url:
            if access.auth_mode != "jwt" or mcp.transport != "streamable-http":
                raise ValueError("mcp.resource_server_url requires JWT and streamable-http")
            if not access.jwt_issuer:
                raise ValueError("mcp.resource_server_url requires jwt_issuer")
            _validate_oauth_endpoint_url(mcp.resource_server_url, "mcp.resource_server_url")
            _validate_oauth_endpoint_url(access.jwt_issuer, "jwt_issuer")
        insecure = []
        if api.host not in _LOCAL_API_HOSTS:
            insecure.append(f"REST host={api.host}")
        if mcp.transport == "streamable-http" and mcp.host not in _LOCAL_API_HOSTS:
            insecure.append(f"MCP host={mcp.host}")
        if access.auth_mode == "none" and insecure:
            if not access.allow_insecure_no_auth:
                raise ValueError("auth_mode='none' with non-loopback listeners is refused")
            warnings.warn(
                "auth_mode='none' on non-loopback listeners (allow_insecure_no_auth=true)",
                stacklevel=2,
            )
        if access.auth_mode != "none" and access.cors_allow_origins == ("*",):
            warnings.warn(
                "auth_mode is enabled but wildcard CORS rejects credentials", stacklevel=2
            )
        if access.control.rules and access.auth_mode != "jwt":
            raise ValueError("access.control.rules require auth_mode='jwt'")

    @property
    def working_dir_path(self) -> Path:
        return Path(self.deployment.working_dir)

    @property
    def input_dir_path(self) -> Path:
        """Where operators place local sources, per workspace; DlightRAG only reads it."""
        return self.working_dir_path / "inputs"

    @property
    def corpus_dir_path(self) -> Path:
        """This service's own corpus files and Run stages: LightRAG's ``INPUT_DIR``."""
        return self.working_dir_path / "corpus"

    @property
    def is_reader(self) -> bool:
        return self.deployment.service_role == "reader"

    @property
    def max_upload_batch_bytes(self) -> int:
        return self.interfaces.max_upload_size_mb * 1024 * 1024

    @property
    def parser_rules(self) -> str:
        return self.corpus.parser_rules
