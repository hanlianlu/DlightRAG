# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Starter endpoints the create form offers as one-click fills.

A Preset is product-owned copy plus an endpoint and the authentication choice that suits its
tier: a service that answers unauthenticated starts on ``none``, and one that cannot be used
without an account starts on the choice that account needs. A preset may also carry the
non-secret headers its endpoint needs stated, and the header a ``header`` credential travels in
by default, so an endpoint that authenticates with a vendor header is still one click away.
Choosing a Preset never creates, enables, or authorizes anything -- it only fills the create
form, and the ordinary create command remains the one authority that validates the endpoint and
owns the Connection.

This list is small, static, code-owned, credential-free, and reviewed by hand. It is
deliberately not a marketplace, a registry lookup, or a discovery surface: the plan's exclusion
of a Connection marketplace still holds.
"""

from dataclasses import dataclass

from .policy import Authentication, StatedHeaders


@dataclass(frozen=True)
class ConnectionPreset:
    """One starter endpoint and the authentication tab that fits it."""

    preset_id: str
    label: str
    endpoint: str
    default_authentication: Authentication
    #: Headers every request to this endpoint must state, as [name, value] pairs. Never secrets.
    endpoint_headers: StatedHeaders = ()
    #: The header a ``header`` credential travels in, and the prefix its value carries, when the
    #: tab above is ``header``: the tab's own defaults, not a credential.
    credential_header: str = "Authorization"
    credential_prefix: str = "Bearer "


PRESETS: tuple[ConnectionPreset, ...] = (
    ConnectionPreset("notion", "Notion", "https://mcp.notion.com/mcp", "oauth"),
    ConnectionPreset("figma", "Figma", "https://mcp.figma.com/mcp", "oauth"),
    ConnectionPreset("alphaxiv", "alphaXiv", "https://api.alphaxiv.org/mcp/v1", "oauth"),
    ConnectionPreset("huggingface", "Hugging Face", "https://huggingface.co/mcp", "none"),
    ConnectionPreset("wolfram", "Wolfram", "https://agenttools.wolfram.com/mcp", "none"),
    # Feishu/Lark's remote MCP takes two headers and no OAuth: a tenant (or user) access token
    # in a header of its own, with no prefix, and a tool allowlist without which the server
    # discovers nothing at all. No wildcard is documented, so the list below names every tool
    # the endpoint documents today -- its three general tools and its seven Docs tools. A
    # shorter list would decide for the owner which of the server's capabilities they may use,
    # and the provider keeps adding tools, so widening it stays the owner's edit. The server
    # filters whatever the credential cannot call anyway: `search-user` needs a user token.
    #
    # The remote endpoint covers the Docs scenario. Messaging and calendar tools live in the
    # provider's local server, which this deployment does not run: MCP transport is Streamable
    # HTTP only, so no preset can reach a tool this endpoint does not publish.
    ConnectionPreset(
        "feishu",
        "Feishu/Lark",
        "https://mcp.feishu.cn/mcp",
        "header",
        endpoint_headers=(
            (
                "X-Lark-MCP-Allowed-Tools",
                "search-user,get-user,fetch-file,search-doc,create-doc,fetch-doc,update-doc,list-docs,get-comments,add-comments",
            ),
        ),
        credential_header="X-Lark-MCP-TAT",
        credential_prefix="",
    ),
    # Kuaicha takes its credential in a vendor header of its own rather than in `Authorization`:
    # the provider's own client config states the API key as `open-authorization: Bearer <key>`.
    # Its handshake and `tools/list` answer without one -- a preset left on `none` would discover
    # two tools and then fail every call with 401 -- so the tier is `header`, and the key the
    # provider issues belongs in the tab the form opens on. The endpoint is `bizveris`, not the
    # `open` host the provider documents from: `open.kuaicha365.com/mcp/` serves its HTML page,
    # and the trailing-slash route on `bizveris` refuses even the handshake.
    ConnectionPreset(
        "kuaicha",
        "Kuaicha (快查)",
        "https://bizveris.kuaicha365.com/mcp",
        "header",
        credential_header="open-authorization",
        credential_prefix="Bearer ",
    ),
)
