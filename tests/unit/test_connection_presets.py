# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Starter presets are vetted public endpoints, not an authority or a marketplace."""

from urllib.parse import urlsplit

from dlightrag.application.connections.policy import (
    credential_header_problem,
    endpoint_headers_problem,
)
from dlightrag.application.connections.presets import PRESETS
from dlightrag.application.connections.service import _PRESET_VIEWS
from dlightrag.engine.network_admission import (
    validate_credential_free_query,
    validate_public_http_url,
)


def test_every_preset_already_passes_the_create_time_endpoint_policy():
    assert PRESETS
    for preset in PRESETS:
        validate_public_http_url(preset.endpoint)
        validate_credential_free_query(preset.endpoint)
        parts = urlsplit(preset.endpoint)
        assert parts.scheme == "https"
        assert not parts.username and not parts.password
        assert not parts.fragment and not parts.query
        # The create route's own ceiling, so a preset can never be rejected for length alone.
        assert 1 <= len(preset.label) <= 100
        assert 1 <= len(preset.endpoint) <= 2048


def test_presets_are_identifiable_and_default_to_a_real_authentication_choice():
    assert len({preset.preset_id for preset in PRESETS}) == len(PRESETS)
    assert len({preset.endpoint for preset in PRESETS}) == len(PRESETS)
    for preset in PRESETS:
        assert preset.preset_id.isascii() and preset.preset_id.islower()
        assert preset.default_authentication in {"none", "bearer", "header", "oauth"}


def test_every_preset_states_only_headers_a_connection_may_state():
    """A preset fills the form, so what it fills must already pass the policy the form obeys."""
    for preset in PRESETS:
        assert endpoint_headers_problem(preset.endpoint_headers) is None
        assert credential_header_problem(preset.credential_header, preset.credential_prefix) is None
        if preset.default_authentication != "header":
            # The defaults are the bearer instance, and no preset renames it in passing.
            assert (preset.credential_header, preset.credential_prefix) == (
                "Authorization",
                "Bearer ",
            )


def test_the_view_projects_every_preset_unchanged():
    projected = [
        (
            view.preset_id,
            view.label,
            view.endpoint,
            view.default_authentication,
            view.endpoint_headers,
            view.credential_header,
            view.credential_prefix,
        )
        for view in _PRESET_VIEWS
    ]
    assert projected == [
        (
            preset.preset_id,
            preset.label,
            preset.endpoint,
            preset.default_authentication,
            preset.endpoint_headers,
            preset.credential_header,
            preset.credential_prefix,
        )
        for preset in PRESETS
    ]


def test_the_feishu_preset_is_data_and_nothing_else():
    """The one endpoint this capability was built for: a public URL, a vendor header, no code."""
    (feishu,) = [preset for preset in PRESETS if preset.preset_id == "feishu"]

    assert feishu.endpoint == "https://mcp.feishu.cn/mcp"
    assert feishu.default_authentication == "header"
    assert feishu.credential_header == "X-Lark-MCP-TAT"
    assert feishu.credential_prefix == ""
    # Without this header the server discovers nothing at all, so a preset without it would be
    # a one-click dead end -- and a preset listing only part of what the endpoint documents would
    # decide for the owner which capabilities they may use. Widening or narrowing stays theirs.
    assert feishu.endpoint_headers == (
        (
            "X-Lark-MCP-Allowed-Tools",
            "search-user,get-user,fetch-file,search-doc,create-doc,fetch-doc,update-doc,list-docs,get-comments,add-comments",
        ),
    )
