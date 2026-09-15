# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Starter presets are vetted public endpoints, not an authority or a marketplace."""

from urllib.parse import urlsplit

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
        assert preset.default_authentication in {"none", "bearer", "oauth"}


def test_the_view_projects_every_preset_unchanged():
    projected = [
        (view.preset_id, view.label, view.endpoint, view.default_authentication)
        for view in _PRESET_VIEWS
    ]
    assert projected == [
        (preset.preset_id, preset.label, preset.endpoint, preset.default_authentication)
        for preset in PRESETS
    ]
