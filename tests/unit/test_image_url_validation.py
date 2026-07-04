# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for image URL validation (SSRF guard)."""

from dlightrag.core.answer_images import (
    AnswerImageBudget,
    _is_unsafe_host,
    _validate_image_url,
)


class TestIsUnsafeHost:
    def test_loopback_v4(self) -> None:
        assert _is_unsafe_host("127.0.0.1") is True
        assert _is_unsafe_host("127.255.255.255") is True

    def test_loopback_v6(self) -> None:
        assert _is_unsafe_host("::1") is True

    def test_private_rfc1918(self) -> None:
        assert _is_unsafe_host("10.0.0.1") is True
        assert _is_unsafe_host("172.16.0.1") is True
        assert _is_unsafe_host("192.168.1.1") is True

    def test_link_local(self) -> None:
        assert _is_unsafe_host("169.254.1.1") is True

    def test_public_ip_is_safe(self) -> None:
        assert _is_unsafe_host("8.8.8.8") is False
        assert _is_unsafe_host("1.1.1.1") is False

    def test_hostname_is_safe(self) -> None:
        """Non-IP hostnames pass — DNS resolution is the provider's job."""
        assert _is_unsafe_host("example.com") is False
        assert _is_unsafe_host("api.openai.com") is False

    def test_trailing_dot_bypass(self) -> None:
        """Trailing dot (DNS FQDN marker) must not bypass IP check."""
        assert _is_unsafe_host("127.0.0.1.") is True
        assert _is_unsafe_host("10.0.0.1.") is True
        assert _is_unsafe_host("169.254.169.254.") is True
        assert _is_unsafe_host("8.8.8.8.") is False

    def test_unspecified_address(self) -> None:
        """0.0.0.0 and :: are unspecified — must be rejected."""
        assert _is_unsafe_host("0.0.0.0") is True
        assert _is_unsafe_host("::") is True

    def test_hex_ip_representation(self) -> None:
        """Non-standard hex IP (e.g. 0x7f000001 = 127.0.0.1) — reject."""
        assert _is_unsafe_host("0x7f000001") is True

    def test_integer_ip_representation(self) -> None:
        """DWORD integer IP (e.g. 2130706433 = 127.0.0.1) — reject."""
        assert _is_unsafe_host("2130706433") is True

    def test_none_is_safe(self) -> None:
        assert _is_unsafe_host(None) is False


class TestValidateImageUrl:
    def test_data_uri_passes(self) -> None:
        assert _validate_image_url("data:image/png;base64,abc", label="t1") is not None

    def test_dlightrag_image_passes(self) -> None:
        assert _validate_image_url("dlightrag-image://img_123", label="t2") is not None

    def test_https_passes(self) -> None:
        assert _validate_image_url("https://example.com/img.png", label="t3") is not None

    def test_http_rejected(self) -> None:
        assert _validate_image_url("http://example.com/img.png", label="t4") is None

    def test_file_scheme_rejected(self) -> None:
        assert _validate_image_url("file:///etc/passwd", label="t5") is None

    def test_no_scheme_rejected(self) -> None:
        assert _validate_image_url("just/a/path.png", label="t6") is None

    def test_https_localhost_ip_rejected(self) -> None:
        assert _validate_image_url("https://127.0.0.1/admin", label="t7") is None

    def test_https_private_ip_rejected(self) -> None:
        assert _validate_image_url("https://10.0.0.1/internal", label="t8") is None
        assert _validate_image_url("https://192.168.1.1/debug", label="t9") is None

    def test_https_aws_metadata_ip_rejected(self) -> None:
        assert _validate_image_url("https://169.254.169.254/latest/meta-data/", label="t10") is None

    def test_https_public_ip_passes(self) -> None:
        assert _validate_image_url("https://8.8.8.8/photo.jpg", label="t11") is not None


class TestAnswerImageBudgetUrlValidation:
    def test_http_url_rejected_in_add_user_image(self) -> None:
        budget = AnswerImageBudget(
            max_images=3,
            max_total_bytes=12_000_000,
            max_bytes_per_image=3_000_000,
            max_px=1536,
            min_px=1024,
            quality=89,
            min_quality=79,
        )
        result = budget.add_user_image("http://evil.com/ssrf-target", label="x1")
        assert result is None

    def test_https_private_ip_rejected_in_add_user_image(self) -> None:
        budget = AnswerImageBudget(
            max_images=3,
            max_total_bytes=12_000_000,
            max_bytes_per_image=3_000_000,
            max_px=1536,
            min_px=1024,
            quality=89,
            min_quality=79,
        )
        result = budget.add_user_image("https://169.254.169.254/latest/meta-data/", label="x2")
        assert result is None

    def test_data_uri_passes_in_add_user_image(self) -> None:
        budget = AnswerImageBudget(
            max_images=3,
            max_total_bytes=12_000_000,
            max_bytes_per_image=3_000_000,
            max_px=1536,
            min_px=1024,
            quality=89,
            min_quality=79,
        )
        result = budget.add_user_image(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            label="x3",
        )
        assert result is not None
        assert result["type"] == "image_url"

    def test_https_domain_passes_in_add_user_image(self) -> None:
        budget = AnswerImageBudget(
            max_images=3,
            max_total_bytes=12_000_000,
            max_bytes_per_image=3_000_000,
            max_px=1536,
            min_px=1024,
            quality=89,
            min_quality=79,
        )
        result = budget.add_user_image("https://cdn.example.com/chart.png", label="x4")
        assert result is not None
        assert result["image_url"]["url"] == "https://cdn.example.com/chart.png"

    def test_http_rejected_in_dict_block(self) -> None:
        budget = AnswerImageBudget(
            max_images=3,
            max_total_bytes=12_000_000,
            max_bytes_per_image=3_000_000,
            max_px=1536,
            min_px=1024,
            quality=89,
            min_quality=79,
        )
        result = budget.add_user_image(
            {"type": "image_url", "image_url": {"url": "http://169.254.169.254/"}},
            label="x5",
        )
        assert result is None

    def test_https_private_ip_rejected_in_dict_block(self) -> None:
        budget = AnswerImageBudget(
            max_images=3,
            max_total_bytes=12_000_000,
            max_bytes_per_image=3_000_000,
            max_px=1536,
            min_px=1024,
            quality=89,
            min_quality=79,
        )
        result = budget.add_user_image(
            {"type": "image_url", "image_url": {"url": "https://10.0.0.1/debug"}},
            label="x6",
        )
        assert result is None

    def test_dlightrag_image_passes_in_dict_block(self) -> None:
        budget = AnswerImageBudget(
            max_images=3,
            max_total_bytes=12_000_000,
            max_bytes_per_image=3_000_000,
            max_px=1536,
            min_px=1024,
            quality=89,
            min_quality=79,
        )
        result = budget.add_user_image(
            {"type": "image_url", "image_url": {"url": "dlightrag-image://img_123"}},
            label="x7",
        )
        assert result is not None
