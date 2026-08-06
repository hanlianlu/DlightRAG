# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MetadataFilter data model."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from dlightrag.api.models import MetadataFilterRequest
from dlightrag.core.client_payloads import metadata_filter_from_payload
from dlightrag.core.retrieval.models import MetadataFilter


class TestMetadataFilter:
    def test_empty_filter(self) -> None:
        f = MetadataFilter()
        assert f.is_empty()

    def test_filename_makes_non_empty(self) -> None:
        f = MetadataFilter(filename="test.pdf")
        assert not f.is_empty()

    def test_custom_metadata_makes_non_empty(self) -> None:
        f = MetadataFilter(custom={"department": "finance"})
        assert not f.is_empty()

    def test_file_extension_makes_non_empty(self) -> None:
        f = MetadataFilter(file_extension=".png")
        assert not f.is_empty()

    def test_file_extension_is_normalized(self) -> None:
        f = MetadataFilter(file_extension=" .PDF ")
        assert f.file_extension == "pdf"

    def test_text_filters_are_trimmed(self) -> None:
        f = MetadataFilter(
            filename=" report.pdf ",
            title=" Annual Report ",
            author=" Zhang San ",
        )

        assert f.filename == "report.pdf"
        assert f.title == "Annual Report"
        assert f.author == "Zhang San"

    def test_date_range_makes_non_empty(self) -> None:
        f = MetadataFilter(creation_date_from=datetime(2024, 1, 1))
        assert not f.is_empty()

    def test_all_none_is_empty(self) -> None:
        f = MetadataFilter(
            filename=None,
            file_extension=None,
            title=None,
            author=None,
            creation_date_from=None,
            creation_date_to=None,
            custom=None,
        )
        assert f.is_empty()


class TestDateNormalization:
    """Callers may write an offset or omit one; storage holds one instant."""

    @staticmethod
    def _parse(**payload: str) -> MetadataFilter:
        # Mirrors a JSON request body rather than in-process construction.
        return MetadataFilter.model_validate(payload)

    def test_offset_is_converted_to_utc(self) -> None:
        parsed = self._parse(creation_date_from="2024-01-01T08:00:00+08:00").creation_date_from

        assert parsed == datetime(2024, 1, 1, 0, 0)
        assert parsed is not None and parsed.tzinfo is None

    def test_bare_timestamp_is_read_as_utc_not_local(self) -> None:
        assert self._parse(creation_date_from="2024-01-01T12:30:00").creation_date_from == datetime(
            2024, 1, 1, 12, 30
        )

    def test_date_only_input_keeps_midnight(self) -> None:
        assert self._parse(creation_date_to="2024-03-05").creation_date_to == datetime(
            2024, 3, 5, 0, 0
        )

    def test_negative_offset_is_converted(self) -> None:
        west = timezone(-timedelta(hours=5))
        f = MetadataFilter(creation_date_from=datetime(2024, 1, 1, 0, 0, tzinfo=west))

        assert f.creation_date_from == datetime(2024, 1, 1, 5, 0)

    def test_aware_utc_input_loses_only_the_marker(self) -> None:
        f = MetadataFilter(creation_date_from=datetime(2024, 1, 1, 9, 0, tzinfo=UTC))

        assert f.creation_date_from == datetime(2024, 1, 1, 9, 0)

    @pytest.mark.parametrize("value", ["2024/01/01", "not-a-date", "Q1 2024"])
    def test_unparseable_input_is_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            self._parse(creation_date_from=value)


class TestRestRequestBoundary:
    """A bad date must fail request validation, not surface as a 500 later."""

    def test_rest_contract_parses_dates(self) -> None:
        req = MetadataFilterRequest.model_validate(
            {"creation_date_from": "2024-01-01T00:00:00+08:00"}
        )

        assert req.creation_date_from == datetime(
            2024, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=8))
        )

    @pytest.mark.parametrize("value", ["not-a-date", "2024/01/01"])
    def test_rest_contract_rejects_unparseable_dates(self, value: str) -> None:
        with pytest.raises(ValidationError):
            MetadataFilterRequest.model_validate({"creation_date_from": value})

    def test_rest_payload_normalizes_when_converted_to_filter(self) -> None:
        payload = MetadataFilterRequest.model_validate(
            {"creation_date_from": "2024-01-01T08:00:00+08:00"}
        )

        converted = metadata_filter_from_payload(payload)

        assert converted is not None
        assert converted.creation_date_from == datetime(2024, 1, 1, 0, 0)
