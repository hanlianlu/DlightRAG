"""Tests for Azure SAS URL generation — verifies connection string parsing and SAS dispatch."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from dlightrag.sourcing.azure_blob import generate_azure_sas_url

CONN_STR = "DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net"


class TestGenerateAzureSasUrl:

    def test_rejects_non_azure_path(self) -> None:
        """Guard: local paths must not reach SAS generation."""
        with pytest.raises(ValueError, match="azure://"):
            generate_azure_sas_url(connection_string=CONN_STR, raw_path="/local/path.pdf")

    @patch("dlightrag.sourcing.azure_blob.generate_blob_sas", return_value="sv=2024&sig=abc")
    def test_parses_connection_string_and_generates_url(self, mock_sas) -> None:
        """Core: connection string → account creds, azure:// path → container/blob, output → signed HTTPS URL."""
        url = generate_azure_sas_url(
            connection_string=CONN_STR,
            raw_path="azure://mycontainer/reports/q1.pdf",
            expiry_seconds=3600,
        )
        assert url == "https://myaccount.blob.core.windows.net/mycontainer/reports/q1.pdf?sv=2024&sig=abc"
        kw = mock_sas.call_args[1]
        assert kw["account_name"] == "myaccount"
        assert kw["container_name"] == "mycontainer"
        assert kw["blob_name"] == "reports/q1.pdf"
