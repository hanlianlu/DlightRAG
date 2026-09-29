# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The PostgreSQL adapter turns the typed settings into asyncpg's connection arguments."""

import datetime
import ssl
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from dlightrag.adapters.postgres.core._connection import pg_connection_kwargs
from dlightrag.application.config import PostgresSettings
from tests.support.loopback import (
    LoopbackCertificate,
    loopback_certificate,
    tls_handshake_succeeds,
)


def _context(**settings: Any) -> ssl.SSLContext:
    context = pg_connection_kwargs(PostgresSettings(**settings))["ssl"]
    assert isinstance(context, ssl.SSLContext)
    return context


def test_the_endpoint_and_credentials_connect_without_tls_by_default() -> None:
    settings = PostgresSettings(host="db", port=5433, user="u", password="p", database="d")

    assert pg_connection_kwargs(settings) == {
        "host": "db",
        "port": 5433,
        "user": "u",
        "password": "p",
        "database": "d",
    }


@pytest.mark.parametrize(
    ("mode", "ssl_argument"),
    [("allow", "absent"), ("prefer", True), ("require", True), ("disable", False)],
)
def test_modes_without_verification_keep_asyncpgs_own_values(
    mode: Any, ssl_argument: object
) -> None:
    kwargs = pg_connection_kwargs(PostgresSettings(ssl_mode=mode))

    assert kwargs.get("ssl", "absent") == ssl_argument


@pytest.mark.parametrize(("mode", "checks_hostname"), [("verify-ca", False), ("verify-full", True)])
def test_verifying_modes_trust_the_configured_root_certificate(
    tmp_path: Path, mode: str, checks_hostname: bool
) -> None:
    certificate = loopback_certificate(tmp_path)

    trusting = _context(ssl_mode=mode, ssl_root_cert=str(certificate.certificate))

    assert trusting.verify_mode == ssl.CERT_REQUIRED
    assert trusting.check_hostname is checks_hostname
    assert tls_handshake_succeeds(certificate.server_context(), trusting)
    assert not tls_handshake_succeeds(certificate.server_context(), _context(ssl_mode=mode))


def test_a_configured_client_certificate_is_presented(tmp_path: Path) -> None:
    certificate = loopback_certificate(tmp_path)
    server = certificate.server_context()
    server.verify_mode = ssl.CERT_REQUIRED
    server.load_verify_locations(cafile=str(certificate.certificate))
    root = {"ssl_mode": "verify-full", "ssl_root_cert": str(certificate.certificate)}

    presenting = _context(
        **root, ssl_cert=str(certificate.certificate), ssl_key=str(certificate.key)
    )

    assert tls_handshake_succeeds(server, presenting)
    assert not tls_handshake_succeeds(server, _context(**root))


def test_a_configured_crl_is_checked_for_the_server_certificate(tmp_path: Path) -> None:
    certificate = loopback_certificate(tmp_path)

    context = _context(
        ssl_mode="verify-full",
        ssl_root_cert=str(certificate.certificate),
        ssl_crl=str(_empty_crl(certificate, tmp_path / "root.crl")),
    )

    assert context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF
    # A leaf CRL check fails without its issuer's CRL, so passing proves the file loaded.
    assert tls_handshake_succeeds(certificate.server_context(), context)


def test_absent_tls_files_are_skipped_as_lightrag_skips_them(tmp_path: Path) -> None:
    """Both pools resolve the same bindings, so they never disagree about TLS."""
    missing = str(tmp_path / "missing.pem")

    context = _context(
        ssl_mode="verify-ca",
        ssl_root_cert=missing,
        ssl_cert=missing,
        ssl_key=missing,
        ssl_crl=missing,
    )

    assert not context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF


def test_an_unreadable_tls_file_fails_as_a_configuration_error(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.pem"
    garbage.write_text("not a certificate")

    with pytest.raises(ValueError, match="PostgreSQL SSL configuration error"):
        pg_connection_kwargs(PostgresSettings(ssl_mode="verify-ca", ssl_root_cert=str(garbage)))


def _empty_crl(certificate: LoopbackCertificate, path: Path) -> Path:
    issuer = x509.load_pem_x509_certificate(certificate.certificate.read_bytes())
    key = serialization.load_pem_private_key(certificate.key.read_bytes(), password=None)
    assert isinstance(key, ec.EllipticCurvePrivateKey)
    now = datetime.datetime.now(datetime.UTC)
    crl = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(issuer.subject)
        .last_update(now - datetime.timedelta(minutes=1))
        .next_update(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    path.write_bytes(crl.public_bytes(serialization.Encoding.PEM))
    return path
