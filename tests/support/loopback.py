# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Loopback endpoints that fail the way real ones do, for transport classification tests.

A constructed exception chain lacks what a real client builds, such as the
implicit context an interrupted TLS handshake leaves behind, so these servers let
a test drive httpcore, anyio, aiohttp, and the provider SDKs through a real
failure on 127.0.0.1 without reaching the network.
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import socket
import ssl
import struct
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

type ConnectionHandler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


@asynccontextmanager
async def loopback_server(
    handler: ConnectionHandler,
    *,
    tls: ssl.SSLContext | None = None,
) -> AsyncIterator[int]:
    """Serve ``handler`` on a loopback port and yield the port."""
    server = await asyncio.start_server(handler, "127.0.0.1", 0, ssl=tls)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()
        await server.wait_closed()


_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def bypass_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep loopback requests off every proxy, the macOS system proxy included.

    A client that trusts its environment follows HTTP(S)_PROXY, and on macOS the
    system proxy settings when no proxy variable is set; NO_PROXY=* makes every
    host bypass both.
    """
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")


async def reset_on_accept(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Reset the connection (RST, not FIN) as soon as it is accepted."""
    connection = writer.get_extra_info("socket")
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


def tls_error(reason: str) -> ssl.SSLError:
    """An SSLError carrying an OpenSSL ``reason``, which a constructed one lacks."""
    error = ssl.SSLError(1, f"[SSL: {reason}] {reason.lower().replace('_', ' ')}")
    error.reason = reason
    return error


@dataclass(frozen=True, slots=True)
class LoopbackCertificate:
    """A self-signed certificate for 127.0.0.1 that only its own clients trust."""

    certificate: Path
    key: Path

    def server_context(
        self,
        *,
        minimum: ssl.TLSVersion | None = None,
        maximum: ssl.TLSVersion | None = None,
        ciphers: str | None = None,
    ) -> ssl.SSLContext:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(self.certificate, self.key)
        return _limit(context, minimum=minimum, maximum=maximum, ciphers=ciphers)

    def client_context(
        self,
        *,
        minimum: ssl.TLSVersion | None = None,
        maximum: ssl.TLSVersion | None = None,
        ciphers: str | None = None,
    ) -> ssl.SSLContext:
        context = ssl.create_default_context(cafile=str(self.certificate))
        return _limit(context, minimum=minimum, maximum=maximum, ciphers=ciphers)


def loopback_certificate(directory: Path) -> LoopbackCertificate:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        # Its own trust anchor, so a client can trust it explicitly.
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    certificate_path = directory / "loopback-certificate.pem"
    key_path = directory / "loopback-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return LoopbackCertificate(certificate=certificate_path, key=key_path)


def tls_handshake_succeeds(server: ssl.SSLContext, client: ssl.SSLContext) -> bool:
    """Whether two contexts complete a TLS handshake, run in memory without sockets."""
    server_in, server_out, client_in, client_out = (ssl.MemoryBIO() for _ in range(4))
    sides = (
        client.wrap_bio(client_in, client_out, server_hostname="127.0.0.1"),
        server.wrap_bio(server_in, server_out, server_side=True),
    )
    done = [False, False]
    for _ in range(10):
        for index, side in enumerate(sides):
            if done[index]:
                continue
            try:
                side.do_handshake()
            except ssl.SSLWantReadError:
                continue
            except ssl.SSLError:
                return False
            done[index] = True
        if all(done):
            return True
        server_in.write(client_out.read())
        client_in.write(server_out.read())
    return False


@contextmanager
def alerting_tls_server(context: ssl.SSLContext) -> Iterator[int]:
    """Serve TLS handshakes from a thread that sends OpenSSL's alert on failure.

    asyncio's TLS server drops the alert of a failed handshake, so its client only
    sees an EOF; a blocking server flushes the alert, as a real endpoint does.
    """
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.05)
    stopping = threading.Event()

    def serve() -> None:
        while not stopping.is_set():
            try:
                connection, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                connection.settimeout(5)
                try:
                    with context.wrap_socket(connection, server_side=True) as tls:
                        tls.recv(1024)
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        stopping.set()
        thread.join(5)
        listener.close()


def _limit(
    context: ssl.SSLContext,
    *,
    minimum: ssl.TLSVersion | None,
    maximum: ssl.TLSVersion | None,
    ciphers: str | None,
) -> ssl.SSLContext:
    if minimum is not None:
        context.minimum_version = minimum
    if maximum is not None:
        context.maximum_version = maximum
    if ciphers is not None:
        context.set_ciphers(ciphers)
    return context


__all__ = [
    "ConnectionHandler",
    "LoopbackCertificate",
    "alerting_tls_server",
    "bypass_proxies",
    "loopback_certificate",
    "loopback_server",
    "reset_on_accept",
    "tls_error",
    "tls_handshake_succeeds",
]
