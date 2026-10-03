# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the Agent Browser's egress proxy admits, read from the Squid configuration it runs.

The pool's only way out is ``agent-browser/egress/squid.conf``, so its rules are the browser's
network boundary (ADR 0032). These tests evaluate that file the way Squid does, for the
requests that matter: which destinations and ports a pool container can reach.
"""

from __future__ import annotations

import ipaddress
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from dlightrag.engine.network_admission import _public_unicast

_CONF = Path(__file__).resolve().parents[2] / "agent-browser/egress/squid.conf"

type Network = ipaddress.IPv4Network | ipaddress.IPv6Network
type Address = ipaddress.IPv4Address | ipaddress.IPv6Address

#: A pool container's address on its Compose network, and the destinations that must not open.
POOL_CLIENT = "172.20.0.5"
NON_PUBLIC = [
    "127.0.0.1",
    "10.1.2.3",
    "172.20.0.5",
    "192.168.65.254",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",
    "224.0.0.1",
    "240.0.0.1",
    "::1",
    "fe80::1",
    "fc00::1",
]
PUBLIC = ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"]


@dataclass
class _Rule:
    action: Literal["allow", "deny"]
    acls: list[str]


@dataclass
class _Squid:
    """Just enough of squid.conf to decide a request: its ACLs and ordered access rules."""

    acls: dict[str, list[tuple[str, list[str]]]] = field(default_factory=dict)
    rules: list[_Rule] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> _Squid:
        squid = cls()
        for line in text.splitlines():
            words = line.split("#", 1)[0].split()
            if not words:
                continue
            if words[0] == "acl":
                squid.acls.setdefault(words[1], []).append((words[2], words[3:]))
            elif words[0] == "http_access":
                squid.rules.append(_Rule(words[1], words[2:]))  # type: ignore[arg-type]
        return squid

    def networks(self, acl: str, kind: str) -> list[Network]:
        return [
            ipaddress.ip_network(value)
            for acl_kind, values in self.acls[acl]
            if acl_kind == kind
            for value in values
        ]

    def decide(self, *, client: str, method: str, destination: str, port: int) -> str:
        """The first rule whose every ACL matches decides, as in Squid."""
        request = {"client": ipaddress.ip_address(client), "destination": destination}
        for rule in self.rules:
            if all(self._matches(name, request, method, port) for name in rule.acls):
                return rule.action
        return "deny"

    def _matches(
        self, name: str, request: dict[str, Address | str], method: str, port: int
    ) -> bool:
        if name.startswith("!"):
            return not self._matches(name[1:], request, method, port)
        if name == "all":
            return True
        destination = ipaddress.ip_address(request["destination"])
        for kind, values in self.acls[name]:
            if kind == "src" and any(
                request["client"] in ipaddress.ip_network(value) for value in values
            ):
                return True
            if kind == "dst" and any(
                _contains(ipaddress.ip_network(value), destination) for value in values
            ):
                return True
            if kind == "port" and str(port) in values:
                return True
            if kind == "method" and method in values:
                return True
        return False


def _contains(network: Network, address: Address) -> bool:
    return address.version == network.version and address in network


def _squid() -> _Squid:
    return _Squid.parse(_CONF.read_text(encoding="utf-8"))


@pytest.mark.parametrize("destination", NON_PUBLIC)
def test_no_non_public_destination_is_reachable_through_the_proxy(destination: str) -> None:
    squid = _squid()
    address = ipaddress.ip_address(destination)

    assert any(_contains(network, address) for network in squid.networks("non_public", "dst"))
    # The proxy refuses what the Host refuses: DlightRAG's own policy rejects it too.
    assert not _public_unicast(address)
    for method, port in (("GET", 80), ("GET", 443), ("CONNECT", 443)):
        assert (
            squid.decide(client=POOL_CLIENT, method=method, destination=destination, port=port)
            == "deny"
        )


@pytest.mark.parametrize("destination", PUBLIC)
def test_public_destinations_are_reachable_over_http_and_through_connect_to_443(
    destination: str,
) -> None:
    squid = _squid()
    address = ipaddress.ip_address(destination)

    assert not any(_contains(network, address) for network in squid.networks("non_public", "dst"))
    assert (
        squid.decide(client=POOL_CLIENT, method="GET", destination=destination, port=80) == "allow"
    )
    assert (
        squid.decide(client=POOL_CLIENT, method="GET", destination=destination, port=443) == "allow"
    )
    assert (
        squid.decide(client=POOL_CLIENT, method="CONNECT", destination=destination, port=443)
        == "allow"
    )


@pytest.mark.parametrize("port", [21, 22, 25, 3000, 3128, 5432, 8100, 8101, 8080, 8443])
def test_only_ports_80_and_443_are_admitted(port: int) -> None:
    squid = _squid()

    assert (
        squid.decide(client=POOL_CLIENT, method="GET", destination="8.8.8.8", port=port) == "deny"
    )
    assert (
        squid.decide(client=POOL_CLIENT, method="CONNECT", destination="8.8.8.8", port=port)
        == "deny"
    )


def test_connect_tunnels_only_to_the_tls_port() -> None:
    squid = _squid()

    assert (
        squid.decide(client=POOL_CLIENT, method="CONNECT", destination="8.8.8.8", port=80) == "deny"
    )
    assert (
        squid.decide(client=POOL_CLIENT, method="CONNECT", destination="8.8.8.8", port=443)
        == "allow"
    )


@pytest.mark.parametrize("client", ["8.8.8.8", "100.64.0.9", "2606:4700:4700::1111"])
def test_only_the_private_networks_compose_assigns_may_use_the_proxy(client: str) -> None:
    assert _squid().decide(client=client, method="GET", destination="1.1.1.1", port=80) == "deny"


def test_no_ipv6_block_contains_the_ipv4_mapped_range() -> None:
    """Squid maps IPv4 into ::ffff:0:0/96, so listing it would deny every IPv4 destination."""
    mapped = ipaddress.ip_network("::ffff:0:0/96")

    for network in _squid().networks("non_public", "dst"):
        if network.version == 6:
            assert not network.overlaps(mapped), network


def test_the_proxy_refuses_everything_dlightrag_itself_refuses_where_addresses_are_routed() -> None:
    """Across the address space, not only the listed examples: no gap between the two policies.

    IPv4 is sampled whole. IPv6 is sampled in global unicast space (2000::/3), which is where
    public addresses live, and in the blocks the configuration lists. The rest of IPv6 is
    unallocated, which ``_public_unicast`` refuses as reserved; Squid cannot list it, because
    its first block, ::/8, contains the IPv4-mapped range, so it is left to routing.
    """
    squid = _squid()
    generator = random.Random(2026)
    sampled: list[Address] = [
        ipaddress.ip_address(generator.getrandbits(32)) for _ in range(20_000)
    ]
    sampled += [
        ipaddress.ip_address((0b001 << 125) | generator.getrandbits(125)) for _ in range(2_000)
    ]
    for network in squid.networks("non_public", "dst"):
        sampled += [network[0], network[-1], network[generator.randrange(network.num_addresses)]]

    gaps = [
        address
        for address in sampled
        if not _public_unicast(address)
        and squid.decide(client=POOL_CLIENT, method="GET", destination=str(address), port=80)
        != "deny"
    ]

    assert gaps == []
