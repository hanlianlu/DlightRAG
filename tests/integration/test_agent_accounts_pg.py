# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Accounts in PostgreSQL: owner-scoped rows, and their re-sealing after a key rotation.

Every test owns a scratch database. A generated password is compared in code and never put into
an assertion, so a failure reports a count or a flag and never the value.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from pydantic import SecretStr

from dlightrag.adapters.postgres.answer.agent_accounts import PGAgentAccountStore
from dlightrag.engine.answer.agent_browser import (
    ACCOUNT_LABEL,
    AgentAccountsBinding,
    RunAgentAccounts,
    StoredAgentAccount,
    generate_password,
    owner_alias,
    reseal_agent_accounts,
)
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_agent_accounts_browser import (
    DOMAIN,
    KEYRING,
    SHOP,
    SITE,
    assert_sent,
    browsing,
    posted,
)
from tests.support.agent_browser import StubMailbox
from tests.support.dns import public_dns
from tests.support.pg import skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

KEYS = {
    "test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE=",
    "next": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI=",
    "gone": "Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2M=",
}


def ring(active: str, *keys: str) -> CredentialCipher:
    ring = {"active": active, "keys": {key: KEYS[key] for key in keys}}
    return CredentialCipher(SecretStr(json.dumps(ring)))


@pytest.fixture(autouse=True)
async def _postgres() -> None:
    await skip_without_postgres()


@pytest.fixture(autouse=True)
def _hosts_resolve_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    async with isolated_run_runtime("agent_accounts") as (_, database):
        yield database


async def register(
    store: PGAgentAccountStore, cipher: CredentialCipher, owner: str, site: str
) -> SecretStr:
    """The owner's parent registers on the site, and the password sealed for it comes back."""
    run = RunAgentAccounts(owner_id=owner, binding=AgentAccountsBinding(store, cipher))
    password = generate_password()
    await run.record(
        "parent",
        site,
        child=False,
        existing=await run.registration_target("parent", site, child=False),
        email=f"{owner}@alias.example",
        username=None,
        password=password,
    )
    return password


def sealed(
    cipher: CredentialCipher, owner: str, site: str, account_id: str, *, bound_to: str
) -> Any:
    """A stored account whose envelope is sealed for the account ``bound_to`` names."""
    key_id, envelope = cipher.seal(
        generate_password(), label=ACCOUNT_LABEL, binding=(owner, site, bound_to)
    )
    return StoredAgentAccount(
        owner, site, account_id, f"{owner}@alias.example", None, key_id, envelope
    )


async def row_of(pool: Any, owner: str, site: str) -> Any:
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT * FROM dlightrag_agent_accounts WHERE owner_id = $1 AND site = $2", owner, site
        )


async def test_accounts_are_owner_scoped_and_a_reset_replaces_the_envelope_of_the_same_account(
    pool: Any,
) -> None:
    store, cipher = PGAgentAccountStore(pool=pool), ring("test", "test")

    first = await register(store, cipher, "alice", "shop.example")
    await register(store, cipher, "alice", "other.example")
    await register(store, cipher, "bob", "shop.example")

    assert await store.account(owner_id="carol", site="shop.example") is None
    alice = await store.account(owner_id="alice", site="shop.example")
    bob = await store.account(owner_id="bob", site="shop.example")
    assert alice is not None and bob is not None
    assert (alice.email, bob.email) == ("alice@alias.example", "bob@alias.example")
    assert alice.account_id != bob.account_id

    reset = await register(store, cipher, "alice", "shop.example")

    renewed = await store.account(owner_id="alice", site="shop.example")
    assert renewed is not None and renewed.account_id == alice.account_id
    assert renewed.envelope != alice.envelope
    binding = ("alice", "shop.example", alice.account_id)
    assert cipher.open(renewed.envelope, label=ACCOUNT_LABEL, binding=binding) == reset
    assert reset != first
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM dlightrag_agent_accounts") == 3


async def test_an_account_names_an_email_or_a_username(pool: Any) -> None:
    store = PGAgentAccountStore(pool=pool)

    await store.save(StoredAgentAccount("alice", "a.example", "id", None, "handle", "test", "{}"))
    with pytest.raises(asyncpg.CheckViolationError):
        await store.save(StoredAgentAccount("alice", "b.example", "id", None, None, "test", "{}"))


async def test_reseal_moves_retired_envelopes_and_skips_ones_no_key_opens(pool: Any) -> None:
    store, old = PGAgentAccountStore(pool=pool), ring("test", "test")
    readable = {
        owner: (site, await register(store, old, owner, site))
        for owner, site in (("alice", "one.example"), ("bob", "two.example"))
    }
    # One envelope sealed under a key the rotated ring no longer holds, and one under a key it
    # holds but sealed for another account, so it opens for nobody.
    await store.save(sealed(ring("gone", "gone"), "carol", "lost.example", "c", bound_to="c"))
    await store.save(sealed(old, "dave", "bad.example", "d", bound_to="someone-else"))
    before = {
        (o, s): await row_of(pool, o, s)
        for o, s in [("alice", "one.example"), ("bob", "two.example")]
    }

    rotated = ring("next", "test", "next")
    resealed = await reseal_agent_accounts(store, rotated)

    assert resealed == 2
    for owner, (site, password) in readable.items():
        row, was = await row_of(pool, owner, site), before[(owner, site)]
        assert row["key_id"] == "next" and row["encrypted_envelope"] != was["encrypted_envelope"]
        # The account is the same one, only sealed anew: nothing the owner did to it moved.
        assert (row["account_id"], row["updated_at"]) == (was["account_id"], was["updated_at"])
        run = RunAgentAccounts(owner_id=owner, binding=AgentAccountsBinding(store, rotated))
        account = await run.login_target("parent", site, child=False)
        assert account is not None and run.password(account) == password
    assert (await row_of(pool, "carol", "lost.example"))["key_id"] == "gone"
    assert (await row_of(pool, "dave", "bad.example"))["key_id"] == "test"
    # What is left under a retired key opens for nobody, and a second pass moves nothing.
    assert await reseal_agent_accounts(store, rotated) == 0


async def test_a_reset_saved_between_the_read_and_the_reseal_wins(pool: Any) -> None:
    store, old = PGAgentAccountStore(pool=pool), ring("test", "test")
    await register(store, old, "alice", "shop.example")
    rotated = ring("next", "test", "next")
    (read,) = await store.sealed_under(key_ids=rotated.retired_key_ids, limit=10)
    binding = (read.owner_id, read.site, read.account_id)
    password = rotated.open(read.envelope, label=ACCOUNT_LABEL, binding=binding)

    # The owner resets the account after the maintenance pass read it, and before it wrote.
    reset = await register(store, rotated, "alice", "shop.example")
    key_id, envelope = rotated.seal(password, label=ACCOUNT_LABEL, binding=binding)
    changed = await store.reseal(read, key_id=key_id, envelope=envelope)

    assert changed is False
    current = await store.account(owner_id="alice", site="shop.example")
    assert current is not None and current.envelope != envelope
    assert rotated.open(current.envelope, label=ACCOUNT_LABEL, binding=binding) == reset


async def test_a_later_run_logs_in_with_the_account_a_parent_registered(
    pool: Any, tmp_path: Path
) -> None:
    cipher = CredentialCipher(SecretStr(KEYRING))
    alias = owner_alias("owner", SITE, DOMAIN)
    async with browsing(
        tmp_path, store=PGAgentAccountStore(pool=pool), mailbox=StubMailbox(DOMAIN)
    ) as first:
        await first.register(await first.form(f"{SHOP}/signup"), email=None)
    registered = await row_of(pool, "owner", SITE)
    binding = ("owner", SITE, registered["account_id"])
    password = cipher.open(registered["encrypted_envelope"], label=ACCOUNT_LABEL, binding=binding)

    # Another Run of the owner, which only the database connects to the first.
    async with browsing(
        tmp_path, store=PGAgentAccountStore(pool=pool), mailbox=StubMailbox(DOMAIN)
    ) as later:
        later.watch(password)
        form = await later.form(f"{SHOP}/signin")
        await later.call(action="login", email_ref=form["email"], password_refs=[form["password"]])
        await later.call(action="click", ref=form["button"])

        fields = posted(later.proxy, "/session")
        assert fields["email"] == [alias]
        assert_sent(fields, password, "password")
    after = await row_of(pool, "owner", SITE)
    # The account is the alias the first Run minted, and logging in changed nothing about it.
    assert registered["email"] == alias
    assert {k: after[k] for k in after.keys()} == {k: registered[k] for k in registered.keys()}
