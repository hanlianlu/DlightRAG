# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Credential pool rotation: order, cooldown, and the one recovery probe."""

import pytest

from dlightrag.engine.credential_rotation import CredentialPool


def _first(pool: CredentialPool) -> int:
    """The first credential one call would ask; the walk stops at the first answer."""
    return next(pool.attempts()).index


def _asked(pool: CredentialPool) -> list[int]:
    return [credential.index for credential in pool.attempts()]


def test_rotation_asks_the_credential_leased_longest_ago_first() -> None:
    pool = CredentialPool(("one", "two", "three"))

    assert [_first(pool) for _ in range(4)] == [1, 2, 3, 1]


def test_a_cooled_credential_is_skipped_and_the_others_keep_rotating() -> None:
    now = [100.0]
    pool = CredentialPool(("one", "two"), clock=lambda: now[0])
    first = next(pool.attempts())
    pool.release(first, cooling=True)

    assert _asked(pool) == [2]

    now[0] += 61.0
    assert _asked(pool) == [1, 2], "the recovered key is back after its window"


def test_when_every_credential_is_cooling_one_probe_goes_out() -> None:
    now = [100.0]
    pool = CredentialPool(("one", "two"), clock=lambda: now[0])
    for credential in list(pool.attempts()):
        pool.release(credential, cooling=True)

    assert _asked(pool) == [1], "one probe, and the chain's own order settles a tie"


def test_the_probe_is_the_credential_that_recovers_first() -> None:
    now = [100.0]
    pool = CredentialPool(("one", "two"), clock=lambda: now[0])
    pool.release(pool.credentials[0], cooling=True)
    now[0] = 150.0
    pool.release(pool.credentials[1], cooling=True)

    assert _asked(pool) == [1], "the first window to end is the one to test"


def test_a_failed_probe_keeps_its_credential_parked_and_the_next_probe_moves_on() -> None:
    now = [100.0]
    pool = CredentialPool(("one", "two"), clock=lambda: now[0])
    pool.release(pool.credentials[0], cooling=True)
    now[0] = 150.0
    pool.release(pool.credentials[1], cooling=True)
    now[0] = 155.0
    probe = next(pool.attempts())
    pool.release(probe, cooling=True)

    now[0] = 161.0
    assert _asked(pool) == [2], "the untried key is asked once its own window ends"


def test_a_call_level_failure_does_not_park_the_credential() -> None:
    now = [100.0]
    pool = CredentialPool(("one", "two"), clock=lambda: now[0])
    first = next(pool.attempts())
    pool.release(first, cooling=False)

    assert _asked(pool) == [2, 1]


def test_a_success_clears_a_parked_credential() -> None:
    now = [100.0]
    pool = CredentialPool(("one", "two"), clock=lambda: now[0])
    first = next(pool.attempts())
    pool.release(first, cooling=True)
    pool.release(first, cooling=False)

    assert _asked(pool) == [2, 1]


def test_a_pool_of_one_yields_one_attempt_and_no_rotation() -> None:
    assert _asked(CredentialPool(("only",))) == [1]


def test_an_empty_pool_yields_nothing() -> None:
    assert list(CredentialPool(()).attempts()) == []


def test_duplicate_secrets_collapse_and_blank_ones_are_refused() -> None:
    pool = CredentialPool(("one", " two ", "one"))

    assert [(item.index, item.secret) for item in pool.credentials] == [(1, "one"), (2, "two")]
    with pytest.raises(ValueError, match="blank secret"):
        CredentialPool(("one", "  "))


def test_a_secret_never_renders_in_a_repr() -> None:
    pool = CredentialPool(("never-echo-me",))

    assert "never-echo-me" not in repr(pool.credentials[0])
    assert "never-echo-me" not in repr(next(pool.attempts()))
