# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Runtime Policies: the settings an administrator changes while the deployment runs.

Configuration is read when a process starts. A Runtime Policy is read when a request needs it,
from PostgreSQL, so a change holds on the next request of every process, with no restart and no
rebuild. One document is kept per policy and replaced whole, and the application knows each
policy by its key: the shape its document must have, and the value that holds until one is
written.
"""

import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Protocol, cast

from pydantic import Field, StrictBool, ValidationError

from dlightrag.application.access.control import AccessSubject
from dlightrag.application.errors import (
    ApplicationInputError,
    ApplicationNotFoundError,
    ApplicationUnavailableError,
)
from dlightrag.application.policies.targeted import Targeted, resolve
from dlightrag.engine.answer.client_contracts import AnswerEffort

#: Whether a reader is offered the thumbs under an answered message.
ANSWER_FEEDBACK = "answer_feedback"

#: The highest agent effort an Answer may run at, whatever its caller chooses; ``max`` is no
#: ceiling at all.
AGENT_EFFORT_CEILING = "agent_effort_ceiling"

#: How many Answers one owner may have queued or running at once; no value is no limit.
ACTIVE_ANSWER_RUNS = "active_answer_runs"

#: A limit is a whole number of Answers, one or more. A JSON ``true`` or ``2.0`` is no count.
type ActiveRunLimit = Annotated[int, Field(strict=True, ge=1)] | None


@dataclass(frozen=True, slots=True)
class Policy[T]:
    """One policy the application knows: its key, the shape of its document, and the value that
    holds until an administrator writes one.

    A policy is added by adding one of these to ``POLICIES``; nothing else here changes.
    """

    key: str
    document: type[Targeted[T]]
    default: T


POLICIES: Mapping[str, Policy[Any]] = MappingProxyType(
    {
        policy.key: policy
        for policy in (
            Policy(ANSWER_FEEDBACK, Targeted[StrictBool], default=True),
            Policy(AGENT_EFFORT_CEILING, Targeted[AnswerEffort], default="max"),
            Policy(ACTIVE_ANSWER_RUNS, Targeted[ActiveRunLimit], default=None),
        )
    }
)


class UnknownPolicyError(ApplicationNotFoundError):
    """No policy has this key. Nothing is kept under a key the application does not know."""

    def __init__(self) -> None:
        super().__init__(f"No runtime policy has that key; the keys are {', '.join(POLICIES)}")


class PolicyDocumentError(ApplicationInputError):
    """A document does not fit the policy it was written for."""


class PolicyUnreadableError(ApplicationUnavailableError):
    """The document kept for a policy no longer fits it, so what it says cannot be known.

    The policy is never read as its code default instead: a policy can switch something off for
    the people it names, and a default would switch it back on for them. It answers again once
    an administrator replaces the document.
    """


@dataclass(frozen=True, slots=True)
class StoredPolicy:
    """A policy document as the store keeps it, with who last wrote it and when."""

    value: Mapping[str, Any]
    updated_at: datetime.datetime
    updated_by: str


class PolicyStore(Protocol):
    """Where policy documents are kept: one per key, replaced whole, the last write standing."""

    async def read_runtime_policy(self, key: str) -> StoredPolicy | None: ...

    async def write_runtime_policy(
        self, key: str, value: Mapping[str, Any], actor: str
    ) -> StoredPolicy: ...


@dataclass(frozen=True, slots=True)
class PolicyState:
    """A policy as it stands. One nobody has written holds its code default and names neither
    who wrote it nor when."""

    key: str
    document: Targeted[Any]
    updated_at: datetime.datetime | None = None
    updated_by: str | None = None


class RuntimePolicies:
    """The Runtime Policies of this deployment, for the administrator who edits them and for
    the requests they decide.

    Every call reads the store once and keeps nothing, so a change holds on the next call, in
    this process and in every other one sharing the database.
    """

    def __init__(self, *, store: PolicyStore) -> None:
        self._store = store

    async def read(self, key: str) -> PolicyState:
        """The policy as it stands: what was last written, or its code default."""
        policy = _known(key)
        return _state(policy, await self._store.read_runtime_policy(key))

    async def write(self, key: str, document: Mapping[str, Any], actor: str) -> PolicyState:
        """Replace the policy's document whole, whoever wrote the one before; the last write
        stands. A document that does not fit the policy is refused and nothing is kept."""
        policy = _known(key)
        try:
            valid = policy.document.model_validate(document)
        except ValidationError as exc:
            raise PolicyDocumentError(_describe(exc)) from None
        stored = await self._store.write_runtime_policy(key, valid.model_dump(mode="json"), actor)
        return _state(policy, stored)

    async def value_for(self, key: str, subject: AccessSubject) -> Any:
        """What the policy says for the caller a request is from."""
        return resolve((await self.read(key)).document, subject)

    async def effort_ceiling(self, subject: AccessSubject) -> AnswerEffort:
        """The highest agent effort an Answer may run at for this caller; ``max`` is no ceiling."""
        return cast(AnswerEffort, await self.value_for(AGENT_EFFORT_CEILING, subject))

    async def active_run_limit(self, subject: AccessSubject) -> int | None:
        """How many Answers this caller's owner may have queued or running; None is no limit."""
        return cast("int | None", await self.value_for(ACTIVE_ANSWER_RUNS, subject))


def _known(key: str) -> Policy[Any]:
    try:
        return POLICIES[key]
    except KeyError:
        raise UnknownPolicyError() from None


def _state[T](policy: Policy[T], stored: StoredPolicy | None) -> PolicyState:
    if stored is None:
        return PolicyState(policy.key, policy.document(default=policy.default))
    try:
        document = policy.document.model_validate(stored.value)
    except ValidationError as exc:
        # Where the kept document is wrong, never what it holds: it names people.
        places = ", ".join(_location(error["loc"]) for error in exc.errors(include_input=False))
        raise PolicyUnreadableError(
            f"The stored '{policy.key}' runtime policy no longer fits its document ({places}); "
            "it is unavailable until an administrator replaces it"
        ) from None
    return PolicyState(policy.key, document, stored.updated_at, stored.updated_by)


def _describe(exc: ValidationError) -> str:
    """Name each invalid field and why. A policy names people, so what was submitted is never
    repeated."""
    return "; ".join(
        f"{_location(error['loc'])}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )


def _location(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc)
