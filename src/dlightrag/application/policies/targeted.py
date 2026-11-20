# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A targeted setting: a default, and ordered rules that give some callers another value.

A rule names a claim and a value, the way an Access Rule does, and the first rule that a
caller's verified claims match decides; a caller no rule matches gets the default. Both
directions work: on for everyone but some, or off for everyone but some. Only a verified JWT
carries claims, so a caller of a deployment that verifies none cannot be told apart from
another, and always gets the default.
"""

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.application.access.control import (
    AccessSubject,
    claim_matches,
)

#: The most rules one policy holds, and the longest claim name and claim value a rule names.
MAX_RULES = 100
MAX_CLAIM_LENGTH = 64
MAX_VALUE_LENGTH = 256


class Rule[T](BaseModel):
    """Callers whose ``claim`` is, or holds, ``value`` get ``then``."""

    # A rule names people, so a refusal says where a document is wrong and never repeats it.
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, hide_input_in_errors=True
    )

    claim: str = Field(min_length=1, max_length=MAX_CLAIM_LENGTH)
    value: str = Field(min_length=1, max_length=MAX_VALUE_LENGTH)
    then: T


class Targeted[T](BaseModel):
    """What everyone gets, and the rules that give some callers something else."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    default: T
    rules: list[Rule[T]] = Field(default_factory=list, max_length=MAX_RULES)


def resolve[T](policy: Targeted[T], subject: AccessSubject) -> T:
    """The value for one caller: the first rule their verified claims match, else the default."""
    if subject is not None and subject.auth_mode == "jwt":
        for rule in policy.rules:
            if claim_matches(subject.claims, rule.claim, rule.value):
                return rule.then
    return policy.default
