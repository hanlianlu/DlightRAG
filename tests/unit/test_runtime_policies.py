# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Runtime Policies: which callers a targeted setting reaches, and what its document may hold.

Where a policy is kept, and that a change is read by the next request, are the store's and the
routes' to show: see the PostgreSQL and Web suites.
"""

from typing import Any

import pytest

from dlightrag.application.access.principal import UserContext
from dlightrag.application.policies.service import (
    ACTIVE_ANSWER_RUNS,
    AGENT_EFFORT_CEILING,
    ANSWER_FEEDBACK,
    PolicyDocumentError,
    RuntimePolicies,
)
from dlightrag.application.policies.targeted import (
    MAX_CLAIM_LENGTH,
    MAX_RULES,
    MAX_VALUE_LENGTH,
    Targeted,
    resolve,
)
from tests.support.policy_store import InMemoryPolicyStore

_RULE = {"claim": "email", "value": "intern@example.com", "then": False}


def _signed_in(**claims: object) -> UserContext:
    return UserContext(user_id="someone", auth_mode="jwt", claims=claims)


def _document(default: bool, *rules: tuple[str, str, bool]) -> Targeted[bool]:
    return Targeted[bool].model_validate(
        {
            "default": default,
            "rules": [
                {"claim": claim, "value": value, "then": then} for claim, value, then in rules
            ],
        }
    )


class TestWhoAPolicyReaches:
    def test_everyone_gets_the_default_where_no_rule_matches(self) -> None:
        document = _document(True, ("email", "intern@example.com", False))

        assert resolve(document, _signed_in(email="someone@example.com")) is True

    def test_a_matching_rule_gives_that_caller_the_other_value(self) -> None:
        document = _document(True, ("email", "intern@example.com", False))

        assert resolve(document, _signed_in(email="intern@example.com")) is False
        assert resolve(document, _signed_in(email="someone@example.com")) is True

    def test_a_claim_of_several_values_matches_any_one_of_them(self) -> None:
        document = _document(True, ("groups", "contractors", False))

        assert resolve(document, _signed_in(groups=["staff", "contractors"])) is False
        assert resolve(document, _signed_in(groups=["staff"])) is True

    def test_the_first_matching_rule_decides(self) -> None:
        staff_off = ("groups", "staff", False)
        lead_on = ("email", "lead@example.com", True)
        lead = _signed_in(email="lead@example.com", groups=["staff"])

        assert resolve(_document(True, staff_off, lead_on), lead) is False
        assert resolve(_document(True, lead_on, staff_off), lead) is True

    def test_off_for_everyone_but_the_people_a_rule_turns_it_on_for(self) -> None:
        document = _document(False, ("email", "pilot@example.com", True))

        assert resolve(document, _signed_in(email="pilot@example.com")) is True
        assert resolve(document, _signed_in(email="someone@example.com")) is False

    @pytest.mark.parametrize(
        "caller",
        [
            None,
            UserContext(
                user_id="anonymous", auth_mode="none", claims={"email": "intern@example.com"}
            ),
            UserContext(
                user_id="anonymous", auth_mode="simple", claims={"email": "intern@example.com"}
            ),
        ],
        ids=["no caller", "none", "simple"],
    )
    def test_a_caller_whose_claims_are_not_verified_only_gets_the_default(
        self, caller: UserContext | None
    ) -> None:
        document = _document(True, ("email", "intern@example.com", False))

        assert resolve(document, caller) is True


class TestWhatADocumentMayHold:
    async def test_the_bounds_are_the_most_a_document_accepts(self) -> None:
        longest = {"claim": "c" * MAX_CLAIM_LENGTH, "value": "v" * MAX_VALUE_LENGTH, "then": False}

        state = await RuntimePolicies(store=InMemoryPolicyStore()).write(
            ANSWER_FEEDBACK, {"default": True, "rules": [longest] * MAX_RULES}, "admin"
        )

        assert len(state.document.rules) == MAX_RULES

    @pytest.mark.parametrize(
        ("document", "names"),
        [
            ({"default": True, "rules": [_RULE] * (MAX_RULES + 1)}, "rules"),
            ({"default": True, "rules": [{**_RULE, "claim": ""}]}, "rules.0.claim"),
            ({"default": True, "rules": [{**_RULE, "claim": "c" * 65}]}, "rules.0.claim"),
            ({"default": True, "rules": [{**_RULE, "value": "   "}]}, "rules.0.value"),
            ({"default": True, "rules": [{**_RULE, "value": "v" * 257}]}, "rules.0.value"),
            ({"default": True, "rules": [{**_RULE, "then": "no"}]}, "rules.0.then"),
            ({"default": True, "rules": [{**_RULE, "also": 1}]}, "rules.0.also"),
            ({"default": True, "rules": [_RULE, {"claim": "e", "value": "v"}]}, "rules.1.then"),
            ({"default": "yes"}, "default"),
            ({"default": 1}, "default"),
            ({"default": None}, "default"),
            ({"rules": []}, "default"),
            ({"default": True, "rules": "everyone"}, "rules"),
            ({"default": True, "unknown": 1}, "unknown"),
        ],
        ids=[
            "more rules than a policy holds",
            "a rule without a claim",
            "a claim name too long",
            "a value of only spaces",
            "a value too long",
            "a result that is not a bool",
            "a field a rule does not have",
            "a rule without its result",
            "a default that is a word",
            "a default that is a number",
            "a default of nothing",
            "no default",
            "rules that are no list",
            "a field the policy does not have",
        ],
    )
    async def test_a_document_that_does_not_fit_is_refused_and_nothing_is_kept(
        self, document: dict[str, Any], names: str
    ) -> None:
        store = InMemoryPolicyStore()

        with pytest.raises(PolicyDocumentError) as refused:
            await RuntimePolicies(store=store).write(ANSWER_FEEDBACK, document, "admin")

        assert names in str(refused.value)
        assert store.rows == {}

    async def test_a_refusal_says_where_a_document_is_wrong_and_never_repeats_it(self) -> None:
        person = "secret.person@example.com"
        document = {"default": True, "rules": [{**_RULE, "value": person, "then": "no"}]}

        with pytest.raises(PolicyDocumentError) as refused:
            await RuntimePolicies(store=InMemoryPolicyStore()).write(
                ANSWER_FEEDBACK, document, "admin"
            )

        assert "rules.0.then" in str(refused.value)
        assert person not in str(refused.value)


class TestWhatTheAnswerPoliciesHold:
    """The two policies that decide an Answer when it is accepted: how high its agents may
    run, and how many of an owner's Answers may be in progress."""

    async def test_a_policy_nobody_wrote_holds_its_code_default(self) -> None:
        policies = RuntimePolicies(store=InMemoryPolicyStore())
        caller = _signed_in(email="someone@example.com")

        assert await policies.effort_ceiling(caller) == "max", "no ceiling at all"
        assert await policies.active_run_limit(caller) is None, "no limit at all"

    async def test_each_reaches_the_callers_a_rule_names_and_no_other(self) -> None:
        policies = RuntimePolicies(store=InMemoryPolicyStore())
        await policies.write(
            AGENT_EFFORT_CEILING,
            {"default": "max", "rules": [{**_RULE, "then": "low"}]},
            "admin",
        )
        await policies.write(
            ACTIVE_ANSWER_RUNS,
            {
                "default": 3,
                "rules": [{**_RULE, "then": 1}, {**_RULE, "value": "lead", "then": None}],
            },
            "admin",
        )
        intern = _signed_in(email="intern@example.com")
        lead = _signed_in(email="lead")
        unverified = UserContext(
            user_id="anonymous", auth_mode="none", claims={"email": "intern@example.com"}
        )

        assert [await policies.effort_ceiling(who) for who in (intern, lead, unverified, None)] == [
            "low",
            "max",
            "max",
            "max",
        ]
        assert [
            await policies.active_run_limit(who) for who in (intern, lead, unverified, None)
        ] == [
            1,
            None,
            3,
            3,
        ]

    async def test_no_limit_is_a_null_and_is_kept_as_one(self) -> None:
        policies = RuntimePolicies(store=InMemoryPolicyStore())

        state = await policies.write(ACTIVE_ANSWER_RUNS, {"default": None}, "admin")

        assert state.document.model_dump(mode="json") == {"default": None, "rules": []}

    @pytest.mark.parametrize(
        ("key", "document", "names"),
        [
            (AGENT_EFFORT_CEILING, {"default": "xhigh"}, "default"),
            (AGENT_EFFORT_CEILING, {"default": "LOW"}, "default"),
            (AGENT_EFFORT_CEILING, {"default": None}, "default"),
            (AGENT_EFFORT_CEILING, {"default": "max", "rules": [{**_RULE, "then": "off"}]}, "then"),
            (ACTIVE_ANSWER_RUNS, {"default": 0}, "default"),
            (ACTIVE_ANSWER_RUNS, {"default": -1}, "default"),
            (ACTIVE_ANSWER_RUNS, {"default": True}, "default"),
            (ACTIVE_ANSWER_RUNS, {"default": 2.0}, "default"),
            (ACTIVE_ANSWER_RUNS, {"default": "3"}, "default"),
            (ACTIVE_ANSWER_RUNS, {"rules": []}, "default"),
            (ACTIVE_ANSWER_RUNS, {"default": 3, "rules": [{**_RULE, "then": 0}]}, "then"),
            (ACTIVE_ANSWER_RUNS, {"default": 3, "rules": [{**_RULE, "then": False}]}, "then"),
        ],
        ids=[
            "a ceiling the ladder of three does not have",
            "a ceiling spelled in capitals",
            "a ceiling of nothing",
            "a rule lowering to off",
            "a limit of none",
            "a negative limit",
            "a limit that is true",
            "a limit that is a float",
            "a limit that is a word",
            "no default limit",
            "a rule limiting to none",
            "a rule limiting to false",
        ],
    )
    async def test_a_document_that_is_no_level_or_count_is_refused_and_nothing_is_kept(
        self, key: str, document: dict[str, Any], names: str
    ) -> None:
        store = InMemoryPolicyStore()

        with pytest.raises(PolicyDocumentError) as refused:
            await RuntimePolicies(store=store).write(key, document, "admin")

        assert names in str(refused.value)
        assert store.rows == {}
