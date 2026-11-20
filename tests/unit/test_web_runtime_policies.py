# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Runtime Policy routes, and answer feedback as the first thing a policy decides.

The access control is the real one over signed callers, and the policy service is the real one
over a store that keeps its documents in memory, so what these tests observe is who may edit a
policy and what the very next rating request of each caller meets.
"""

import datetime
from typing import Any

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.server import create_app
from dlightrag.application.access.control import AccessAction
from dlightrag.application.access.workspaces import WorkspaceRecord
from dlightrag.application.answer_runs.service import AgentEffortOffer
from dlightrag.application.config.sections import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
)
from dlightrag.application.corpus_admin.workspace_catalog import (
    WorkspaceCatalogCursorCodec,
    WorkspaceCatalogPage,
)
from dlightrag.application.policies.service import (
    AGENT_EFFORT_CEILING,
    ANSWER_FEEDBACK,
    RuntimePolicies,
    StoredPolicy,
)
from dlightrag.engine.answer.feedback import AnswerFeedback
from tests.support.application_double import application_double, delegate
from tests.support.config_helpers import mutate_config
from tests.support.policy_store import InMemoryPolicyStore
from tests.unit.conftest import answer_capabilities
from tests.unit.web.answer_run_fixtures import NOW, RUN_ID, linked_turn

_SECRET = "test-only-web-jwt-key-32-bytes!!"
_TEAM = "https://team.cloudflareaccess.com"
_ADMIN = "admin@example.com"
_OPERATOR = "ops@example.com"
_ALICE = "alice@example.com"
_BOB = "bob@example.com"

_POLICY_URL = f"/web/api/runtime-policies/{ANSWER_FEEDBACK}"
_BOOTSTRAP_URL = "/web/api/bootstrap"
_FEEDBACK_URL = f"/web/api/answer/{RUN_ID}/feedback"
_JUDGEMENT = {"agent": "main", "sequence": 7, "value": "up"}
_RETRACTION = {"agent": "main", "sequence": 7}
_OFF_FOR_ALICE = {
    "default": True,
    "rules": [{"claim": "email", "value": _ALICE, "then": False}],
}
_ON_FOR_BOB_ONLY = {
    "default": False,
    "rules": [{"claim": "email", "value": _BOB, "then": True}],
}


def _as(email: str) -> dict[str, str]:
    """What a scripted client signed in as ``email`` sends."""
    claims = {
        "iss": _TEAM,
        "sub": email,
        "email": email,
        "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(claims, _SECRET, algorithm='HS256')}"}


@pytest.fixture
def store() -> InMemoryPolicyStore:
    return InMemoryPolicyStore()


@pytest.fixture
def application(test_config: DlightragConfig, store: InMemoryPolicyStore):
    """Everyone from the team is a common user; one administrator, and one operator who holds
    the one action that manages policies."""
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", _SECRET)
    mutate_config(
        test_config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="email", value=_ADMIN, workspaces=["*"], actions=["admin"]
                ),
                AccessControlRuleConfig(
                    claim="email",
                    value=_OPERATOR,
                    workspaces=["*"],
                    actions=[AccessAction.RUNTIME_POLICY_MANAGE],
                ),
                AccessControlRuleConfig(
                    claim="iss", value=_TEAM, workspaces=["*"], actions=["common_user"]
                ),
            ],
        ),
    )
    created = application_double(test_config, policies=RuntimePolicies(store=store))
    created.web_conversations.turn_for_run.return_value = linked_turn()
    created.answers.record_feedback.return_value = AnswerFeedback(
        agent="main", sequence=7, value="up", reasons=(), note=None, updated_at=NOW
    )
    created.answers.answer_feedback.return_value = ()
    # What the bootstrap reads besides the policy: one workspace everyone from the team may query.
    created.answers.capabilities.return_value = answer_capabilities()
    created.answers.agent_effort_offer.return_value = AgentEffortOffer(("low", "high"), None)
    workspace: WorkspaceRecord = {
        "workspace": "default",
        "display_name": "Default",
        "embedding_model": "e",
        "created_at": None,
        "updated_at": None,
    }
    created.corpora.alist_workspace_records.return_value = [workspace]
    created.corpora.list_workspace_records_page.return_value = WorkspaceCatalogPage(
        items=(workspace,), next_cursor=None, fetched_rows=1
    )
    created.corpora.workspace_catalog_cursor_codec = WorkspaceCatalogCursorCodec(b"bootstrap-test")
    return created


@pytest.fixture
async def client(application):
    app = create_app()
    app.state.application = application
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as created:
        yield created


async def _send(client: AsyncClient, method: str, url: str, caller: str, **request: Any) -> Any:
    """One request as ``caller``, a scripted client that keeps no cookies between requests."""
    client.cookies.clear()
    return await client.request(method, url, headers=_as(caller), **request)


@pytest.mark.parametrize(
    ("caller", "may"),
    [(_ADMIN, True), (_OPERATOR, True), (_ALICE, False)],
    ids=["an administrator", "an operator holding the action", "an editor"],
)
async def test_only_a_manager_reads_or_replaces_a_policy(
    client: AsyncClient, store: InMemoryPolicyStore, caller: str, may: bool
) -> None:
    read = await _send(client, "GET", _POLICY_URL, caller)
    replaced = await _send(client, "PUT", _POLICY_URL, caller, json=_OFF_FOR_ALICE)

    assert (read.status_code, replaced.status_code) == ((200, 200) if may else (403, 403))
    assert (ANSWER_FEEDBACK in store.rows) is may, "a refused caller changes nothing"


async def test_a_caller_nobody_signed_in_manages_nothing(
    client: AsyncClient, store: InMemoryPolicyStore
) -> None:
    read = await client.get(_POLICY_URL)
    replaced = await client.put(_POLICY_URL, json=_OFF_FOR_ALICE)

    assert (read.status_code, replaced.status_code) == (401, 401)
    assert store.rows == {}


async def test_a_policy_nobody_wrote_reads_as_its_code_default(client: AsyncClient) -> None:
    response = await _send(client, "GET", _POLICY_URL, _ADMIN)

    assert response.status_code == 200
    assert response.json() == {
        "key": ANSWER_FEEDBACK,
        "policy": {"default": True, "rules": []},
        "updated_at": None,
        "updated_by": None,
    }


async def test_what_a_replacement_answers_is_what_a_read_then_finds(
    client: AsyncClient, store: InMemoryPolicyStore
) -> None:
    sloppy = {
        "default": True,
        "rules": [{"claim": " email ", "value": f" {_ALICE} ", "then": False}],
    }

    replaced = await _send(client, "PUT", _POLICY_URL, _ADMIN, json=sloppy)
    read = await _send(client, "GET", _POLICY_URL, _ADMIN)

    assert replaced.status_code == 200
    assert replaced.json() == {
        "key": ANSWER_FEEDBACK,
        "policy": _OFF_FOR_ALICE,
        "updated_at": "2026-10-09T08:00:00Z",
        "updated_by": _ADMIN,
    }
    assert read.json() == replaced.json()
    assert store.rows[ANSWER_FEEDBACK].value == _OFF_FOR_ALICE


async def test_a_policy_the_application_does_not_know_is_not_found_to_a_manager_only(
    client: AsyncClient, store: InMemoryPolicyStore
) -> None:
    url = "/web/api/runtime-policies/no_such_policy"

    manager = [
        await _send(client, "GET", url, _ADMIN),
        await _send(client, "PUT", url, _ADMIN, json={"default": True}),
    ]
    editor = [
        await _send(client, "GET", url, _ALICE),
        await _send(client, "PUT", url, _ALICE, json={"default": True}),
    ]

    assert [response.status_code for response in manager] == [404, 404]
    assert ANSWER_FEEDBACK in manager[0].json()["detail"]
    assert [response.status_code for response in editor] == [403, 403]
    assert store.rows == {}


@pytest.mark.parametrize(
    "body",
    [{"default": "yes", "rules": []}, ["default"]],
    ids=["a document the policy cannot hold", "a body that is no document"],
)
async def test_a_document_the_policy_cannot_hold_is_refused_and_replaces_nothing(
    client: AsyncClient, store: InMemoryPolicyStore, body: Any
) -> None:
    await _send(client, "PUT", _POLICY_URL, _ADMIN, json=_OFF_FOR_ALICE)

    response = await _send(client, "PUT", _POLICY_URL, _ADMIN, json=body)

    assert response.status_code == 422
    assert store.rows[ANSWER_FEEDBACK].value == _OFF_FOR_ALICE, "what was kept stays"


async def test_a_change_to_the_policy_holds_on_the_very_next_rating_request(
    client: AsyncClient, application: Any
) -> None:
    async def rate(caller: str) -> int:
        return (await _send(client, "PUT", _FEEDBACK_URL, caller, json=_JUDGEMENT)).status_code

    async def retract(caller: str) -> int:
        return (
            await _send(client, "DELETE", _FEEDBACK_URL, caller, params=_RETRACTION)
        ).status_code

    # Nobody edited the policy: everyone is offered feedback.
    assert (await rate(_ALICE), await rate(_BOB)) == (200, 200)

    # Off for one person: she is refused at once when she rates and he is not, but taking back a
    # judgement she gave only removes her own data, so it still works for her.
    assert (await _send(client, "PUT", _POLICY_URL, _ADMIN, json=_OFF_FOR_ALICE)).status_code == 200
    refused = await _send(client, "PUT", _FEEDBACK_URL, _ALICE, json=_JUDGEMENT)
    assert (refused.status_code, refused.json()["detail"]) == (
        403,
        "Answer feedback is turned off for you",
    )
    assert (await retract(_ALICE), await retract(_BOB)) == (204, 204)
    assert await rate(_BOB) == 200
    assert application.answers.record_feedback.await_count == 3, "her rating was never attempted"
    assert application.answers.retract_feedback.await_count == 2, "her judgement is taken back"

    # The other way round: off for everyone but the people a rule turns it on for.
    assert (
        await _send(client, "PUT", _POLICY_URL, _ADMIN, json=_ON_FOR_BOB_ONLY)
    ).status_code == 200
    assert (await rate(_ALICE), await rate(_BOB)) == (403, 200)

    # And back on for everyone, by replacing the document whole.
    assert (
        await _send(client, "PUT", _POLICY_URL, _ADMIN, json={"default": True})
    ).status_code == 200
    assert (await rate(_ALICE), await rate(_BOB)) == (200, 200)


async def test_a_reader_the_policy_turns_feedback_off_for_is_told_so_and_given_no_judgements(
    client: AsyncClient, application: Any
) -> None:
    await _send(client, "PUT", _POLICY_URL, _ADMIN, json=_OFF_FOR_ALICE)

    alice = await _send(client, "GET", _FEEDBACK_URL, _ALICE)
    bob = await _send(client, "GET", _FEEDBACK_URL, _BOB)

    assert alice.json() == {"run_id": RUN_ID, "enabled": False, "feedback": []}
    assert bob.json() == {"run_id": RUN_ID, "enabled": True, "feedback": []}
    application.answers.answer_feedback.assert_awaited_once()


async def test_the_bootstrap_tells_each_reader_whether_the_policy_offers_them_feedback(
    client: AsyncClient,
) -> None:
    async def offered(*callers: str) -> tuple[bool, ...]:
        answers = [await _send(client, "GET", _BOOTSTRAP_URL, caller) for caller in callers]
        assert [answer.status_code for answer in answers] == [200] * len(callers)
        return tuple(answer.json()["answer_feedback_enabled"] for answer in answers)

    # Nobody edited the policy: its code default offers it to everyone.
    assert await offered(_ALICE, _BOB, _ADMIN) == (True, True, True)

    # Off for one person, and the very next bootstrap says so.
    await _send(client, "PUT", _POLICY_URL, _ADMIN, json=_OFF_FOR_ALICE)
    assert await offered(_ALICE, _BOB, _ADMIN) == (False, True, True)

    # The other way round: off for everyone but the people a rule turns it on for.
    await _send(client, "PUT", _POLICY_URL, _ADMIN, json=_ON_FOR_BOB_ONLY)
    assert await offered(_ALICE, _BOB, _ADMIN) == (False, True, False)


async def test_a_kept_policy_that_no_longer_validates_is_unavailable_until_an_admin_replaces_it(
    client: AsyncClient, application: Any, store: InMemoryPolicyStore
) -> None:
    unreadable = {"default": True, "rules": [{"claim": "email", "value": _ALICE, "then": "maybe"}]}
    store.rows[ANSWER_FEEDBACK] = StoredPolicy(unreadable, NOW, _ADMIN)

    rated = await _send(client, "PUT", _FEEDBACK_URL, _BOB, json=_JUDGEMENT)
    read = await _send(client, "GET", _POLICY_URL, _ADMIN)
    booted = await _send(client, "GET", _BOOTSTRAP_URL, _BOB)

    # The page still loads, with the thumbs off: a document nobody can read must not take the whole
    # product down, and must not offer what it may have switched off.
    assert booted.status_code == 200
    assert booted.json()["answer_feedback_enabled"] is False

    for response in (rated, read):
        assert response.status_code == 503, "never the code default for a document unread"
        assert response.json()["error_type"] == "unavailable"
        detail = response.json()["detail"]
        assert ANSWER_FEEDBACK in detail
        assert "rules.0.then" in detail
        assert _ALICE not in detail
        assert "maybe" not in detail
    application.answers.record_feedback.assert_not_awaited()

    replaced = await _send(client, "PUT", _POLICY_URL, _ADMIN, json=_OFF_FOR_ALICE)
    rated_again = await _send(client, "PUT", _FEEDBACK_URL, _BOB, json=_JUDGEMENT)

    assert (replaced.status_code, rated_again.status_code) == (200, 200)


async def test_the_bootstrap_offers_each_reader_the_efforts_the_ceiling_leaves_them(
    client: AsyncClient, application: Any
) -> None:
    from tests.unit.test_answer_service import _service

    # The real offer over the real policies, so what the page is told is what a Run is held to.
    delegate(
        application.answers,
        _service(policies=application.policies),
        "agent_effort_offer",
    )
    url = f"/web/api/runtime-policies/{AGENT_EFFORT_CEILING}"
    lowered_for_alice = {
        "default": "max",
        "rules": [{"claim": "email", "value": _ALICE, "then": "low"}],
    }

    async def offered(caller: str) -> dict[str, Any]:
        response = await _send(client, "GET", _BOOTSTRAP_URL, caller)
        assert response.status_code == 200
        return response.json()["agent_effort"]

    # Nobody wrote a ceiling: everyone is offered every level the model can express.
    everyone = {"levels": ["low", "high", "max"], "default": None}
    assert (await offered(_ALICE), await offered(_BOB)) == (everyone, everyone)

    # Lowered for one person, and the very next bootstrap says so: she is offered the one level
    # her Answers run at, and the control marks it, since it is what she gets by choosing nothing.
    assert (await _send(client, "PUT", url, _ADMIN, json=lowered_for_alice)).status_code == 200
    assert await offered(_ALICE) == {"levels": ["low"], "default": "low"}
    assert await offered(_BOB) == everyone


async def test_the_bootstrap_still_loads_while_the_ceiling_cannot_be_read(
    client: AsyncClient, store: InMemoryPolicyStore, application: Any
) -> None:
    from tests.unit.test_answer_service import _service

    delegate(
        application.answers,
        _service(policies=application.policies),
        "agent_effort_offer",
    )
    store.rows[AGENT_EFFORT_CEILING] = StoredPolicy({"default": "xhigh"}, NOW, _ADMIN)

    booted = await _send(client, "GET", _BOOTSTRAP_URL, _BOB)

    # Nothing is offered, as nothing may be: a submission is refused until the document is
    # replaced, and the page, which does not depend on it, loads.
    assert booted.status_code == 200
    assert booted.json()["agent_effort"] == {"levels": [], "default": None}
