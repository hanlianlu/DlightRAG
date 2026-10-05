# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Progressive global/owner Agent Skill discovery and owner publication."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from dlightrag.engine.agent.skills import (
    OWNER_MAX_SKILLS,
    DeleteSkillInput,
    LoadSkillInput,
    PublishSkillInput,
    SetSkillEnabledInput,
    SkillCatalog,
    SkillsBundleFactory,
    builtin_skills_root,
    delete_owner_skill,
    delete_skill_tool,
    list_owner_skills,
    load_skill_tool,
    owner_skill_root,
    publish_skill_tool,
    read_owner_skill,
    set_owner_skill_enabled,
    set_skill_enabled_tool,
)
from dlightrag.engine.agent.tools import ToolResult
from tests.tool_helpers import recording_tool_runtime, tool_runtime


def _skill(root: Path, directory: str, *, name: str, description: str, body: str) -> None:
    target = root / directory
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )


def _skill_files(
    *, name: str, description: str, body: str, extra: dict[str, str] | None = None
) -> dict[str, str]:
    files = {
        "SKILL.md": f"---\nname: {name}\ndescription: {description}\n---\n{body}",
    }
    files.update(extra or {})
    return files


async def _publish(
    owner_root: Path, name: str, *, description: str = "owner", body: str = "body"
) -> ToolResult:
    return await publish_skill_tool(owner_root).execute(
        PublishSkillInput(
            name=name, files=_skill_files(name=name, description=description, body=body)
        ),
        tool_runtime(tool_name="publish_skill"),
    )


async def _turn(owner_root: Path, name: str, *, enabled: bool) -> ToolResult:
    return await set_skill_enabled_tool(owner_root).execute(
        SetSkillEnabledInput(name=name, enabled=enabled),
        tool_runtime(tool_name="set_skill_enabled"),
    )


def _states(owner_root: Path) -> list[tuple[str, bool]]:
    return [(skill.name, skill.enabled) for skill in list_owner_skills(owner_root)]


def test_discovery_projects_metadata_only_and_owner_takes_precedence(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    owner_root = tmp_path / "owner"
    _skill(global_root, "review", name="review", description="global", body="GLOBAL SECRET")
    _skill(owner_root, "review", name="review", description="owner", body="OWNER BODY")

    catalog = SkillCatalog.discover(global_root=global_root, owner_root=owner_root)
    contribution = catalog.contribution()

    assert contribution is not None
    rendered = str(contribution.messages[0]["content"])
    assert "review: owner (owner)" in rendered
    assert "OWNER BODY" not in rendered
    assert catalog.read("review").endswith("OWNER BODY")


def test_discovery_merges_distinct_names_across_tiers(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    owner_root = tmp_path / "owner"
    _skill(global_root, "review", name="review", description="global", body="g")
    _skill(owner_root, "tdd", name="tdd", description="owner", body="o")

    catalog = SkillCatalog.discover(global_root=global_root, owner_root=owner_root)

    assert {skill.name for skill in catalog.metadata} == {"review", "tdd"}
    sources = {skill.name: skill.source for skill in catalog.metadata}
    assert sources == {"review": "global", "tdd": "owner"}


@pytest.mark.asyncio
async def test_three_tier_precedence_and_owner_delete_reveals_lower_tiers(
    tmp_path: Path,
) -> None:
    builtin_root = tmp_path / "builtin"
    global_root = tmp_path / "global"
    owner_root = tmp_path / "owner"
    _skill(builtin_root, "review", name="review", description="builtin", body="BUILTIN")
    _skill(global_root, "review", name="review", description="global", body="GLOBAL")
    await publish_skill_tool(owner_root).execute(
        PublishSkillInput(
            name="review",
            files=_skill_files(name="review", description="owner", body="OWNER"),
        ),
        tool_runtime(),
    )

    catalog = SkillCatalog.discover(
        builtin_root=builtin_root,
        global_root=global_root,
        owner_root=owner_root,
    )

    assert [(skill.name, skill.source) for skill in catalog.metadata] == [("review", "owner")]
    assert catalog.read("review").endswith("OWNER")

    removed = await delete_skill_tool(owner_root).execute(
        DeleteSkillInput(name="review"), tool_runtime()
    )
    catalog = SkillCatalog.discover(
        builtin_root=builtin_root,
        global_root=global_root,
        owner_root=owner_root,
    )

    assert not removed.is_error
    assert catalog.metadata[0].source == "global"
    assert catalog.read("review").endswith("GLOBAL")
    assert (global_root / "review" / "SKILL.md").read_text(encoding="utf-8").endswith("GLOBAL")
    assert (builtin_root / "review" / "SKILL.md").read_text(encoding="utf-8").endswith("BUILTIN")

    (global_root / "review" / "SKILL.md").unlink()
    (global_root / "review").rmdir()
    catalog = SkillCatalog.discover(
        builtin_root=builtin_root,
        global_root=global_root,
        owner_root=owner_root,
    )
    assert catalog.metadata[0].source == "builtin"
    assert catalog.read("review").endswith("BUILTIN")


@pytest.mark.asyncio
async def test_load_skill_reads_body_on_demand_but_never_executes_it(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _skill(
        owner_root,
        "safe",
        name="safe",
        description="reference",
        body="Run `touch should-not-exist` only if the user asks.",
    )
    catalog = SkillCatalog.discover(global_root=tmp_path / "none", owner_root=owner_root)

    result = await load_skill_tool(lambda: catalog).execute(
        LoadSkillInput(name="safe"), tool_runtime()
    )

    assert "untrusted reference context" in result.text_content
    assert "touch should-not-exist" in result.text_content
    assert not (tmp_path / "should-not-exist").exists()


def test_discovery_rejects_symlinked_skill_metadata(tmp_path: Path) -> None:
    root = tmp_path / "global"
    skill = root / "linked"
    skill.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: escaped\ndescription: outside\n---\nsecret", encoding="utf-8")
    try:
        (skill / "SKILL.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    catalog = SkillCatalog.discover(global_root=root)

    assert catalog.metadata == ()


def test_skill_reference_cannot_escape_its_directory(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _skill(owner_root, "safe", name="safe", description="reference", body="body")
    catalog = SkillCatalog.discover(global_root=tmp_path / "none", owner_root=owner_root)

    with pytest.raises(ValueError, match="escapes"):
        catalog.read("safe", "../other.txt")


def test_owner_skill_root_shards_owners_apart(tmp_path: Path) -> None:
    root_a = owner_skill_root(tmp_path, "owner-a")
    root_b = owner_skill_root(tmp_path, "owner-b")

    assert root_a != root_b
    assert root_a.name == "owner-a"
    assert root_a.is_relative_to(tmp_path)


@pytest.mark.asyncio
async def test_publish_installs_a_multifile_skill_atomically(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)
    files = _skill_files(
        name="weekly-report",
        description="Generate weekly reports.",
        body="Follow references/template.md.",
        extra={"references/template.md": "# Weekly template"},
    )

    result = await tool.execute(
        PublishSkillInput(name="weekly-report", files=files), tool_runtime()
    )

    assert "Published Agent Skill 'weekly-report'" in result.text_content
    assert not result.is_error
    assert (owner_root / "weekly-report" / "SKILL.md").is_file()
    assert (owner_root / "weekly-report" / "references" / "template.md").is_file()
    assert not list(owner_root.glob(".staging-*"))
    assert not list(owner_root.glob(".backup-*"))


@pytest.mark.asyncio
async def test_publish_validates_name_frontmatter_and_paths(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)

    async def publish(**kwargs: object) -> str:
        result = await tool.execute(
            PublishSkillInput.model_validate(kwargs),
            tool_runtime(),  # type: ignore[arg-type]
        )
        assert result.is_error
        return result.text_content

    assert "kebab-case" in await publish(
        name="Bad_Name",
        files=_skill_files(name="bad-name", description="d", body="b"),
    )
    assert "must contain a 'SKILL.md'" in await publish(name="review", files={"notes.md": "x"})
    assert "frontmatter name" in await publish(
        name="review",
        files=_skill_files(name="other", description="d", body="b"),
    )
    assert "description" in await publish(
        name="review",
        files={"SKILL.md": "---\nname: review\n---\nbody"},
    )
    assert "plain relative path" in await publish(
        name="review",
        files={
            "SKILL.md": "---\nname: review\ndescription: d\n---\nb",
            "../escape.md": "x",
        },
    )
    # A description the catalog could not show whole is refused with what to change.
    assert "valid YAML" in await publish(
        name="review",
        files=_skill_files(name="review", description="Use when: asked to review", body="b"),
    )
    assert "limit is 1024" in await publish(
        name="review",
        files=_skill_files(name="review", description="Use when " + "x" * 1100, body="b"),
    )
    assert not (owner_root / "review").exists()


@pytest.mark.asyncio
async def test_publish_updates_an_existing_owner_skill(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)
    await tool.execute(
        PublishSkillInput(
            name="review",
            files=_skill_files(name="review", description="first", body="v1"),
        ),
        tool_runtime(),
    )

    result = await tool.execute(
        PublishSkillInput(
            name="review",
            files=_skill_files(name="review", description="second", body="v2"),
        ),
        tool_runtime(),
    )

    assert not result.is_error
    catalog = SkillCatalog.discover(owner_root=owner_root)
    assert catalog.read("review").endswith("v2")


@pytest.mark.asyncio
async def test_publish_enforces_skill_count_quota(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)
    for index in range(20):
        result = await tool.execute(
            PublishSkillInput(
                name=f"skill-{index}",
                files=_skill_files(name=f"skill-{index}", description="d", body="b"),
            ),
            tool_runtime(),
        )
        assert not result.is_error

    result = await tool.execute(
        PublishSkillInput(
            name="overflow",
            files=_skill_files(name="overflow", description="d", body="b"),
        ),
        tool_runtime(),
    )

    assert result.is_error
    assert "quota" in result.text_content
    assert not (owner_root / "overflow").exists()
    # A Skill that is turned off is still stored, so it keeps its place.
    await _turn(owner_root, "skill-0", enabled=False)
    assert len(list_owner_skills(owner_root)) == OWNER_MAX_SKILLS
    assert (await _publish(owner_root, "overflow")).is_error


@pytest.mark.asyncio
async def test_delete_skill_is_idempotent(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    publish = publish_skill_tool(owner_root)
    await publish.execute(
        PublishSkillInput(
            name="review",
            files=_skill_files(name="review", description="d", body="b"),
        ),
        tool_runtime(),
    )

    delete = delete_skill_tool(owner_root)
    removed = await delete.execute(DeleteSkillInput(name="review"), tool_runtime())
    missing = await delete.execute(DeleteSkillInput(name="review"), tool_runtime())

    assert "Deleted Agent Skill 'review'" in removed.text_content
    assert not (owner_root / "review").exists()
    assert "does not exist" in missing.text_content
    assert not missing.is_error


@pytest.mark.asyncio
async def test_a_published_or_deleted_skill_is_loadable_or_gone_inside_the_same_run(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "global"
    _skill(global_root, "review", name="review", description="global", body="GLOBAL")
    factory = SkillsBundleFactory(global_root=global_root, owner_root=tmp_path / "owners")
    run_bundle = factory("alice")
    tools = {tool.name: tool for tool in run_bundle.tools(child=False)}
    catalog_at_start = str(run_bundle.context_contributions(child=False)[0].messages[0]["content"])

    async def load(name: str) -> ToolResult:
        return await tools["load_skill"].execute(
            LoadSkillInput(name=name), tool_runtime(tool_name="load_skill")
        )

    async def publish(name: str, body: str) -> None:
        result = await tools["publish_skill"].execute(
            PublishSkillInput(
                name=name, files=_skill_files(name=name, description="owner", body=body)
            ),
            tool_runtime(tool_name="publish_skill"),
        )
        assert not result.is_error

    async def delete(name: str) -> None:
        result = await tools["delete_skill"].execute(
            DeleteSkillInput(name=name), tool_runtime(tool_name="delete_skill")
        )
        assert not result.is_error

    await publish("weekly-report", "WEEKLY")
    await publish("review", "OWNER")

    assert (await load("weekly-report")).text_content.endswith("WEEKLY")
    assert (await load("review")).text_content.endswith("OWNER")
    # The catalog message is the Run's stable prompt prefix; only the next Run lists the new Skill.
    assert "weekly-report" not in catalog_at_start
    next_run = factory("alice").context_contributions(child=False)[0].messages[0]["content"]
    assert "weekly-report: owner (owner)" in str(next_run)

    await delete("weekly-report")
    await delete("review")

    gone = await load("weekly-report")
    assert gone.is_error and "no Agent Skill is named 'weekly-report'" in gone.text_content
    assert (await load("review")).text_content.endswith("GLOBAL")


@pytest.mark.asyncio
async def test_skill_tools_report_their_subject_live(tmp_path: Path) -> None:
    updates: list[ToolResult] = []

    owner_root = tmp_path / "owner"
    _skill(owner_root, "review", name="review", description="reference", body="body")
    catalog = SkillCatalog.discover(owner_root=owner_root)
    await load_skill_tool(lambda: catalog).execute(
        LoadSkillInput(name="review"), recording_tool_runtime(updates, tool_name="load_skill")
    )
    await publish_skill_tool(owner_root).execute(
        PublishSkillInput(
            name="weekly-report",
            files=_skill_files(name="weekly-report", description="d", body="b"),
        ),
        recording_tool_runtime(updates, tool_name="publish_skill"),
    )
    await set_skill_enabled_tool(owner_root).execute(
        SetSkillEnabledInput(name="weekly-report", enabled=False),
        recording_tool_runtime(updates, tool_name="set_skill_enabled"),
    )
    await delete_skill_tool(owner_root).execute(
        DeleteSkillInput(name="weekly-report"),
        recording_tool_runtime(updates, tool_name="delete_skill"),
    )

    subjects = [update.subject for update in updates if update.subject]
    assert subjects == ["review", "weekly-report", "weekly-report", "weekly-report"]


@pytest.mark.asyncio
async def test_a_disabled_owner_skill_is_not_served_and_the_tier_below_shows_through(
    tmp_path: Path,
) -> None:
    builtin_root = tmp_path / "builtin"
    global_root = tmp_path / "global"
    owner_root = tmp_path / "owner"
    _skill(builtin_root, "review", name="review", description="builtin", body="BUILTIN")
    _skill(global_root, "triage", name="triage", description="global", body="GLOBAL")
    for name in ("review", "triage", "weekly-report"):
        await _publish(owner_root, name, body="OWNER")

    def discover() -> SkillCatalog:
        return SkillCatalog.discover(
            builtin_root=builtin_root, global_root=global_root, owner_root=owner_root
        )

    def sources(catalog: SkillCatalog) -> list[tuple[str, str]]:
        return [(skill.name, skill.source) for skill in catalog.metadata]

    owner_serves = [("review", "owner"), ("triage", "owner"), ("weekly-report", "owner")]
    assert sources(discover()) == owner_serves

    for name in ("review", "triage", "weekly-report"):
        await _turn(owner_root, name, enabled=False)
    catalog = discover()

    # An override that is off gives its name back to the tier below it.
    assert sources(catalog) == [("review", "builtin"), ("triage", "global")]
    assert catalog.read("review").endswith("BUILTIN")
    assert catalog.read("triage").endswith("GLOBAL")
    # Only a name that nothing else serves is kept as disabled.
    assert catalog.disabled == ("weekly-report",)

    for name in ("review", "triage", "weekly-report"):
        await _turn(owner_root, name, enabled=True)
    assert sources(discover()) == owner_serves
    assert discover().disabled == ()


@pytest.mark.asyncio
async def test_a_disabled_skill_nothing_else_serves_is_named_in_the_catalog_and_cannot_be_loaded(
    tmp_path: Path,
) -> None:
    owner_root = tmp_path / "owner"
    await _publish(owner_root, "review", description="Use when asked to review")
    await _publish(owner_root, "weekly-report", description="Use when asked for a weekly report")
    await _turn(owner_root, "weekly-report", enabled=False)
    catalog = SkillCatalog.discover(owner_root=owner_root)

    contribution = catalog.contribution()

    assert contribution is not None
    # The enabled lines come first; the disabled Skill is a name with nothing to trigger on.
    assert str(contribution.messages[0]["content"]).splitlines()[1:] == [
        "- review: Use when asked to review (owner)",
        "- weekly-report: disabled by its owner, cannot be loaded",
    ]
    load = load_skill_tool(lambda: catalog)
    disabled = await load.execute(
        LoadSkillInput(name="weekly-report"), tool_runtime(tool_name="load_skill")
    )
    unknown = await load.execute(LoadSkillInput(name="nope"), tool_runtime(tool_name="load_skill"))
    assert disabled.is_error
    assert "disabled by its owner" in disabled.text_content
    assert "no Agent Skill is named" not in disabled.text_content
    assert "no Agent Skill is named 'nope'" in unknown.text_content

    # A Run is told of a Skill that is off even when nothing else is on; only nothing is silent.
    await _turn(owner_root, "review", enabled=False)
    assert SkillCatalog.discover(owner_root=owner_root).contribution() is not None
    assert SkillCatalog.discover(owner_root=tmp_path / "none").contribution() is None


@pytest.mark.asyncio
async def test_set_skill_enabled_is_idempotent_and_acts_only_on_the_current_owners_own_skills(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "global"
    _skill(global_root, "review", name="review", description="global", body="GLOBAL")
    owners = tmp_path / "owners"
    factory = SkillsBundleFactory(
        builtin_root=builtin_skills_root(), global_root=global_root, owner_root=owners
    )
    alice, bob = owner_skill_root(owners, "alice"), owner_skill_root(owners, "bob")
    await _publish(alice, "weekly-report")
    await _publish(bob, "ledger")

    async def turn(owner: str, name: str, enabled: bool) -> ToolResult:
        tools = {tool.name: tool for tool in factory(owner).tools(child=False)}
        return await tools["set_skill_enabled"].execute(
            SetSkillEnabledInput(name=name, enabled=enabled),
            tool_runtime(tool_name="set_skill_enabled"),
        )

    off = await turn("alice", "weekly-report", False)
    off_again = await turn("alice", "weekly-report", False)

    assert not off.is_error and off_again.text_content == off.text_content
    assert "can no longer be loaded" in off.text_content
    assert "marks it disabled from your next answer run" in off.text_content
    assert _states(alice) == [("weekly-report", False)]

    on = await turn("alice", "weekly-report", True)
    on_again = await turn("alice", "weekly-report", True)

    assert not on.is_error and on_again.text_content == on.text_content
    assert "can be loaded again" in on.text_content
    assert _states(alice) == [("weekly-report", True)]

    # A built-in, a global, an unknown, a malformed and another owner's name are none of the
    # owner's own Skills: each is refused with a list of the owner's, and nothing is written.
    for name in ("skill-creator", "review", "nope", "Bad_Name", "ledger"):
        refused = await turn("alice", name, False)
        assert refused.is_error
        assert "only your own skills can be turned off" in refused.text_content
        assert refused.text_content.endswith("Your skills: weekly-report.")
    assert (await turn("carol", "weekly-report", False)).text_content.endswith("Your skills: none.")
    assert _states(bob) == [("ledger", True)]
    assert list(global_root.rglob(".disabled")) == []


@pytest.mark.asyncio
async def test_a_republished_skill_stays_disabled_and_a_deleted_one_leaves_no_marker_behind(
    tmp_path: Path,
) -> None:
    owner_root = tmp_path / "owner"
    await _publish(owner_root, "review", body="v1")
    await _turn(owner_root, "review", enabled=False)

    republished = await _publish(owner_root, "review", body="v2")

    assert not republished.is_error
    # The new text is the Skill's, but turning it off was the owner's act: it stays off, and
    # the result says so and how to turn it on.
    assert "disabled" in republished.text_content
    assert "set_skill_enabled(name='review', enabled=true)" in republished.text_content
    assert "load_skill now" not in republished.text_content
    assert _states(owner_root) == [("review", False)]
    assert (read_owner_skill(owner_root, "review") or "").endswith("v2")
    assert [path.name for path in owner_root.iterdir()] == ["review"]

    assert delete_owner_skill(owner_root, "review") is True
    assert not (await _publish(owner_root, "review")).is_error
    assert _states(owner_root) == [("review", True)]


@pytest.mark.asyncio
async def test_a_publish_that_cannot_swap_leaves_a_disabled_skill_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner_root = tmp_path / "owner"
    await _publish(owner_root, "review", body="v1")
    await _turn(owner_root, "review", enabled=False)
    rename = Path.rename

    def full_disk(self: Path, target: Path) -> Path:
        if self.name.startswith(".staging-"):
            raise OSError("no space left on device")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", full_disk)

    failed = await _publish(owner_root, "review", body="v2")

    assert failed.is_error and "no space left on device" in failed.text_content
    assert _states(owner_root) == [("review", False)]
    assert (read_owner_skill(owner_root, "review") or "").endswith("v1")
    assert [path.name for path in owner_root.iterdir()] == ["review"]


@pytest.mark.asyncio
async def test_a_skill_turned_off_while_it_is_being_republished_stays_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner_root = tmp_path / "owner"
    await _publish(owner_root, "review", body="v1")
    rename = Path.rename

    def turned_off_as_the_swap_begins(self: Path, target: Path) -> Path:
        if self.name == "review":
            set_owner_skill_enabled(owner_root, "review", False)
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", turned_off_as_the_swap_begins)

    republished = await _publish(owner_root, "review", body="v2")

    # The owner's switch landed after the publish had begun and before it swapped; it is the
    # state the new text is installed in, not one the swap overwrites.
    assert not republished.is_error
    assert _states(owner_root) == [("review", False)]
    assert (read_owner_skill(owner_root, "review") or "").endswith("v2")


def test_the_owners_own_skills_are_listed_on_or_off_in_name_order_and_a_broken_one_is_left_out(
    tmp_path: Path,
) -> None:
    owner_root = tmp_path / "owner"
    _skill(owner_root, "zeta", name="zeta", description="Use when Z", body="ZETA")
    _skill(owner_root, "alpha", name="alpha", description="Use when A", body="ALPHA")
    broken = owner_root / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_text(
        "---\nname: broken\ndescription: Use when: asked\n---\nb", encoding="utf-8"
    )

    turned_off = set_owner_skill_enabled(owner_root, "alpha", False)

    assert turned_off is not None
    assert (turned_off.name, turned_off.description, turned_off.enabled) == (
        "alpha",
        "Use when A",
        False,
    )
    assert [(s.name, s.description, s.enabled) for s in list_owner_skills(owner_root)] == [
        ("alpha", "Use when A", False),
        ("zeta", "Use when Z", True),
    ]
    # A Skill is off exactly while a `.disabled` file sits in its directory.
    assert (owner_root / "alpha" / ".disabled").is_file()
    assert not (owner_root / "zeta" / ".disabled").exists()
    # Its document is the owner's to read whether it is on or off, and a broken one has none.
    assert (read_owner_skill(owner_root, "alpha") or "").endswith("ALPHA")
    assert (read_owner_skill(owner_root, "zeta") or "").endswith("ZETA")
    assert read_owner_skill(owner_root, "broken") is None
    assert list_owner_skills(tmp_path / "nobody") == ()
    # A document past the bound `load_skill` keeps is refused here as it is there.
    (owner_root / "zeta" / "SKILL.md").write_text(
        "---\nname: zeta\ndescription: Use when Z\n---\n" + "z" * 50_000, encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exceeds 50000 characters"):
        read_owner_skill(owner_root, "zeta")

    set_owner_skill_enabled(owner_root, "alpha", True)
    assert not (owner_root / "alpha" / ".disabled").exists()


def test_a_symlinked_skill_or_marker_is_never_followed(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    for name in ("review", "triage"):
        _skill(owner_root, name, name=name, description="Use when asked", body="b")
    _skill(tmp_path, "real", name="real", description="Use when asked", body="b")
    (tmp_path / "somewhere").write_text("keep", encoding="utf-8")
    try:
        (owner_root / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
        (owner_root / "review" / ".disabled").symlink_to(tmp_path / "somewhere")
        (owner_root / "triage" / ".disabled").symlink_to(tmp_path / "nowhere")
    except OSError:
        pytest.skip("symlinks are unavailable")

    # A link in the marker's place is not the marker, whatever it points at, and turning the
    # Skill off writes nothing through it.
    assert _states(owner_root) == [("review", True), ("triage", True)]
    for name in ("review", "triage"):
        with pytest.raises(OSError):
            set_owner_skill_enabled(owner_root, name, False)
    assert (tmp_path / "somewhere").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "nowhere").exists()
    # A linked Skill directory is none of the owner's, so nothing reaches what it points at.
    assert set_owner_skill_enabled(owner_root, "linked", False) is None
    assert read_owner_skill(owner_root, "linked") is None
    with pytest.raises(OSError):
        delete_owner_skill(owner_root, "linked")
    assert (tmp_path / "real" / "SKILL.md").is_file()
    assert not (tmp_path / "real" / ".disabled").exists()


def test_a_name_that_is_not_kebab_case_reaches_no_path(tmp_path: Path) -> None:
    owners = tmp_path / "owners"
    alice = owners / "alice"
    _skill(alice, "review", name="review", description="Use when asked", body="b")
    _skill(owners, "bob", name="bob", description="Use when asked", body="b")

    for name in ("../bob", "bob/../bob", "review/..", "..", ".", "", "Review", ".staging-1"):
        assert read_owner_skill(alice, name) is None
        assert set_owner_skill_enabled(alice, name, False) is None
        assert delete_owner_skill(alice, name) is False

    assert (owners / "bob" / "SKILL.md").is_file()
    assert _states(alice) == [("review", True)]


@pytest.mark.asyncio
async def test_a_skill_turned_off_inside_a_run_stops_loading_at_once_and_the_next_run_lists_it(
    tmp_path: Path,
) -> None:
    factory = SkillsBundleFactory(owner_root=tmp_path / "owners")
    tools = {tool.name: tool for tool in factory("alice").tools(child=False)}

    async def call(tool: str, arguments: BaseModel) -> ToolResult:
        return await tools[tool].execute(arguments, tool_runtime(tool_name=tool))

    await call(
        "publish_skill",
        PublishSkillInput(
            name="weekly-report",
            files=_skill_files(name="weekly-report", description="owner", body="WEEKLY"),
        ),
    )
    assert (await call("load_skill", LoadSkillInput(name="weekly-report"))).text_content.endswith(
        "WEEKLY"
    )

    await call("set_skill_enabled", SetSkillEnabledInput(name="weekly-report", enabled=False))

    # `load_skill` resolves names as they are when it is called; the next Run's catalog says the
    # Skill is off.
    refused = await call("load_skill", LoadSkillInput(name="weekly-report"))
    assert refused.is_error and "disabled by its owner" in refused.text_content
    next_run = str(factory("alice").context_contributions(child=False)[0].messages[0]["content"])
    assert "- weekly-report: disabled by its owner, cannot be loaded" in next_run
    assert factory("bob").context_contributions(child=False) == ()


@pytest.mark.asyncio
async def test_a_folded_description_is_listed_whole_on_one_line(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    folded = (
        "---\nname: weekly-report\ndescription: >\n"
        "  Use when the user asks for a weekly report and\n"
        "  wants the data checked before it is sent.\n---\n# Weekly report\n"
    )
    result = await publish_skill_tool(owner_root).execute(
        PublishSkillInput.model_validate({"name": "weekly-report", "files": {"SKILL.md": folded}}),
        tool_runtime(),  # type: ignore[arg-type]
    )
    assert not result.is_error

    contribution = SkillCatalog.discover(owner_root=owner_root).contribution()

    assert contribution is not None
    assert (
        "- weekly-report: Use when the user asks for a weekly report and wants the data "
        "checked before it is sent. (owner)"
    ) in str(contribution.messages[0]["content"])


@pytest.mark.asyncio
async def test_a_description_that_yaml_would_cut_short_is_refused_at_publish(
    tmp_path: Path,
) -> None:
    owner_root = tmp_path / "owner"

    async def publish(description: str) -> ToolResult:
        content = f"---\nname: triage\ndescription: {description}\n---\n# Triage\n"
        return await publish_skill_tool(owner_root).execute(
            PublishSkillInput.model_validate({"name": "triage", "files": {"SKILL.md": content}}),
            tool_runtime(),  # type: ignore[arg-type]
        )

    cut = await publish("Use when the user names an issue like #123")
    assert cut.is_error and "starts a YAML comment" in cut.text_content
    assert not (owner_root / "triage").exists()

    # Quoting keeps the whole value, and a comment on a line of its own is a comment.
    assert not (await publish('"Use when the user names an issue like #123"')).is_error
    content = "---\nname: triage\ndescription: Use when asked\n# why\n---\nb"
    assert not (
        await publish_skill_tool(owner_root).execute(
            PublishSkillInput.model_validate({"name": "triage", "files": {"SKILL.md": content}}),
            tool_runtime(),  # type: ignore[arg-type]
        )
    ).is_error


def test_a_malformed_skill_is_left_out_of_the_catalog(tmp_path: Path) -> None:
    root = tmp_path / "global"
    _skill(root, "good", name="good", description="Use when asked", body="b")
    broken = root / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_text(
        "---\nname: broken\ndescription: Use when: asked\n---\nb", encoding="utf-8"
    )

    catalog = SkillCatalog.discover(global_root=root)

    assert [skill.name for skill in catalog.metadata] == ["good"]


def test_the_directories_of_a_publish_in_flight_are_not_skills(tmp_path: Path) -> None:
    root = tmp_path / "owner"
    _skill(root, "review", name="review", description="Use when asked", body="LIVE")
    _skill(root, ".staging-1f", name="weekly-report", description="Half written", body="x")
    _skill(root, ".backup-review-1f", name="review", description="Replaced", body="OLD")

    catalog = SkillCatalog.discover(owner_root=root)

    assert [skill.name for skill in catalog.metadata] == ["review"]
    assert catalog.read("review").endswith("LIVE")


def test_a_frontmatter_the_yaml_reader_rejects_costs_only_its_own_skill(tmp_path: Path) -> None:
    root = tmp_path / "global"
    _skill(root, "good", name="good", description="Use when asked", body="b")
    broken = root / "typo"
    broken.mkdir()
    (broken / "SKILL.md").write_text(
        "---\nname: typo\ndescription: Use when asked\nupdated: 2025-02-30\n---\nb",
        encoding="utf-8",
    )

    catalog = SkillCatalog.discover(global_root=root)

    assert [skill.name for skill in catalog.metadata] == ["good"]


def test_a_skill_file_that_is_not_utf8_text_costs_only_its_own_skill(tmp_path: Path) -> None:
    root = tmp_path / "global"
    _skill(root, "good", name="good", description="Use when asked", body="b")
    broken = root / "latin"
    broken.mkdir()
    (broken / "SKILL.md").write_bytes(
        "---\nname: latin\ndescription: Use when asked\n---\ncaf\xe9".encode("latin-1")
    )

    catalog = SkillCatalog.discover(global_root=root)

    assert [skill.name for skill in catalog.metadata] == ["good"]


@pytest.mark.asyncio
async def test_a_failed_skill_load_is_an_error_result_that_says_what_is_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "global"
    _skill(root, "review", name="review", description="Use when asked", body="body")
    tool = load_skill_tool(lambda: SkillCatalog.discover(global_root=root))

    async def load(name: str, path: str = "SKILL.md") -> ToolResult:
        return await tool.execute(
            LoadSkillInput(name=name, path=path),
            tool_runtime(tool_name="load_skill"),  # type: ignore[arg-type]
        )

    assert not (await load("review")).is_error
    unknown = await load("nope")
    missing = await load("review", "references/gone.md")
    escaping = await load("review", "../other/SKILL.md")

    assert unknown.is_error and "no Agent Skill is named 'nope'" in unknown.text_content
    assert missing.is_error
    assert "'references/gone.md' is not a file in Agent Skill 'review'" in missing.text_content
    assert escaping.is_error
