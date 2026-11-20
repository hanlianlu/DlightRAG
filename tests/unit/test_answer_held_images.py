# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a Run looked at reaches the Answer, through the one placement grammar it already had.

A Research Run holds the images it read -- screenshots, and the pictures it viewed on the pages it
fetched. Before this, an Answer could only ever draw an Evidence Image that retrieval had selected
for a cited corpus chunk, so on an empty corpus a Run could read a dozen photographs and place none
of them: the Model's only remaining move was to describe the pictures in prose, which is what one
reported Answer did three turns in a row. These tests pin the rule that closed that gap, and pin it
at the two seams that decide it: what the Run admits, and how a transport addresses it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.engine.answer.media import held_evidence_images
from dlightrag.engine.answer.results import project_answer_result, store_answer_result

_RUN_ID = "0199a0a0-0000-7000-8000-0000000000aa"
#: The id a `view` call advertises: the resource whose bytes a reader's request is answered from --
#: the picture itself, or the rendered representation of a document. It is in the Run's manifest,
#: which is what makes it placeable at all.
#: The id a `view` call advertises for the pixels it prepared, and the id a screenshot carries.
#: Both are durable Run resources; neither is in the Resource registry, which is the point.
_VIEWED = "res-view-fedcba9876543210"
_SHOT = "res-shot-0123456789abcdef"


def _held(answer: str, *held: str, **kwargs: Any) -> list[dict[str, Any]]:
    """What the Answer places of the images the Run holds."""
    return held_evidence_images(answer, held=held, run_id=_RUN_ID, **kwargs)


def _stored(answer: str, images: list[dict[str, Any]]) -> dict[str, Any]:
    return store_answer_result(
        answer=answer,
        contexts={},
        sources=[],
        evidence_images=images,
        trace={},
        image_descriptions=[],
    )


def test_a_placed_held_image_becomes_evidence_and_an_unplaced_one_does_not() -> None:
    """Placement is the whole selection rule: the Model names what the reader should see.

    A Run holds every image on every page it read -- logos, thumbnails, furniture -- so admitting
    them all would bury the Answer. An image the Model only looked at costs the reader nothing, and
    one it names is admitted.
    """
    answer = f"Today's picture:\n\n![Rotation Station](evidence:{_VIEWED})\n"

    images = _held(answer, _VIEWED)

    assert [image["id"] for image in images] == [_VIEWED]
    assert images[0]["resource_id"] == _VIEWED
    assert images[0]["workspace"] == ""
    assert images[0]["chunk_id"] == ""


def test_every_held_resource_id_the_answer_names_is_admitted() -> None:
    """A viewed picture and a screenshot are both Resources the Run holds, and both are placeable.

    An earlier cut admitted only ids the Resource registry knew; tool attachments are durable
    resources the registry is never told about, so that gate denied the Answer exactly the images
    the Agent had gone and looked at -- and screenshots permanently, because recovery skips their
    rows when it rebuilds the registry. Whether the bytes exist, and whether this reader may have
    them, is the run-resource route's question, answered against the Run's own rows.
    """
    placed = f"![Viewed](evidence:{_VIEWED})\n![Shot](evidence:{_SHOT})"

    images = _held(placed, _VIEWED, _SHOT)

    assert [image["id"] for image in images] == sorted([_SHOT, _VIEWED])
    assert {image["resource_id"] for image in images} == {_SHOT, _VIEWED}


def test_an_id_named_twice_is_admitted_once() -> None:
    """The same picture placed twice is one Evidence Image with two placements."""
    answer = f"![a](evidence:{_VIEWED})\n\n![b](evidence:{_VIEWED})"

    assert [image["id"] for image in _held(answer, _VIEWED)] == [_VIEWED]


def test_a_marker_the_answer_never_places_admits_nothing() -> None:
    """No marker, no image: an Answer that describes its pictures admits none of them."""
    assert _held("I looked at twelve photographs of the Pantanal.", _VIEWED) == []


def test_a_held_image_is_addressed_on_the_run_resource_route() -> None:
    """The reader gets the bytes the Model looked at, from the route that already serves them.

    Nothing is copied and no second registry is built: the same resource id the Run holds is the
    address the browser asks for, which is also why a surface that serves no such route (MCP keeps
    its calls cookie-free) is left with no address rather than an invented one.
    """
    answer = f"![Shot](evidence:{_VIEWED})"
    stored = _stored(answer, _held(answer, _VIEWED))

    web = project_answer_result(
        stored,
        run_id=_RUN_ID,
        artifact_url_prefix="/web/api/answer",
        run_resource_prefix="/web/api/runs",
    )
    assert web["evidence_images"][0]["url"] == f"/web/api/runs/{_RUN_ID}/resources/{_VIEWED}"

    cookieless = project_answer_result(stored, run_id=_RUN_ID)
    assert cookieless["evidence_images"][0]["url"] == ""

    # Both provenances keep working: a corpus image is still addressed under its own chunk.
    corpus = _stored("", [])
    corpus["evidence_images"] = [
        {
            "id": "chunk-1",
            "chunk_id": "chunk-1",
            "workspace": "default",
            "source_ref": "1",
            "label": "Page 3",
            "answer_image_sent": True,
        }
    ]
    projected = project_answer_result(corpus, image_url_prefix="/web/api/images")
    assert projected["evidence_images"][0]["url"] == "/web/api/images/default/chunk-1?size=full"


def test_the_web_presentation_places_a_held_image_inline() -> None:
    """End of the path: the marker becomes an inline picture, not a dangling token.

    This is the assertion the reported Answer would have failed. Its turns named pictures in prose
    because a run-held image had no id to name and no route to address; here the same shape of
    answer comes out as one markdown part carrying an inline Evidence Image placement.
    """
    answer = (
        f"## Today\n\n![Rotation Station](evidence:{_VIEWED})\n\nthe wind farm at Tehachapi Pass."
    )
    stored = _stored(answer, _held(answer, _VIEWED))
    projected = project_answer_result(
        stored,
        run_id=_RUN_ID,
        artifact_url_prefix="/web/api/answer",
        run_resource_prefix="/web/api/runs",
    )

    presentation = build_answer_presentation(
        answer=projected["answer"],
        sources=projected["sources"],
        evidence_images=projected["evidence_images"],
        artifacts=projected["artifacts"],
        artifact_outcome=projected["artifact_outcome"],
    )

    inline = [part for part in presentation.parts if part.type == "evidence_image" and part.inline]
    assert len(inline) == 1
    image = inline[0].evidence_image
    assert image is not None
    assert image.id == _VIEWED
    assert image.label == "Rotation Station"
    assert image.url == f"/web/api/runs/{_RUN_ID}/resources/{_VIEWED}"
    # Placed inline, so the default Visual Evidence strip does not repeat it below the Answer.
    assert presentation.evidence_images == []


def test_the_tool_advertises_exactly_the_syntax_the_answer_accepts() -> None:
    """The two halves are wired to each other, and this is where a break would show.

    A `view` result tells the Model how to place the picture, and the Answer stage decides what it
    may place. If either side drifts -- a different scheme, a label that is not the one the reader
    sees, an id nothing can resolve -- the Model writes a marker that resolves to nothing and the
    reader gets a literal ``![...](...)`` in the Answer. That failure is silent, which is why the
    seam is asserted rather than assumed.
    """
    from dlightrag.engine.answer.tools.resources import _placement

    advertised = _placement(_VIEWED, "Rotation Station")

    assert (
        advertised
        == "\nTo show the reader this image: ![Rotation Station](evidence:" + _VIEWED + ")"
    )
    assert [image["id"] for image in _held(advertised, _VIEWED)] == [_VIEWED]


def test_an_id_retrieval_already_claimed_is_not_claimed_again_as_a_held_image() -> None:
    """One ``evidence:`` namespace, two provenances, and the corpus keeps what it claimed.

    A cited chunk's id and a Run-held Resource id are drawn from the same marker syntax, so a
    projection that let both admit one id would carry two records for it -- last one wins -- and
    address the corpus figure on the run-resource route, which answers not-found. A picture that
    rendered inline before would stop rendering, so the split is asserted rather than assumed.
    """
    answer = f"![Page 3](evidence:chunk-9)\n\n![Shot](evidence:{_SHOT})"
    cited = [{"id": "chunk-9", "chunk_id": "chunk-9", "workspace": "default"}]

    held = _held(answer, _SHOT, "chunk-9", cited=cited)

    assert [image["id"] for image in held] == [_SHOT]
    assert "chunk-9" not in {image["id"] for image in held}


def test_an_id_the_run_holds_no_image_for_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Model writes these ids itself, and can write one that names nothing.

    A record for it would be addressed on the run-resource route, which answers not-found, so the
    reader would get a broken image where the Model meant a picture. No record, and one warning
    that names the Run and the ids, is the whole of it.
    """
    invented = "res-view-00000000000000000000000000000000"
    answer = f"![Real](evidence:{_VIEWED})\n\n![Invented](evidence:{invented})"

    with caplog.at_level(logging.WARNING, logger="dlightrag.engine.answer.media"):
        images = _held(answer, _VIEWED)

    assert [image["id"] for image in images] == [_VIEWED]
    [record] = caplog.records
    assert _RUN_ID in record.getMessage()
    assert invented in record.getMessage()
    assert _VIEWED not in record.getMessage()


def test_an_id_retrieval_claimed_is_not_logged_as_one_the_run_lacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retrieval's own ids are the corpus's to place, so they are no mistake of the Model's."""
    answer = "![Page 3](evidence:chunk-9)"
    cited = [{"id": "chunk-9", "chunk_id": "chunk-9", "workspace": "default"}]

    with caplog.at_level(logging.WARNING, logger="dlightrag.engine.answer.media"):
        assert _held(answer, cited=cited) == []

    assert caplog.records == []


def test_a_dropped_id_leaves_the_reader_the_alt_text_and_no_broken_image() -> None:
    """With no record, the marker is no placement: the Web presentation draws no slot for it.

    The browser then gets an ``<img>`` with an alt and no source, which reads as its alt text, where
    an image with a source the route cannot serve would have shown a broken-image icon.
    """
    answer = "![Rotation Station](evidence:res-view-00000000000000000000000000000000)"
    stored = _stored(answer, _held(answer, _VIEWED))
    projected = project_answer_result(
        stored,
        run_id=_RUN_ID,
        artifact_url_prefix="/web/api/answer",
        run_resource_prefix="/web/api/runs",
    )

    assert projected["evidence_images"] == []
    assert [part["type"] for part in projected["parts"]] == ["markdown"]

    presentation = build_answer_presentation(
        answer=projected["answer"],
        sources=projected["sources"],
        evidence_images=projected["evidence_images"],
        artifacts=projected["artifacts"],
        artifact_outcome=projected["artifact_outcome"],
    )

    assert [part.type for part in presentation.parts] == ["markdown"]
    assert presentation.parts[0].html is not None
    assert 'alt="Rotation Station"' in presentation.parts[0].html
    assert "src=" not in presentation.parts[0].html
