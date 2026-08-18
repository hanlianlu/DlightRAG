# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The final M3 durable Answer result and its strict read-side projection.

The sole durable M3 result shape is: ``answer``, ``answer_sources``,
``report_sources``, nullable ``primary_report``, ``artifacts``,
``answer_images``, ``trace``, nullable ``usage``, and ``image_descriptions``
(M3-D5). ``contexts``, ``references``, and answer blocks are not durable
fields: references and blocks are derived on read from the durable source
identities, so they are never stored twice.

No production store reads this decoder yet; it lands ahead of the Task 4
persistence switch.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dlightrag.answer.runs.results import answer_blocks_from_markdown


@dataclass(frozen=True, slots=True)
class DurableSourceIdentity:
    """One durable source identity the result references.

    Transport-facing downloads and visual authorization are projected at read
    time; only the identity and its locator are durable.
    """

    id: str
    source_uri: str
    title: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DurableSourceIdentity:
        if not isinstance(payload.get("id"), str) or not payload["id"].strip():
            raise ValueError("durable source requires a non-empty id")
        if not isinstance(payload.get("source_uri"), str) or not payload["source_uri"].strip():
            raise ValueError("durable source requires a non-empty source_uri")
        title = payload.get("title")
        if title is not None and not isinstance(title, str):
            raise ValueError("durable source title must be a string")
        return cls(id=payload["id"], source_uri=payload["source_uri"], title=title)

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "source_uri": self.source_uri}
        if self.title is not None:
            payload["title"] = self.title
        return payload


@dataclass(frozen=True, slots=True)
class FinalAnswerResult:
    """The validated M3 result shape for new runs."""

    answer: str
    answer_sources: tuple[DurableSourceIdentity, ...]
    trace: Mapping[str, Any]
    report_sources: tuple[DurableSourceIdentity, ...] = ()
    primary_report: str | None = None
    artifacts: tuple[Mapping[str, Any], ...] = ()
    answer_images: tuple[Mapping[str, Any], ...] = ()
    usage: Mapping[str, Any] | None = None
    image_descriptions: tuple[str, ...] = ()


def validate_final_result(payload: Mapping[str, Any]) -> FinalAnswerResult:
    """Strictly decode one durable M3 result; malformed shapes raise.

    Unknown fields are rejected so a stored payload can never silently carry a
    checkpoint-era or future-milestone shape.
    """
    allowed = {
        "answer",
        "answer_sources",
        "report_sources",
        "primary_report",
        "artifacts",
        "answer_images",
        "trace",
        "usage",
        "image_descriptions",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"final result has unknown fields: {sorted(unknown)}")

    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise ValueError("final result requires a string answer")
    raw_sources = payload.get("answer_sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ValueError("final result requires an answer_sources list")
    answer_sources = tuple(DurableSourceIdentity.from_payload(source) for source in raw_sources)

    report_sources = _optional_sources(payload.get("report_sources"), name="report_sources")
    primary_report = payload.get("primary_report")
    if primary_report is not None and not isinstance(primary_report, str):
        raise ValueError("final result primary_report must be a string or null")

    artifacts = payload.get("artifacts", ())
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ValueError("final result requires an artifacts list")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("final result artifacts must be objects")

    answer_images = payload.get("answer_images", ())
    if not isinstance(answer_images, Sequence) or isinstance(answer_images, (str, bytes)):
        raise ValueError("final result requires an answer_images list")
    for image in answer_images:
        if not isinstance(image, Mapping):
            raise ValueError("final result answer_images must be objects")

    trace = payload.get("trace")
    if not isinstance(trace, Mapping):
        raise ValueError("final result requires a trace object")

    usage = payload.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise ValueError("final result usage must be an object or null")

    image_descriptions = payload.get("image_descriptions", ())
    if not isinstance(image_descriptions, Sequence) or isinstance(image_descriptions, (str, bytes)):
        raise ValueError("final result requires an image_descriptions list")
    for description in image_descriptions:
        if not isinstance(description, str):
            raise ValueError("final result image descriptions must be strings")

    return FinalAnswerResult(
        answer=answer,
        answer_sources=answer_sources,
        trace=dict(trace),
        report_sources=report_sources,
        primary_report=primary_report,
        artifacts=tuple(dict(artifact) for artifact in artifacts),
        answer_images=tuple(dict(image) for image in answer_images),
        usage=dict(usage) if usage is not None else None,
        image_descriptions=tuple(image_descriptions),
    )


def derive_references(
    result: FinalAnswerResult,
) -> list[dict[str, Any]]:
    """Derive validated cited-document references from durable source identities.

    References are a read-side projection: the same answer over the same
    sources always derives the same references, so they are never durable.
    """
    return [
        {"id": source.id, "title": source.title or "Source"} for source in result.answer_sources
    ]


def derive_answer_blocks(
    result: FinalAnswerResult,
) -> list[dict[str, Any]]:
    """Derive markdown/image_ref answer blocks from the answer and its images."""
    images = [dict(image) for image in result.answer_images if isinstance(image, Mapping)]
    return answer_blocks_from_markdown(result.answer, images)


def _optional_sources(
    value: Any,
    *,
    name: str,
) -> tuple[DurableSourceIdentity, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"final result {name} must be a list or null")
    return tuple(DurableSourceIdentity.from_payload(source) for source in value)


__all__ = [
    "DurableSourceIdentity",
    "FinalAnswerResult",
    "derive_answer_blocks",
    "derive_references",
    "validate_final_result",
]
