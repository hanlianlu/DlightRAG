# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe semantic projection shared by browser history, SSE, and Artifact views."""

import html as html_module
from collections.abc import Callable, Mapping
from typing import Any, Literal

import nh3
from pydantic import Field

from dlightrag.adapters.http.browser.markdown import (
    LinkedImagePlacement,
    inject_highlights,
    normalize_chunk_source,
    render_chunk_content,
    render_markdown,
)
from dlightrag.adapters.http.browser.run_resources import rewrite_image_sources
from dlightrag.adapters.http.browser.safe_html import sanitize_html_fragment
from dlightrag.adapters.http.browser.video_playback import VideoPlaybackLink, video_playback_link
from dlightrag.application.corpus_admin import validate_public_web_url
from dlightrag.engine.answer.citations.contracts import (
    SourceReferencePayload,
)
from dlightrag.engine.answer.citations.syntax import Citation
from dlightrag.engine.answer.client_contracts import ClientContractModel
from dlightrag.engine.answer.reference import resolve_artifact_target

_CHUNK_ALLOWED_TAGS = {
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "caption",
    "colgroup",
    "col",
    "p",
    "br",
    "hr",
    "b",
    "i",
    "em",
    "strong",
    "u",
    "s",
    "del",
    "sub",
    "sup",
    "mark",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "span",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "pre",
    "code",
    "a",
    "blockquote",
    "abbr",
    "details",
    "summary",
}
_CHUNK_ALLOWED_ATTRS: dict[str, set[str]] = {
    "*": {"class"},
    "a": {"href", "title"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan", "scope"},
    "col": {"span"},
    "colgroup": {"span"},
}


class PresentationImage(ClientContractModel):
    id: str = ""
    chunk_id: str = ""
    source_ref: str = ""
    url: str
    thumbnail_url: str
    label: str = ""
    answer_image_sent: bool = True


class PresentationArtifactIssue(ClientContractModel):
    kind: str
    description: str
    resource_id: str | None = None


class PresentationArtifact(ClientContractModel):
    resource_id: str
    media_type: str
    label: str
    filename: str
    byte_size: int
    digest: str
    presentation: Literal["image", "video", "markdown", "html", "pdf", "text", "download"]
    status: Literal["available", "unavailable"]
    uri: str
    width: int | None = None
    height: int | None = None
    data_url: str | None = None
    download_url: str | None = None
    presentation_url: str | None = None
    issue: PresentationArtifactIssue | None = None


class PresentationArtifactOutcome(ClientContractModel):
    status: Literal["complete", "partial", "failed"] = "complete"
    issues: list[PresentationArtifactIssue] = Field(default_factory=list)


class PresentationLinkCard(ClientContractModel):
    """One link the page itself declared to be a video (ADR 0026).

    The preview is a link out. A separate video_links descriptor may offer
    reader-activated playback (ADR 0028); OG metadata never grants that authority.
    """

    url: str
    title: str = ""
    description: str = ""
    site: str = ""
    image: str | None = None


class PresentationPart(ClientContractModel):
    type: Literal["markdown", "artifact", "evidence_image", "link_card"]
    text: str = ""
    html: str = ""
    artifact: PresentationArtifact | None = None
    evidence_image: PresentationImage | None = None
    card: PresentationLinkCard | None = None
    inline: bool = False
    # Server-created placeholder in the full Markdown HTML, not a source offset.
    slot: int | None = Field(default=None, ge=0)


class PresentationSourceChunk(ClientContractModel):
    chunk_idx: int | None = None
    page_number: int | None = None
    content_html: str = ""
    image_url: str | None = None
    thumbnail_url: str | None = None


class PresentationSource(ClientContractModel):
    id: str
    title: str
    source_url: str | None = None
    download_url: str | None = None
    chunks: list[PresentationSourceChunk]


class AnswerPresentation(ClientContractModel):
    video_links: list[VideoPlaybackLink] = Field(default_factory=list)
    answer_text: str
    parts: list[PresentationPart]
    sources: list[PresentationSource]
    evidence_images: list[PresentationImage]
    link_cards: list[PresentationLinkCard] = Field(default_factory=list)
    artifacts: list[PresentationArtifact]
    artifact_outcome: PresentationArtifactOutcome


def _reference_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    ref = str(ref_id)
    return ref if chunk_idx is None or chunk_idx == "" else f"{ref}-{chunk_idx}"


def _reference_aria_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    ref = str(ref_id)
    return f"Source {ref}" if chunk_idx in {None, ""} else f"Source {ref}, chunk {chunk_idx}"


def render_answer_html(
    answer: str,
    *,
    known_sources: Mapping[str, str],
    place_resource: Callable[[str, str, bool], str | None] | None = None,
    place_linked_image: Callable[[str, str], LinkedImagePlacement | None] | None = None,
    citation_links: Mapping[str, str] | None = None,
) -> str:
    """Render one Markdown segment with semantic citation controls.

    ``known_sources`` maps each source id this surface publishes to its title. A
    marker outside that set stays literal text: badging it would promise a source
    the click cannot open. A badge carries its source title as a tooltip, which is
    the only source name a private corpus document has in the interface.
    """

    def render_citation(citation: Citation) -> str:
        ref_id, chunk_idx = citation.ref_id, citation.chunk_idx
        title = known_sources.get(ref_id)
        if title is None:
            return html_module.escape(citation.marker)
        chunk = f' data-chunk="{chunk_idx}"' if chunk_idx is not None else ""
        return (
            f'<cite class="citation-badge" data-ref="{ref_id}"{chunk} '
            f'role="button" tabindex="0" title="{html_module.escape(title, quote=True)}" '
            f'aria-label="{_reference_aria_label(ref_id, chunk_idx)}">'
            f"{_reference_label(ref_id, chunk_idx)}</cite>"
        )

    return sanitize_html_fragment(
        render_markdown(
            answer,
            place_resource=place_resource,
            place_linked_image=place_linked_image,
            citation_links=citation_links,
            render_citation=render_citation,
        )
    )


def render_source_chunk_html(content: str, phrases: list[str] | None = None) -> str:
    source = normalize_chunk_source(content)
    html = nh3.clean(
        render_chunk_content(source), tags=_CHUNK_ALLOWED_TAGS, attributes=_CHUNK_ALLOWED_ATTRS
    )
    return inject_highlights(html, source, phrases) if phrases else html


def _public_source_url(value: str) -> str | None:
    try:
        return validate_public_web_url(value.strip())
    except ValueError:
        return None


def _presentation_source(value: SourceReferencePayload | dict[str, Any]) -> PresentationSource:
    source = SourceReferencePayload.model_validate(value)
    return PresentationSource(
        id=source.id,
        title=source.title or "Source",
        source_url=_public_source_url(source.source_uri),
        download_url=source.download_url,
        chunks=[
            PresentationSourceChunk(
                chunk_idx=chunk.chunk_idx,
                page_number=chunk.page_number,
                content_html=render_source_chunk_html(chunk.content, chunk.highlight_phrases)
                if chunk.content
                else "",
                image_url=chunk.image_url,
                thumbnail_url=chunk.thumbnail_url,
            )
            for chunk in (source.chunks or [])
        ],
    )


def build_answer_presentation(
    *,
    answer: str,
    sources: list[SourceReferencePayload] | list[dict[str, Any]],
    evidence_images: list[dict[str, Any]],
    artifacts: list[dict[str, Any]] | None = None,
    artifact_outcome: dict[str, Any] | None = None,
    artifact_bindings: Mapping[str, str] | None = None,
    image_rewrites: Mapping[str, str] | None = None,
    link_cards: list[dict[str, Any]] | None = None,
) -> AnswerPresentation:
    """Build the bounded Web projection used identically by SSE and history."""
    # Document bindings are consumed by this adapter, not by the browser widget.
    artifact_values = [
        {key: value for key, value in item.items() if key != "artifact_bindings"}
        for item in artifacts or []
    ]
    image_values = evidence_images
    card_values = link_cards or []
    # Validate the sources once: they are both the payload this surface publishes
    # and the ref set its citation badges may point at.
    presentation_sources = [_presentation_source(source) for source in sources]
    known_sources = {source.id: source.title for source in presentation_sources}
    citation_links = {
        source.id: source.source_url for source in presentation_sources if source.source_url
    }
    artifacts_by_id = {str(item.get("resource_id") or ""): item for item in artifact_values}
    images_by_id = {str(item.get("id") or ""): item for item in image_values}
    placements: list[PresentationPart] = []
    video_links: dict[str, VideoPlaybackLink] = {}
    linked_evidence_ids: set[str] = set()

    def place_resource(href: str, label: str, image: bool) -> str | None:
        if not image and (video := video_playback_link(href)) is not None:
            video_links[video.url] = video
        scheme, _, evidence_resource = href.partition(":")
        resource = resolve_artifact_target(href, artifact_bindings or {})
        slot = len(placements)
        if resource in artifacts_by_id:
            item = artifacts_by_id[resource]
            placements.append(
                PresentationPart(
                    type="artifact",
                    artifact=PresentationArtifact.model_validate(
                        {**item, "label": label or item.get("label")}
                    ),
                    inline=image,
                    slot=slot,
                )
            )
        elif scheme.lower() == "evidence" and image and evidence_resource in images_by_id:
            item = images_by_id[evidence_resource]
            placements.append(
                PresentationPart(
                    type="evidence_image",
                    evidence_image=PresentationImage.model_validate(
                        {**item, "label": label or item.get("label")}
                    ),
                    inline=True,
                    slot=slot,
                )
            )
        else:
            return None
        return f'<span class="answer-resource-slot-{slot}"></span>'

    def place_linked_image(href: str, label: str) -> LinkedImagePlacement | None:
        resource = resolve_artifact_target(href, artifact_bindings or {})
        item = artifacts_by_id.get(resource or "")
        if item is not None:
            if (
                item.get("status") == "available"
                and item.get("presentation") == "image"
                and item.get("data_url")
            ):
                return LinkedImagePlacement(image_url=str(item["data_url"]))
            # Failed resources and non-image Artifacts remain actionable in a
            # normal slot. Replace the containing anchor to avoid nesting its
            # controls, and do not pretend a PDF/video is a passive image.
            return LinkedImagePlacement(replacement_html=place_resource(href, label, False))
        scheme, _, evidence_resource = href.partition(":")
        if scheme.lower() == "evidence" and evidence_resource in images_by_id:
            image = images_by_id[evidence_resource]
            source = image.get("url") or image.get("thumbnail_url")
            if source:
                linked_evidence_ids.add(evidence_resource)
                return LinkedImagePlacement(image_url=str(source))
        return None

    # Parse the whole answer exactly once. Splitting around resource-looking
    # source text first could turn code or a title into a new link on reparse.
    rendered = render_answer_html(
        answer,
        known_sources=known_sources,
        place_resource=place_resource,
        place_linked_image=place_linked_image,
        citation_links=citation_links,
    )
    parts = [
        PresentationPart(
            type="markdown",
            text=answer,
            html=rewrite_image_sources(rendered, image_rewrites or {}),
        ),
        *placements,
    ]
    inline_evidence = linked_evidence_ids | {
        part.evidence_image.id
        for part in parts
        if part.type == "evidence_image" and part.evidence_image is not None
    }
    return AnswerPresentation(
        video_links=list(video_links.values()),
        answer_text=answer,
        parts=parts,
        sources=presentation_sources,
        evidence_images=[
            PresentationImage.model_validate(image)
            for image in image_values
            if str(image.get("id") or "") not in inline_evidence
        ],
        link_cards=[PresentationLinkCard.model_validate(card) for card in card_values],
        artifacts=[PresentationArtifact.model_validate(item) for item in artifact_values],
        artifact_outcome=PresentationArtifactOutcome.model_validate(
            artifact_outcome or {"status": "complete", "issues": []}
        ),
    )


__all__ = [
    "AnswerPresentation",
    "PresentationArtifact",
    "PresentationArtifactIssue",
    "PresentationArtifactOutcome",
    "PresentationImage",
    "PresentationLinkCard",
    "PresentationPart",
    "PresentationSource",
    "PresentationSourceChunk",
    "build_answer_presentation",
    "render_answer_html",
    "render_source_chunk_html",
]
