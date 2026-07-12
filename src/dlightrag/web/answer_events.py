# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-facing answer stream presenter for the web UI."""

import dataclasses
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from dlightrag.citations import finalize_answer
from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens
from dlightrag.core.answer_highlights import enrich_semantic_highlights
from dlightrag.core.answer_media import answer_blocks_from_markdown, answer_images_from_sources
from dlightrag.core.answer_turn import PreparedAnswerTurn
from dlightrag.core.client_payloads import project_source_payloads
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.core.scope import RequestScope
from dlightrag.observability import trace_observation
from dlightrag.utils import log_safe
from dlightrag.web.events import (
    AnswerDoneEvent,
    AnswerErrorEvent,
    AnswerMetaEvent,
    AnswerProgressEvent,
    AnswerTraceEvent,
)
from dlightrag.web.safe_html import safe_answer_done, safe_answer_preview, safe_source_panel
from dlightrag.web.sse import sse_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _AnswerPayload:
    done: AnswerDoneEvent
    sources: list[SourceReferencePayload]
    flat_contexts: list[dict[str, Any]]


async def _build_answer_done_payload(
    *,
    clean_answer: str,
    contexts: dict[str, Any],
    image_descriptions: Any,
    manager: Any,
    cfg: Any,
    workspace: str,
    downloadable_workspaces: set[str] | None = None,
) -> _AnswerPayload:
    """Build the done-event payload from retrieval contexts and LLM output."""
    resolver = SourceDownloadLinkBuilder(base_url="/web/files/raw")
    finalized = finalize_answer(
        clean_answer,
        contexts,
        image_url_prefix="/web/images",
        default_workspace=workspace or manager.config.workspace,
    )
    cited_images = answer_images_from_sources(
        finalized.sources,
        contexts={"chunks": finalized.flat_contexts},
    )
    source_payloads = project_source_payloads(
        finalized.sources,
        resolver=resolver,
        downloadable_workspaces=downloadable_workspaces,
    )
    answer_images = cited_images
    answer_blocks = answer_blocks_from_markdown(finalized.answer, cited_images)

    done = AnswerDoneEvent(
        html=safe_answer_done(
            answer=finalized.answer,
            sources=source_payloads,
            answer_images=answer_images,
        ),
        answer=finalized.answer,
        image_descriptions=image_descriptions,
        answer_images=answer_images,
        answer_blocks=answer_blocks,
    )
    return _AnswerPayload(done, source_payloads, finalized.flat_contexts)


# ---------------------------------------------------------------------------
# Main SSE stream
# ---------------------------------------------------------------------------


async def stream_answer_events(
    *,
    manager: Any,
    cfg: Any,
    turn: PreparedAnswerTurn,
    workspaces: list[str] | None,
    workspace: str,
    scope: RequestScope | None = None,
    downloadable_workspaces: set[str] | None = None,
) -> AsyncIterator[str]:
    """Yield browser SSE events for one answer request, under a request-root span."""
    ws_list = workspaces or [workspace or manager.config.workspace]
    async with trace_observation(
        "answer_stream_pipeline",
        as_type="chain",
        input={"query": turn.current_query},
        metadata={
            "workspaces": ws_list,
            "history_turns": len(turn.text_history),
        },
    ):
        async for event in _emit_answer_events(
            manager=manager,
            cfg=cfg,
            turn=turn,
            ws_list=ws_list,
            workspace=workspace,
            scope=scope,
            downloadable_workspaces=downloadable_workspaces,
        ):
            yield event


async def _emit_answer_events(
    *,
    manager: Any,
    cfg: Any,
    turn: PreparedAnswerTurn,
    ws_list: list[str],
    workspace: str,
    scope: RequestScope | None = None,
    downloadable_workspaces: set[str] | None = None,
) -> AsyncIterator[str]:
    """Emit the SSE event sequence for one answer request."""
    full_answer = ""
    token_iter: AsyncIterator[str] | str | None = None
    try:
        history_kept = len(turn.text_history)
        yield sse_event("meta", AnswerMetaEvent(history_kept=history_kept))

        t0 = time.monotonic()
        logger.debug("[SSE] query received: %s", log_safe(turn.current_query))

        yield sse_event("progress", AnswerProgressEvent(phase="planning"))

        contexts, token_iter = await manager._aanswer_stream_prepared(
            turn,
            workspaces=ws_list,
            scope=scope,
        )
        t1 = time.monotonic()
        logger.info("[SSE] retrieval+stream setup done (%.1fs)", t1 - t0)

        yield sse_event("progress", AnswerProgressEvent(phase="generating"))
        logger.info("[SSE] stream started")

        # ── Stream tokens ──────────────────────────────────────────
        accumulated_text = ""
        last_preview_ts = 0.0
        last_preview_len = 0

        async for chunk in iter_answer_tokens(
            token_iter, idle_timeout=manager.config.answer_stream_idle_timeout
        ):
            full_answer += chunk
            accumulated_text += chunk
            yield sse_event("token", chunk)
            now = time.monotonic()
            new_chars = len(accumulated_text) - last_preview_len
            if now - last_preview_ts > 0.3 and new_chars > 20:
                yield sse_event("preview", safe_answer_preview(accumulated_text))
                last_preview_ts = now
                last_preview_len = len(accumulated_text)

        clean_answer = getattr(token_iter, "answer", None) or full_answer
        image_descriptions = getattr(token_iter, "image_descriptions", []) or []

        # ── Build done payload ─────────────────────────────────────
        effective_workspace = workspace or manager.config.workspace
        payload = await _build_answer_done_payload(
            clean_answer=clean_answer,
            contexts=contexts,
            image_descriptions=image_descriptions,
            manager=manager,
            cfg=cfg,
            workspace=effective_workspace,
            downloadable_workspaces=downloadable_workspaces,
        )

        yield sse_event("done", payload.done)

        # ── Post-done enrichment (trace, highlights) ───────────────
        trace = getattr(token_iter, "trace", None)
        if isinstance(trace, dict) and trace:
            yield sse_event("trace", AnswerTraceEvent(trace=trace))

        highlighted_sources = await enrich_semantic_highlights(
            payload.sources,
            answer_text=payload.done.answer,
            config=cfg,
        )
        has_highlights = any(
            chunk.highlight_phrases
            for source in highlighted_sources
            if source.chunks
            for chunk in source.chunks
        )
        if has_highlights:
            yield sse_event("highlights", safe_source_panel(sources=highlighted_sources))

    except Exception:
        logger.exception("Answer streaming failed")
        yield sse_event("error", AnswerErrorEvent(message="Service error. Please try again."))
    finally:
        await aclose_answer_stream(token_iter)


__all__ = ["stream_answer_events"]
