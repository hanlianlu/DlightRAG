# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-facing answer stream presenter for the web UI."""

import dataclasses
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from dlightrag.citations import finalize_answer
from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens
from dlightrag.core.answer_highlights import enrich_semantic_highlights
from dlightrag.core.answer_media import answer_blocks_from_markdown, answer_images_from_sources
from dlightrag.core.client_payloads import project_source_payloads
from dlightrag.core.retrieval.source_url_resolver import SourceUrlResolver
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


async def _session_image_cards(
    *,
    manager: Any,
    session_id: str,
    image_ids: list[str],
    image_descriptions: Any,
    scope: RequestScope | None,
) -> list[dict[str, Any]]:
    stored_session_images: list[str] = []
    if image_ids:
        try:
            stored_session_images = await manager.aget_session_image_data(
                session_id,
                image_ids,
                scope=scope,
            )
        except Exception:
            logger.warning("Failed to load %d session images", len(image_ids), exc_info=True)
            stored_session_images = []

    answer_images: list[dict[str, Any]] = []
    for i, cid in enumerate(image_ids):
        desc = ""
        if isinstance(image_descriptions, dict):
            desc = image_descriptions.get(cid, "")
        elif isinstance(image_descriptions, list) and i < len(image_descriptions):
            desc = image_descriptions[i]
        data_url = stored_session_images[i] if i < len(stored_session_images) else ""
        answer_images.append(
            {
                "id": cid,
                "chunk_id": cid,
                "source_ref": "",
                "url": data_url,
                "thumbnail_url": data_url,
                "label": desc or f"Visual {i + 1}",
            }
        )
    return answer_images


@dataclasses.dataclass(frozen=True, slots=True)
class _AnswerPayload:
    done: AnswerDoneEvent
    sources: list[Any]
    flat_contexts: list[dict[str, Any]]


async def _build_answer_done_payload(
    *,
    clean_answer: str,
    contexts: dict[str, Any],
    current_image_ids: list[str],
    image_descriptions: Any,
    manager: Any,
    session_id: str,
    scope: RequestScope | None,
    cfg: Any,
    workspace: str,
) -> _AnswerPayload:
    """Build the done-event payload from retrieval contexts and LLM output."""
    session_cards = await _session_image_cards(
        manager=manager,
        session_id=session_id,
        image_ids=current_image_ids,
        image_descriptions=image_descriptions,
        scope=scope,
    )
    resolver = SourceUrlResolver(
        input_dir=str(cfg.input_dir_path),
        workspace=workspace or manager.config.workspace,
    )
    finalized = finalize_answer(
        clean_answer,
        contexts,
        image_url_prefix="/web/images",
        default_workspace=workspace or manager.config.workspace,
    )
    source_payloads = project_source_payloads(finalized.sources, resolver=resolver)
    cited_images = answer_images_from_sources(
        finalized.sources,
        contexts={"chunks": finalized.flat_contexts},
    )
    answer_images = session_cards + cited_images
    answer_blocks = answer_blocks_from_markdown(finalized.answer, cited_images)

    done = AnswerDoneEvent(
        html=safe_answer_done(
            answer=finalized.answer,
            sources=source_payloads,
            answer_images=answer_images,
        ),
        answer=finalized.answer,
        current_image_ids=current_image_ids,
        image_descriptions=image_descriptions,
        answer_images=answer_images,
        answer_blocks=answer_blocks,
    )
    return _AnswerPayload(done, finalized.sources, finalized.flat_contexts)


async def _save_checkpoint(
    *,
    manager: Any,
    session_id: str,
    query: str,
    answer: str,
    query_images: list[dict[str, Any]],
    current_image_ids: list[str],
    contexts: dict[str, Any],
    sources: list[Any],
    workspace: str,
    scope: RequestScope | None,
) -> bool:
    """Persist conversation turn.  Returns False when the save failed."""
    save_checkpoint = getattr(manager, "asave_turn_checkpoint", None)
    if not callable(save_checkpoint):
        return True  # no checkpoint backend — not a failure

    save_checkpoint_func = cast(Callable[..., Awaitable[None]], save_checkpoint)
    try:
        cited_ids = [s.id for s in sources if s.id]

        query_content: list[dict[str, Any]] | None = None
        if query_images and current_image_ids:
            query_content = [{"type": "text", "text": query}]
            for img_id in current_image_ids:
                query_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"dlightrag-image://{img_id}"},
                    }
                )

        answer_content: list[dict[str, Any]] | None = None
        if answer:
            answer_content = [{"type": "text", "text": answer}]

        await save_checkpoint_func(
            session_id=session_id,
            query=query,
            answer=answer,
            query_content=query_content,
            answer_content=answer_content,
            contexts=contexts,
            cited_chunk_ids=cited_ids,
            workspace=workspace,
            scope=scope,
        )
        return True
    except Exception:
        logger.warning("Checkpoint save failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main SSE stream
# ---------------------------------------------------------------------------


async def stream_answer_events(
    *,
    manager: Any,
    cfg: Any,
    query: str,
    conversation_history: list[dict[str, Any]] | None,
    workspaces: list[str] | None,
    workspace: str,
    query_images: list[dict[str, Any]],
    session_id: str,
    scope: RequestScope | None = None,
) -> AsyncIterator[str]:
    """Yield browser SSE events for one answer request, under a request-root span."""
    ws_list = workspaces or [workspace or manager.config.workspace]
    async with trace_observation(
        "answer_stream_pipeline",
        as_type="chain",
        input={"query": query},
        metadata={
            "workspaces": ws_list,
            "history_turns": len(conversation_history) if conversation_history else 0,
        },
    ):
        async for event in _emit_answer_events(
            manager=manager,
            cfg=cfg,
            query=query,
            conversation_history=conversation_history,
            ws_list=ws_list,
            workspace=workspace,
            query_images=query_images,
            session_id=session_id,
            scope=scope,
        ):
            yield event


async def _emit_answer_events(
    *,
    manager: Any,
    cfg: Any,
    query: str,
    conversation_history: list[dict[str, Any]] | None,
    ws_list: list[str],
    workspace: str,
    query_images: list[dict[str, Any]],
    session_id: str,
    scope: RequestScope | None = None,
) -> AsyncIterator[str]:
    """Emit the SSE event sequence for one answer request."""
    full_answer = ""
    token_iter: AsyncIterator[str] | str | None = None
    try:
        history_kept = len(conversation_history) if conversation_history else 0
        yield sse_event("meta", AnswerMetaEvent(history_kept=history_kept))

        t0 = time.monotonic()
        logger.debug("[SSE] query received: %s", log_safe(query))

        yield sse_event("progress", AnswerProgressEvent(phase="planning"))

        contexts, token_iter = await manager.aanswer_stream(
            query,
            conversation_history=conversation_history,
            workspaces=ws_list,
            query_images=query_images,
            session_id=session_id or None,
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
        current_image_ids = getattr(token_iter, "current_image_ids", []) or []
        image_descriptions = getattr(token_iter, "image_descriptions", []) or []

        # ── Build done payload ─────────────────────────────────────
        effective_workspace = workspace or manager.config.workspace
        payload = await _build_answer_done_payload(
            clean_answer=clean_answer,
            contexts=contexts,
            current_image_ids=current_image_ids,
            image_descriptions=image_descriptions,
            manager=manager,
            session_id=session_id,
            scope=scope,
            cfg=cfg,
            workspace=effective_workspace,
        )

        # ── Save checkpoint (before done event, so checkpoint_saved is accurate) ──
        checkpoint_saved = await _save_checkpoint(
            manager=manager,
            session_id=session_id or "",
            query=query,
            answer=payload.done.answer,
            query_images=query_images,
            current_image_ids=current_image_ids,
            contexts=contexts,
            sources=payload.sources,
            workspace=effective_workspace,
            scope=scope,
        )
        payload.done.checkpoint_saved = checkpoint_saved
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
            resolver = SourceUrlResolver(input_dir=str(cfg.input_dir_path))
            highlighted_payloads = project_source_payloads(
                highlighted_sources,
                resolver=resolver,
            )
            yield sse_event("highlights", safe_source_panel(sources=highlighted_payloads))

    except Exception:
        logger.exception("Answer streaming failed")
        yield sse_event("error", AnswerErrorEvent(message="Service error. Please try again."))
    finally:
        await aclose_answer_stream(token_iter)


__all__ = ["stream_answer_events"]
