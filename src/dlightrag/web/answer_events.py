# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-facing answer stream presenter for the web UI."""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from dlightrag.citations import finalize_answer
from dlightrag.core.answer_highlights import enrich_semantic_highlights
from dlightrag.core.answer_media import answer_blocks_from_markdown, answer_images_from_sources
from dlightrag.core.retrieval.source_url_resolver import SourceUrlResolver
from dlightrag.core.scope import RequestScope
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
            stored_session_images = await manager.get_session_image_data(
                session_id,
                image_ids,
                scope=scope,
            )
        except Exception:
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
    """Yield browser SSE events for one answer request."""
    full_answer = ""
    try:
        history_kept = len(conversation_history) if conversation_history else 0
        yield sse_event("meta", AnswerMetaEvent(history_kept=history_kept))

        t0 = time.monotonic()
        logger.info("[SSE] query received: %s", query[:80])

        yield sse_event("progress", AnswerProgressEvent(phase="planning"))

        ws_list = workspaces or [workspace or manager.config.workspace]
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

        accumulated_text = ""
        last_preview_ts = 0.0
        last_preview_len = 0

        if token_iter is None:
            pass
        elif isinstance(token_iter, str):
            full_answer = token_iter
            accumulated_text = token_iter
            yield sse_event("token", token_iter)
        else:
            async for chunk in token_iter:
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
            source_url_resolver=resolver,
            image_url_prefix="/web/images",
            default_workspace=workspace or manager.config.workspace,
        )
        cited_images = answer_images_from_sources(
            finalized.sources,
            contexts={"chunks": finalized.flat_contexts},
        )
        answer_images = session_cards + cited_images
        answer_blocks = answer_blocks_from_markdown(finalized.answer, cited_images)

        done_payload = AnswerDoneEvent(
            html=safe_answer_done(
                answer=finalized.answer,
                sources=finalized.sources,
                answer_images=answer_images,
            ),
            answer=finalized.answer,
            current_image_ids=current_image_ids,
            image_descriptions=image_descriptions,
            answer_images=answer_images,
            answer_blocks=answer_blocks,
        )
        yield sse_event("done", done_payload)

        save_checkpoint = getattr(manager, "save_turn_checkpoint", None)
        if callable(save_checkpoint):
            save_checkpoint_func = cast(Callable[..., Awaitable[None]], save_checkpoint)
            try:
                cited_ids = [s.id for s in finalized.sources if s.id]

                # Build multimodal query_content / answer_content for
                # future round-trip restoration.  Store image REFERENCES
                # (dlightrag-image://) not base64 payloads.
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
                if finalized.answer:
                    answer_content = [{"type": "text", "text": finalized.answer}]

                await save_checkpoint_func(
                    session_id=session_id or "",
                    query=query,
                    answer=finalized.answer,
                    query_content=query_content,
                    answer_content=answer_content,
                    contexts=contexts,
                    cited_chunk_ids=cited_ids,
                    workspace=workspace or manager.config.workspace,
                    scope=scope,
                )
            except Exception:
                logger.warning("Checkpoint save failed", exc_info=True)
        else:
            logger.debug("Manager does not expose save_turn_checkpoint; skipping checkpoint")

        trace = getattr(token_iter, "trace", None)
        if isinstance(trace, dict) and trace:
            yield sse_event("trace", AnswerTraceEvent(trace=trace))

        highlighted_sources = await enrich_semantic_highlights(
            finalized.sources,
            answer_text=finalized.answer,
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


__all__ = ["stream_answer_events"]
