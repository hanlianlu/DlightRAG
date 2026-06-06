# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-facing answer stream presenter for the web UI."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from dlightrag.citations import extract_highlights_for_sources, finalize_answer
from dlightrag.core.retrieval.source_url_resolver import SourceUrlResolver
from dlightrag.core.scope import RequestScope
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
                "chunk_id": cid,
                "url": data_url,
                "thumb_url": data_url,
                "label": desc or f"Visual {i + 1}",
            }
        )
    return answer_images


def _retrieved_image_cards(
    *,
    flat_contexts: list[dict[str, Any]],
    seen_img_ids: set[str],
    workspace: str,
    default_workspace: str,
    existing_count: int,
) -> list[dict[str, Any]]:
    answer_images: list[dict[str, Any]] = []
    for chunk in flat_contexts:
        cid = chunk.get("chunk_id", "")
        if not cid or cid in seen_img_ids:
            continue
        if chunk.get("_answer_image_sent") is False:
            continue
        image_url = chunk.get("image_url")
        thumb_url = chunk.get("thumbnail_url") or image_url
        image_data = chunk.get("image_data")
        if not image_url and not image_data:
            continue
        ws = chunk.get("_workspace") or workspace or default_workspace
        if not image_url and image_data:
            image_url = f"/web/images/{ws}/{cid}?size=full"
            thumb_url = f"/web/images/{ws}/{cid}?size=thumb"
        label = chunk.get("file_path", "") or f"Visual {existing_count + len(answer_images) + 1}"
        answer_images.append(
            {
                "chunk_id": cid,
                "url": image_url,
                "thumb_url": thumb_url,
                "label": label,
            }
        )
        seen_img_ids.add(cid)
    return answer_images


async def stream_answer_events(
    *,
    manager: Any,
    cfg: Any,
    query: str,
    conversation_history: list[dict[str, str]] | None,
    workspaces: list[str] | None,
    workspace: str,
    query_images: list[str],
    session_id: str,
    scope: RequestScope | None = None,
) -> AsyncIterator[str]:
    """Yield browser SSE events for one answer request."""
    full_answer = ""
    try:
        history_kept = len(conversation_history) if conversation_history else 0
        yield sse_event("meta", {"history_kept": history_kept})

        t0 = time.monotonic()
        logger.info("[SSE] query received: %s", query[:80])

        yield sse_event("progress", {"phase": "planning"})

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

        yield sse_event("progress", {"phase": "generating"})
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
        seen_img_ids = {str(img.get("chunk_id", "")) for img in session_cards}

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
        flat_contexts = finalized.flat_contexts
        answer_images = [
            *session_cards,
            *_retrieved_image_cards(
                flat_contexts=flat_contexts,
                seen_img_ids=seen_img_ids,
                workspace=workspace,
                default_workspace=manager.config.workspace,
                existing_count=len(session_cards),
            ),
        ]

        all_cited_ids: set[str] = set()
        for cids in finalized.cited_chunks.values():
            all_cited_ids.update(cids)
        cited_images = [img for img in answer_images if img.get("chunk_id", "") in all_cited_ids]
        session_images = [
            img for img in answer_images if str(img.get("chunk_id", "")).startswith("img_")
        ]
        answer_images = session_images + cited_images

        done_payload = {
            "html": safe_answer_done(
                answer=finalized.answer,
                sources=finalized.sources,
                answer_images=answer_images,
            ),
            "answer": finalized.answer,
            "current_image_ids": current_image_ids,
            "image_descriptions": image_descriptions,
        }
        yield sse_event("done", done_payload)

        save_checkpoint = getattr(manager, "save_turn_checkpoint", None)
        if callable(save_checkpoint):
            save_checkpoint_func = cast(Callable[..., Awaitable[None]], save_checkpoint)
            try:
                cited_ids = [s.id for s in finalized.sources if s.id]
                await save_checkpoint_func(
                    session_id=session_id or "",
                    query=query,
                    answer=finalized.answer,
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
            yield sse_event("trace", trace)

        highlight_cfg = cfg.citations.highlights
        has_text_chunks = any(
            chunk.content for src in finalized.sources if src.chunks for chunk in src.chunks
        )
        if highlight_cfg.enabled and has_text_chunks:
            try:
                from dlightrag.models.llm import get_keyword_model_func

                llm_func = get_keyword_model_func(cfg)
                highlighted_sources = await asyncio.wait_for(
                    extract_highlights_for_sources(
                        sources=finalized.sources,
                        answer_text=finalized.answer,
                        llm_func=llm_func,
                        max_concurrency=highlight_cfg.max_concurrency,
                        max_input_chars=highlight_cfg.max_input_chars,
                        cache_size=highlight_cfg.cache_size,
                    ),
                    timeout=highlight_cfg.timeout,
                )
                yield sse_event("highlights", safe_source_panel(sources=highlighted_sources))
            except TimeoutError:
                logger.warning(
                    "Highlight extraction timed out (%.1fs), skipping",
                    highlight_cfg.timeout,
                )
            except Exception:
                logger.warning("Highlight extraction failed", exc_info=True)

    except Exception:
        logger.exception("Answer streaming failed")
        yield sse_event("error", "Service error. Please try again.")


__all__ = ["stream_answer_events"]
