# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for chat interface and answer generation."""

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from dlightrag.citations import CitationProcessor, extract_highlights_for_sources
from dlightrag.citations.source_builder import build_sources
from dlightrag.web.deps import get_manager, get_workspace, templates
from dlightrag.web.markdown import render_markdown

logger = logging.getLogger(__name__)

router = APIRouter()


def _render_partial(name: str, **ctx: Any) -> str:
    """Render a Jinja2 partial template to string."""
    return templates.env.get_template(name).render(**ctx)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, workspace: str = Depends(get_workspace)):
    """Main page."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"workspace": workspace},
    )


@router.post("/answer")
async def answer_stream(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Stream answer via SSE, then swap in enriched citations."""
    body = await request.json()
    query = str(body.get("query", ""))
    if not query:
        return HTMLResponse("<span>Please enter a question.</span>")

    conversation_history: list[dict[str, str]] | None = body.get("conversation_history")

    # Extract images (base64 from frontend). Raw base64 is passed to the
    # manager; it handles answer budgeting, semantic VLM enhancement, session
    # memory, and direct visual retrieval.
    images_b64: list[str] = body.get("images", [])
    clean_images: list[str] = []
    if images_b64:
        for b64 in images_b64[:3]:  # enforce 3-image limit
            try:
                img_bytes = base64.b64decode(b64)
                if len(img_bytes) > 10 * 1024 * 1024:  # 10MB server-side limit
                    logger.warning("Skipping oversized image (%d bytes)", len(img_bytes))
                    continue
                clean_images.append(b64)
            except Exception:
                logger.warning("Failed to decode uploaded image", exc_info=True)

    # Extract workspaces (multi-select from frontend)
    workspaces: list[str] | None = body.get("workspaces")
    session_id = str(body.get("session_id") or "")

    manager = get_manager(request)
    cfg = manager.config

    async def event_generator() -> AsyncIterator[str]:
        full_answer = ""
        try:
            # Tell frontend how many history messages are in context
            history_kept = len(conversation_history) if conversation_history else 0
            yield f"event: meta\ndata: {json.dumps({'history_kept': history_kept})}\n\n"

            t0 = time.monotonic()
            logger.info("[SSE] query received: %s", query[:80])

            # --- Phase 1: Query planning / retrieval ---
            yield f"event: progress\ndata: {json.dumps({'phase': 'planning'})}\n\n"

            ws_list = workspaces or [workspace or manager.config.workspace]
            contexts, token_iter = await manager.aanswer_stream(
                query,
                conversation_history=conversation_history,
                workspaces=ws_list,
                query_images=clean_images,
                session_id=session_id or None,
            )
            t1 = time.monotonic()
            logger.info("[SSE] retrieval+stream setup done (%.1fs)", t1 - t0)

            yield f"event: progress\ndata: {json.dumps({'phase': 'generating'})}\n\n"
            logger.info("[SSE] stream started")

            accumulated_text = ""
            last_preview_ts = 0.0
            last_preview_len = 0

            if token_iter is None:
                pass
            elif isinstance(token_iter, str):
                full_answer = token_iter
                accumulated_text = token_iter
                yield f"event: token\ndata: {json.dumps(token_iter)}\n\n"
            else:
                async for chunk in token_iter:
                    full_answer += chunk
                    accumulated_text += chunk
                    yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
                    now = time.monotonic()
                    new_chars = len(accumulated_text) - last_preview_len
                    if now - last_preview_ts > 0.3 and new_chars > 20:
                        preview_html = render_markdown(accumulated_text)
                        yield f"event: preview\ndata: {json.dumps(preview_html)}\n\n"
                        last_preview_ts = now
                        last_preview_len = len(accumulated_text)

            clean_answer = getattr(token_iter, "answer", None) or full_answer

            flat_contexts = []
            for items in contexts.values():
                if isinstance(items, list):
                    flat_contexts.extend(items)

            sources = build_sources(
                contexts,
                image_url_prefix="/web/images",
                default_workspace=workspace or manager.config.workspace,
            )

            processor = CitationProcessor(
                contexts=flat_contexts,
                available_sources=sources,
            )
            result = processor.process(clean_answer)

            done_html = _render_partial(
                "partials/answer_done.html",
                answer=result.answer,
                sources=result.sources,
            )
            done_payload = {
                "html": done_html,
                "answer": result.answer,
                "current_image_ids": getattr(token_iter, "current_image_ids", []),
                "image_descriptions": getattr(token_iter, "image_descriptions", []),
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

            trace = getattr(token_iter, "trace", None)
            if isinstance(trace, dict) and trace:
                yield f"event: trace\ndata: {json.dumps(trace)}\n\n"

            highlight_cfg = cfg.citations.highlights
            has_text_chunks = any(
                chunk.content for src in result.sources if src.chunks for chunk in src.chunks
            )
            if highlight_cfg.enabled and has_text_chunks:
                try:
                    from dlightrag.models.llm import get_chat_model_func

                    llm_func = get_chat_model_func(cfg)
                    highlighted_sources = await asyncio.wait_for(
                        extract_highlights_for_sources(
                            sources=result.sources,
                            answer_text=result.answer,
                            llm_func=llm_func,
                            max_concurrency=highlight_cfg.max_concurrency,
                            max_input_chars=highlight_cfg.max_input_chars,
                            cache_size=highlight_cfg.cache_size,
                        ),
                        timeout=highlight_cfg.timeout,
                    )
                    highlights_html = _render_partial(
                        "partials/source_panel.html",
                        sources=highlighted_sources,
                    )
                    yield f"event: highlights\ndata: {json.dumps(highlights_html)}\n\n"
                except TimeoutError:
                    logger.warning(
                        "Highlight extraction timed out (%.1fs), skipping",
                        highlight_cfg.timeout,
                    )
                except Exception:
                    logger.warning("Highlight extraction failed", exc_info=True)

        except Exception:
            logger.exception("Answer streaming failed")
            yield "event: error\ndata: Service error. Please try again.\n\n"
        finally:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
