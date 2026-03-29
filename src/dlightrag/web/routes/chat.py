# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for chat interface and answer generation."""

import asyncio
import base64
import json
import logging
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from dlightrag.citations import CitationProcessor, extract_highlights_for_sources
from dlightrag.citations.source_builder import build_sources
from dlightrag.config import get_config
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

    cfg = get_config()

    # Extract images (base64 from frontend)
    images_b64: list[str] = body.get("images", [])
    multimodal_content: list[dict[str, Any]] | None = None
    tmp_paths: list[str] = []
    if images_b64:
        multimodal_content = []
        for b64 in images_b64[:3]:  # enforce 3-image limit
            try:
                img_bytes = base64.b64decode(b64)
                if len(img_bytes) > 10 * 1024 * 1024:  # 10MB server-side limit
                    logger.warning("Skipping oversized image (%d bytes)", len(img_bytes))
                    continue
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                tmp_paths.append(tmp.name)
                multimodal_content.append({"type": "image", "img_path": tmp.name})
            except Exception:
                logger.warning("Failed to decode uploaded image", exc_info=True)
        if not multimodal_content:
            multimodal_content = None

    # Extract workspaces (multi-select from frontend)
    workspaces: list[str] | None = body.get("workspaces")

    manager = get_manager(request)

    async def event_generator() -> AsyncIterator[str]:
        full_answer = ""
        try:
            # Tell frontend how many history messages are in context
            history_kept = len(conversation_history) if conversation_history else 0
            yield f"event: meta\ndata: {json.dumps({'history_kept': history_kept})}\n\n"

            t0 = time.monotonic()
            logger.info("[SSE] query received: %s", query[:80])

            # --- Phase 1: Query planning ---
            yield f"event: progress\ndata: {json.dumps({'phase': 'planning'})}\n\n"

            ws_list = workspaces or [workspace or manager._config.workspace]
            planner = manager._get_query_planner()
            plan = await planner.plan(
                query,
                conversation_history=conversation_history,
                max_turns=manager._config.max_conversation_turns,
                max_tokens=manager._config.max_conversation_tokens,
            )
            t1 = time.monotonic()
            logger.info(
                "[SSE] planning done (%.1fs): original=%r, standalone=%r",
                t1 - t0,
                plan.original_query[:60],
                plan.standalone_query[:60],
            )

            # --- Phase 2: Retrieval ---
            yield f"event: progress\ndata: {json.dumps({'phase': 'searching'})}\n\n"

            retrieval = await manager.aretrieve(
                query,
                plan=plan,
                workspaces=ws_list,
                multimodal_content=multimodal_content,
            )
            t2 = time.monotonic()
            logger.info("[SSE] retrieval done (%.1fs)", t2 - t1)

            # --- Phase 3: Answer generation (streaming) ---
            yield f"event: progress\ndata: {json.dumps({'phase': 'generating'})}\n\n"

            engine = manager._get_answer_engine()
            contexts, token_iter = await engine.generate_stream(
                plan.standalone_query,
                retrieval.contexts,
            )
            logger.info("[SSE] stream started (%.1fs since retrieval)", time.monotonic() - t2)

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

            sources = build_sources(contexts)

            processor = CitationProcessor(
                contexts=flat_contexts,
                available_sources=sources,
            )
            result = processor.process(clean_answer)

            if result.sources:
                ref_lines = [f"[{s.id}] {s.title}" for s in result.sources]
                result.answer += "\n\n### References\n" + "\n".join(ref_lines)

            done_html = _render_partial(
                "partials/answer_done.html",
                answer=result.answer,
                sources=result.sources,
            )
            yield f"event: done\ndata: {json.dumps(done_html)}\n\n"

            has_text_chunks = any(
                chunk.content for src in result.sources if src.chunks for chunk in src.chunks
            )
            if has_text_chunks:
                try:
                    from dlightrag.models.llm import get_chat_model_func

                    llm_func = get_chat_model_func(cfg)
                    highlighted_sources = await asyncio.wait_for(
                        extract_highlights_for_sources(
                            sources=result.sources,
                            answer_text=result.answer,
                            llm_func=llm_func,
                        ),
                        timeout=5.0,
                    )
                    highlights_html = _render_partial(
                        "partials/source_panel.html",
                        sources=highlighted_sources,
                    )
                    yield f"event: highlights\ndata: {json.dumps(highlights_html)}\n\n"
                except TimeoutError:
                    logger.warning("Highlight extraction timed out (5s), skipping")
                except Exception:
                    logger.warning("Highlight extraction failed", exc_info=True)

        except Exception:
            logger.exception("Answer streaming failed")
            yield "event: error\ndata: Service error. Please try again.\n\n"
        finally:
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
