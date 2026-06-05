# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for chat interface and answer generation."""

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from dlightrag.citations import CitationProcessor, extract_highlights_for_sources
from dlightrag.citations.source_builder import build_sources
from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.web.deps import get_manager, get_workspace, render_partial, templates
from dlightrag.web.markdown import render_markdown

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, workspace: str = Depends(get_workspace)):
    """Main page."""
    from dlightrag.utils import normalize_workspace

    manager = get_manager(request)
    try:
        workspaces = await manager.list_workspace_records()
    except Exception:
        workspaces = [
            {
                "workspace": workspace,
                "display_name": workspace,
                "embedding_model": manager.config.embedding.model,
            }
        ]

    known = {str(row["workspace"]) for row in workspaces}
    active_raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in active_raw.split(",") if item.strip()]
    active = [item for item in active if item in known]

    primary = normalize_workspace(request.cookies.get("dlightrag_workspace", workspace))
    if not active and primary in known:
        active = [primary]
    if not active:
        active = [
            workspace
            if workspace in known
            else (workspaces[0]["workspace"] if workspaces else "default")
        ]

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "workspace": workspace,
            "workspaces": workspaces,
            "active_workspaces": active,
        },
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

            current_image_ids = getattr(token_iter, "current_image_ids", []) or []
            image_descriptions = getattr(token_iter, "image_descriptions", []) or []

            answer_images = []
            seen_img_ids = set()

            # Session images carry their base64 data inline — use data: URLs
            # because /web/images/ expects chunk IDs from the vector DB, not
            # session-local IDs like img_0/img_1.
            stored_session_images: list[str] = []
            if current_image_ids:
                try:
                    stored_session_images = await manager.get_session_image_data(
                        session_id, current_image_ids
                    )
                except Exception:
                    stored_session_images = []

            for i, cid in enumerate(current_image_ids):
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
                seen_img_ids.add(cid)

            flat_contexts = []
            for items in contexts.values():
                if isinstance(items, list):
                    flat_contexts.extend(items)

            # Extract visual chunks from retrieval contexts for the image gallery.
            # Only include images that were actually sent to the answer LLM (the
            # AnswerContextPacker marks in-budget chunks with _answer_image_sent).
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
                ws = chunk.get("_workspace") or workspace or manager.config.workspace
                if not image_url and image_data:
                    image_url = f"/web/images/{ws}/{cid}?size=full"
                    thumb_url = f"/web/images/{ws}/{cid}?size=thumb"
                label = chunk.get("file_path", "") or f"Visual {len(answer_images) + 1}"
                answer_images.append(
                    {
                        "chunk_id": cid,
                        "url": image_url,
                        "thumb_url": thumb_url,
                        "label": label,
                    }
                )
                seen_img_ids.add(cid)

            resolver = PathResolver(
                input_dir=str(cfg.input_dir_path),
                workspace=workspace or manager.config.workspace,
            )
            sources = build_sources(
                contexts,
                path_resolver=resolver,
                image_url_prefix="/web/images",
                default_workspace=workspace or manager.config.workspace,
            )

            processor = CitationProcessor(
                contexts=flat_contexts,
                available_sources=sources,
            )
            result = processor.process(clean_answer)

            # Only show images from chunks the LLM actually cited, not every
            # chunk that happened to be in the retrieval contexts (or even in
            # the prompt — images sent but not referenced don't belong in the
            # gallery).  Session/user-uploaded images (img_N) always stay.
            all_cited_ids: set[str] = set()
            for cids in result.cited_chunks.values():
                all_cited_ids.update(cids)
            cited_images = [
                img for img in answer_images if img.get("chunk_id", "") in all_cited_ids
            ]
            session_images = [
                img for img in answer_images if str(img.get("chunk_id", "")).startswith("img_")
            ]
            answer_images = session_images + cited_images

            done_html = render_partial(
                "partials/answer_done.html",
                answer=result.answer,
                sources=result.sources,
                answer_images=answer_images,
            )
            done_payload = {
                "html": done_html,
                "answer": result.answer,
                "current_image_ids": getattr(token_iter, "current_image_ids", []),
                "image_descriptions": getattr(token_iter, "image_descriptions", []),
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

            # Save conversation checkpoint for multi-turn persistence
            try:
                cited_ids = []
                for s in result.sources:
                    if s.id:
                        cited_ids.append(s.id)
                await manager.save_turn_checkpoint(
                    session_id=session_id or "",
                    query=query,
                    answer=clean_answer,
                    contexts=contexts,
                    cited_chunk_ids=cited_ids,
                )
            except Exception:
                logger.warning("Checkpoint save failed", exc_info=True)

            trace = getattr(token_iter, "trace", None)
            if isinstance(trace, dict) and trace:
                yield f"event: trace\ndata: {json.dumps(trace)}\n\n"

            highlight_cfg = cfg.citations.highlights
            has_text_chunks = any(
                chunk.content for src in result.sources if src.chunks for chunk in src.chunks
            )
            if highlight_cfg.enabled and has_text_chunks:
                try:
                    from dlightrag.models.llm import get_keyword_model_func

                    llm_func = get_keyword_model_func(cfg)
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
                    highlights_html = render_partial(
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
