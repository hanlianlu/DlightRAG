# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-facing answer stream presenter for the web UI."""

import asyncio
import dataclasses
import hashlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from dlightrag.citations import finalize_answer
from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens
from dlightrag.core.answer_errors import classify_answer_error
from dlightrag.core.answer_highlights import enrich_semantic_highlights
from dlightrag.core.answer_media import answer_blocks_from_markdown, answer_images_from_sources
from dlightrag.core.answer_turn import PreparedAnswerTurn
from dlightrag.core.client_payloads import project_source_payloads
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.core.scope import RequestScope
from dlightrag.observability import trace_observation, trace_sensitive_enabled
from dlightrag.storage.web_conversations import CommitTurnResult
from dlightrag.utils import log_safe
from dlightrag.utils.images import ValidatedWebImage
from dlightrag.web.conversation_models import ConversationSummary
from dlightrag.web.conversations import (
    PreparedWebConversation,
    WebConversationService,
    WebConversationUnavailableError,
)
from dlightrag.web.events import (
    AnswerDoneEvent,
    AnswerErrorEvent,
    AnswerMetaEvent,
    AnswerProgressEvent,
    AnswerTraceEvent,
)
from dlightrag.web.safe_html import safe_answer_done, safe_answer_preview, safe_source_panel
from dlightrag.web.sse import sse_event

if TYPE_CHECKING:
    from dlightrag.core.servicemanager import RAGServiceManager

logger = logging.getLogger(__name__)
_PERSISTENCE_HEARTBEAT_SECONDS = 10.0


def _capability_metrics(manager: RAGServiceManager, turn: PreparedAnswerTurn) -> dict[str, Any]:
    """Resolver/selection/capability metrics known before generation (design §18)."""
    capability = manager.answer_image_capability
    plan = turn.plan
    return {
        "history_image_catalog_count": turn.history_image_catalog_count,
        "history_images_selected": len(plan.selected_history_image_ids) if plan is not None else 0,
        "history_image_resolution_status": turn.history_image_resolution_status,
        "answer_image_capability_status": capability.status
        if capability is not None
        else "unknown",
        "answer_image_configured_ceiling": (
            capability.configured_ceiling if capability is not None else 0
        ),
        "answer_image_effective_limit": (
            capability.effective_max_images if capability is not None else 0
        ),
    }


def _answer_transport_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    """Transport metrics derived from the final assembled answer messages (design §18).

    ``answer_images_total`` is the count of image blocks the assembler placed in
    the final messages, not the RAG-only budget.
    """
    sent = int(trace.get("answer_context_images_sent", 0) or 0)
    skipped = int(trace.get("answer_context_images_skipped", 0) or 0)
    return {
        "answer_images_current": int(trace.get("answer_images_current", 0) or 0),
        "answer_images_history": int(trace.get("answer_images_history", 0) or 0),
        "answer_images_rag": int(trace.get("answer_images_rag", 0) or 0),
        "answer_images_total": int(trace.get("answer_images_total", 0) or 0),
        "answer_image_bytes_total": int(trace.get("answer_image_budget_used_bytes", 0) or 0),
        "rag_visual_descriptions_included": sent + skipped,
        "rag_raw_images_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _AnswerPayload:
    done: AnswerDoneEvent
    sources: list[SourceReferencePayload]
    flat_contexts: list[dict[str, Any]]


def _conversation_summary(value: dict[str, Any] | None) -> ConversationSummary | None:
    if value is None:
        return None
    return ConversationSummary(
        conversation_id=str(value["conversation_id"]),
        title=value.get("title"),
        created_at=value["created_at"],
        updated_at=value["updated_at"],
    )


async def _finish_persistence_task(
    task: asyncio.Task[CommitTurnResult],
) -> CommitTurnResult:
    """Await a bounded persistence task despite caller cancellation."""
    while True:
        if task.cancelled():
            return CommitTurnResult(False, "commit_outcome_unknown", None, None)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                return CommitTurnResult(False, "commit_outcome_unknown", None, None)
            continue


def _done_from_committed_turn(commit: CommitTurnResult) -> AnswerDoneEvent:
    answer = commit.assistant_text or ""
    snapshot = commit.answer_sources or {}
    source_values = snapshot.get("sources", [])
    sources: list[SourceReferencePayload] = []
    for value in source_values:
        source = SourceReferencePayload.model_validate(value)
        if source.chunks is None:
            source = source.model_copy(update={"chunks": []})
        sources.append(source)
    answer_images = snapshot.get("answer_images", [])
    if not isinstance(answer_images, list):
        answer_images = []
    summary = _conversation_summary(commit.summary)
    return AnswerDoneEvent(
        html=safe_answer_done(answer=answer, sources=sources, answer_images=answer_images),
        answer=answer,
        current_image_ids=list(commit.current_image_ids),
        image_descriptions=commit.image_descriptions,
        answer_images=answer_images,
        answer_blocks=answer_blocks_from_markdown(answer, answer_images),
        conversation_saved=True,
        conversation=summary,
    )


async def _build_answer_done_payload(
    *,
    clean_answer: str,
    contexts: dict[str, Any],
    image_descriptions: Any,
    manager: RAGServiceManager,
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
        conversation_saved=False,
    )
    return _AnswerPayload(done, source_payloads, finalized.flat_contexts)


# ---------------------------------------------------------------------------
# Main SSE stream
# ---------------------------------------------------------------------------


async def stream_answer_events(
    *,
    manager: RAGServiceManager,
    cfg: Any,
    query: str,
    workspaces: list[str] | None,
    workspace: str,
    scope: RequestScope | None = None,
    downloadable_workspaces: set[str] | None = None,
    conversation_service: WebConversationService,
    prepared_conversation: PreparedWebConversation,
    validated_images: tuple[ValidatedWebImage, ...],
    submission_id: str,
) -> AsyncGenerator[str]:
    """Yield browser SSE events for one answer request, under a request-root span.

    The request-root span is opened here, then query planning runs lazily inside
    it (see ``_emit_answer_events``). Planning shares this task and OTEL context,
    so ``query_planning`` nests under ``answer_stream_pipeline`` and the whole
    turn -- plan, retrieve, generate, highlight -- lands in one trace. An
    already-committed submission replays below without planning at all.
    """
    ws_list = workspaces or [workspace or manager.config.workspace]
    if trace_sensitive_enabled():
        identity = {
            "principal_id": prepared_conversation.principal_id,
            "conversation_id": prepared_conversation.conversation_id,
        }
    else:
        identity = {
            "principal_hash": prepared_conversation.principal_id,
            "conversation_hash": hashlib.sha256(
                prepared_conversation.conversation_id.encode("utf-8")
            ).hexdigest(),
        }
    metadata = {
        "workspaces": ws_list,
        **identity,
        "history_turns_loaded": len(prepared_conversation.text_history) // 2,
        "current_image_count": len(validated_images),
    }
    async with trace_observation(
        "answer_stream_pipeline",
        as_type="chain",
        input={"query": query},
        metadata=metadata,
    ) as observation:
        if prepared_conversation.committed_submission is not None:
            done = _done_from_committed_turn(prepared_conversation.committed_submission)
            observation.update(
                metadata={
                    "conversation_saved": True,
                    "conversation_save_reason": None,
                    "submission_replayed": True,
                }
            )
            yield sse_event("done", done)
            return
        emitter = _emit_answer_events(
            manager=manager,
            cfg=cfg,
            query=query,
            ws_list=ws_list,
            workspace=workspace,
            scope=scope,
            downloadable_workspaces=downloadable_workspaces,
            conversation_service=conversation_service,
            prepared_conversation=prepared_conversation,
            validated_images=validated_images,
            observation=observation,
            submission_id=submission_id,
        )
        try:
            async for event in emitter:
                yield event
        finally:
            await emitter.aclose()


async def _emit_answer_events(
    *,
    manager: Any,
    cfg: Any,
    query: str,
    ws_list: list[str],
    workspace: str,
    scope: RequestScope | None = None,
    downloadable_workspaces: set[str] | None = None,
    conversation_service: WebConversationService,
    prepared_conversation: PreparedWebConversation,
    validated_images: tuple[ValidatedWebImage, ...],
    observation: Any = None,
    submission_id: str,
) -> AsyncGenerator[str]:
    """Emit the SSE event sequence for one answer request."""
    full_answer = ""
    token_iter: AsyncIterator[str] | str | None = None
    conversation_saved = False
    save_reason: str | None = "answer_incomplete"
    persistence_started = False
    commit_task: asyncio.Task[CommitTurnResult] | None = None
    try:
        history_kept = len(prepared_conversation.text_history)
        yield sse_event("meta", AnswerMetaEvent(history_kept=history_kept))

        t0 = time.monotonic()
        logger.debug("[SSE] query received: %s", log_safe(query))

        yield sse_event("progress", AnswerProgressEvent(phase="planning"))

        # Plan inside the request-root span so query_planning nests under it
        # (same task/OTEL context). The web layer owns the conversation image
        # store, so planner-selected history images are materialized here too.
        turn = await conversation_service.prepare_answer_turn(
            manager=manager,
            prepared=prepared_conversation,
            query=query,
            current_images=[image.model_block for image in validated_images],
            workspaces=ws_list,
        )
        if observation is not None:
            observation.update(metadata=_capability_metrics(manager, turn))

        contexts, token_iter = await manager._aanswer_stream_prepared(
            turn,
            workspaces=ws_list,
            scope=scope,
        )
        t1 = time.monotonic()
        logger.info("[SSE] planning+retrieval+stream setup done (%.1fs)", t1 - t0)

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
        image_descriptions = getattr(token_iter, "image_descriptions", {}) or {}

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
        answer_sources = {
            "sources": [source.model_dump(mode="json") for source in payload.sources],
            "answer_images": payload.done.answer_images,
        }
        cancellation_pending = False
        try:
            persistence_started = True
            commit_task = asyncio.create_task(
                conversation_service.commit_answer(
                    prepared_conversation,
                    submission_id=submission_id,
                    user_text=turn.current_query,
                    assistant_text=payload.done.answer,
                    answer_sources=answer_sources,
                    queried_workspaces=ws_list,
                    images=validated_images,
                    image_descriptions=image_descriptions,
                )
            )
            while True:
                try:
                    commit = await asyncio.wait_for(
                        asyncio.shield(commit_task),
                        timeout=_PERSISTENCE_HEARTBEAT_SECONDS,
                    )
                    break
                except TimeoutError:
                    yield sse_event("progress", AnswerProgressEvent(phase="saving"))
                except asyncio.CancelledError:
                    cancellation_pending = True
                    commit = await _finish_persistence_task(commit_task)
                    break
            if commit.saved and commit.replayed:
                done = _done_from_committed_turn(commit)
            else:
                summary = _conversation_summary(commit.summary)
                done = payload.done.model_copy(
                    update={
                        "current_image_ids": list(commit.current_image_ids) if commit.saved else [],
                        "conversation_saved": commit.saved,
                        "conversation_save_reason": commit.reason,
                        "conversation": summary,
                    }
                )
            conversation_saved = commit.saved
            save_reason = commit.reason
        except WebConversationUnavailableError:
            logger.exception("Conversation storage unavailable after answer completion")
            done = payload.done.model_copy(
                update={
                    "current_image_ids": [],
                    "conversation_saved": False,
                    "conversation_save_reason": "storage_unavailable",
                }
            )
            save_reason = "storage_unavailable"
        except Exception:
            logger.exception("Conversation persistence failed after answer completion")
            done = payload.done.model_copy(
                update={
                    "current_image_ids": [],
                    "conversation_saved": False,
                    "conversation_save_reason": "persistence_failed",
                }
            )
            save_reason = "persistence_failed"

        if cancellation_pending:
            raise asyncio.CancelledError

        yield sse_event("done", done)

        # ── Post-done enrichment (trace, highlights) ───────────────
        trace = getattr(token_iter, "trace", None)
        if isinstance(trace, dict) and trace:
            yield sse_event("trace", AnswerTraceEvent(trace=trace))
        if observation is not None:
            observation.update(
                metadata=_answer_transport_metrics(trace if isinstance(trace, dict) else {})
            )
        highlighted_sources = await enrich_semantic_highlights(
            payload.sources,
            answer_text=done.answer,
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
            if conversation_saved:
                await conversation_service.update_answer_highlights(
                    prepared_conversation,
                    submission_id=submission_id,
                    answer_sources={
                        "sources": [
                            source.model_dump(mode="json") for source in highlighted_sources
                        ],
                        "answer_images": payload.done.answer_images,
                    },
                )

    except asyncio.CancelledError, GeneratorExit:
        if persistence_started and commit_task is not None:
            try:
                commit = await _finish_persistence_task(commit_task)
                conversation_saved = commit.saved
                save_reason = commit.reason
            except WebConversationUnavailableError:
                save_reason = "storage_unavailable"
            except Exception:
                save_reason = "persistence_failed"
        elif not conversation_saved:
            save_reason = "cancelled"
        raise
    except Exception as exc:
        if not conversation_saved:
            save_reason = "answer_failed"
        error_kind = classify_answer_error(exc)
        if observation is not None:
            status = str(exc) if trace_sensitive_enabled() else "answer_stream_failed"
            observation.update(
                level="ERROR",
                status_message=status,
                metadata={"error_kind": error_kind},
            )
        logger.exception("Answer streaming failed")
        yield sse_event("error", AnswerErrorEvent(message="Service error. Please try again."))
    finally:
        if observation is not None:
            observation.update(
                metadata={
                    "conversation_saved": conversation_saved,
                    "conversation_save_reason": save_reason,
                }
            )
        await aclose_answer_stream(token_iter)


__all__ = ["stream_answer_events"]
