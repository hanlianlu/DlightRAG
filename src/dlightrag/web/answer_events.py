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
from dlightrag.citations.schemas import SourceReference, SourceReferencePayload
from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens
from dlightrag.core.answer.errors import (
    AnswerInputError,
    InvalidToolConfigurationError,
    classify_answer_error,
)
from dlightrag.core.answer.highlights import enrich_semantic_highlights
from dlightrag.core.answer.media import answer_images_from_sources
from dlightrag.core.answer.turn import PreparedAnswerTurn
from dlightrag.core.answer_runs.snapshots import dump_answer_snapshot, load_answer_snapshot
from dlightrag.core.client_payloads import project_source_payloads
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.core.servicemanager import answer_trace_output
from dlightrag.observability import trace_observation, trace_sensitive_enabled
from dlightrag.storage.web_conversations import CommitTurnResult
from dlightrag.utils import log_safe
from dlightrag.web.attachment_models import ValidatedWebAttachment
from dlightrag.web.conversation_models import ConversationSummary
from dlightrag.web.conversations import (
    PreparedWebConversation,
    WebConversationService,
    WebConversationUnavailableError,
)
from dlightrag.web.events import (
    AnswerDoneEvent,
    AnswerErrorEvent,
    AnswerProgressEvent,
)
from dlightrag.web.safe_html import safe_answer_done, safe_answer_preview, safe_source_panel
from dlightrag.web.sse import sse_event

if TYPE_CHECKING:
    from dlightrag.core.servicemanager import RAGServiceManager

logger = logging.getLogger(__name__)
_SSE_HEARTBEAT_SECONDS = 10.0


def _capability_metrics(manager: RAGServiceManager) -> dict[str, Any]:
    """Resolver/selection/capability metrics known before generation."""
    capability = manager.answer_image_capability
    return {
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
    """Transport metrics derived from the final assembled answer messages."""
    return {
        "answer_images_current": int(trace.get("answer_images_current", 0) or 0),
        "answer_images_rag": int(trace.get("answer_images_rag", 0) or 0),
        "answer_images_total": int(trace.get("answer_images_total", 0) or 0),
        "answer_image_bytes_total": int(trace.get("answer_image_budget_used_bytes", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _AnswerPayload:
    done: AnswerDoneEvent
    sources: list[SourceReferencePayload]
    internal_sources: list[SourceReference]
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


def _done_from_committed_turn(
    commit: CommitTurnResult,
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
) -> AnswerDoneEvent:
    answer = commit.assistant_text or ""
    internal_sources = load_answer_snapshot(commit.answer_sources or {"sources": []})
    sources = project_source_payloads(
        internal_sources,
        resolver=SourceDownloadLinkBuilder(base_url="/web/files/raw"),
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    answer_images = answer_images_from_sources(
        internal_sources,
        visual_workspaces=visual_workspaces,
    )
    summary = _conversation_summary(commit.summary)
    return AnswerDoneEvent(
        html=safe_answer_done(answer=answer, sources=sources, answer_images=answer_images),
        answer=answer,
        answer_images=answer_images,
        conversation_saved=True,
        conversation=summary,
    )


async def _build_answer_done_payload(
    *,
    clean_answer: str,
    contexts: dict[str, Any],
    manager: RAGServiceManager,
    cfg: Any,
    workspace: str,
    conversation_id: str,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
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
        visual_workspaces=visual_workspaces,
    )
    source_payloads = project_source_payloads(
        finalized.sources,
        resolver=resolver,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    answer_images = cited_images
    done = AnswerDoneEvent(
        html=safe_answer_done(
            answer=finalized.answer,
            sources=source_payloads,
            answer_images=answer_images,
        ),
        answer=finalized.answer,
        answer_images=answer_images,
        conversation_saved=False,
    )
    return _AnswerPayload(done, source_payloads, finalized.sources, finalized.flat_contexts)


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
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    conversation_service: WebConversationService,
    prepared_conversation: PreparedWebConversation,
    validated_attachments: tuple[ValidatedWebAttachment, ...] = (),
    submission_id: str,
) -> AsyncGenerator[str]:
    """Yield browser SSE events for one answer request, under a request-root span.

    The request-root span is opened here, then query planning runs lazily inside
    it (see ``_emit_answer_events``). Planning shares this task and OTEL context,
    so ``retrieval_planning`` nests under ``answer_pipeline`` and the whole
    turn -- plan, retrieve, generate, highlight -- lands in one trace. An
    already-committed submission replays below without planning at all.
    """
    committed = prepared_conversation.committed_submission
    ws_list = (
        list(committed.queried_workspaces)
        if committed is not None and committed.queried_workspaces
        else workspaces or [workspace or manager.config.workspace]
    )
    if trace_sensitive_enabled():
        conversation_ref = prepared_conversation.conversation_id
        identity = {
            "principal_id": prepared_conversation.principal_id,
            "conversation_id": conversation_ref,
        }
    else:
        conversation_ref = hashlib.sha256(
            prepared_conversation.conversation_id.encode("utf-8")
        ).hexdigest()
        identity = {
            "principal_hash": prepared_conversation.principal_id,
            "conversation_hash": conversation_ref,
        }
    metadata = {
        "stream": True,
        "workspaces": ws_list,
        **identity,
        "history_turns_loaded": len(prepared_conversation.text_history) // 2,
        "current_attachment_count": len(validated_attachments),
    }
    async with trace_observation(
        "answer_pipeline",
        as_type="chain",
        input={"query": query},
        metadata=metadata,
        session_id=conversation_ref,
    ) as observation:
        if prepared_conversation.committed_submission is not None:
            done = _done_from_committed_turn(
                prepared_conversation.committed_submission,
                downloadable_workspaces=downloadable_workspaces,
                visual_workspaces=visual_workspaces,
            )
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
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
            conversation_service=conversation_service,
            prepared_conversation=prepared_conversation,
            validated_attachments=validated_attachments,
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
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    conversation_service: WebConversationService,
    prepared_conversation: PreparedWebConversation,
    validated_attachments: tuple[ValidatedWebAttachment, ...] = (),
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
        t0 = time.monotonic()
        logger.debug("[SSE] query received: %s", log_safe(query))

        yield sse_event("progress", AnswerProgressEvent(phase="planning"))

        # Build the turn and its request resources. Current-turn attachments carry
        # inline bytes (the manager extracts verified images into current-image
        # blocks and registers documents); prior attachments are lazy authorized
        # resources. Planning runs inside the manager under this request-root span.
        turn = PreparedAnswerTurn(
            current_query=query,
            text_history=tuple(prepared_conversation.text_history),
        )
        resources = conversation_service.build_answer_resources(
            prepared_conversation,
            validated_attachments,
        )
        if observation is not None:
            observation.update(metadata=_capability_metrics(manager))

        yield sse_event("progress", AnswerProgressEvent(phase="searching"))

        answer_task = asyncio.create_task(
            manager._aanswer_stream_prepared(
                turn,
                workspaces=ws_list,
                resources=resources,
            )
        )
        try:
            while True:
                try:
                    contexts, token_iter = await asyncio.wait_for(
                        asyncio.shield(answer_task),
                        timeout=_SSE_HEARTBEAT_SECONDS,
                    )
                    break
                except TimeoutError:
                    yield sse_event("progress", AnswerProgressEvent(phase="searching"))
        finally:
            if token_iter is None:
                if not answer_task.done():
                    answer_task.cancel()
                try:
                    _contexts, unclaimed_stream = await answer_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug(
                        "Unclaimed answer stream setup failed during consumer cancellation",
                        exc_info=True,
                    )
                else:
                    await aclose_answer_stream(unclaimed_stream)
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
            # Preview re-renders and resends the whole answer, so widen the
            # cadence as it grows to keep the streamed cost near O(n) rather
            # than O(n^2) for long answers.
            answer_len = len(accumulated_text)
            min_interval = 0.3 + answer_len / 8000.0
            min_new_chars = max(20, answer_len // 30)
            if now - last_preview_ts > min_interval and new_chars > min_new_chars:
                yield sse_event("preview", safe_answer_preview(accumulated_text))
                last_preview_ts = now
                last_preview_len = len(accumulated_text)

        clean_answer = getattr(token_iter, "answer", None) or full_answer

        # ── Build done payload ─────────────────────────────────────
        effective_workspace = workspace or manager.config.workspace
        payload = await _build_answer_done_payload(
            clean_answer=clean_answer,
            contexts=contexts,
            manager=manager,
            cfg=cfg,
            workspace=effective_workspace,
            conversation_id=prepared_conversation.conversation_id,
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
        )
        answer_sources = dump_answer_snapshot(payload.internal_sources)
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
                    attachments=validated_attachments,
                )
            )
            while True:
                try:
                    commit = await asyncio.wait_for(
                        asyncio.shield(commit_task),
                        timeout=_SSE_HEARTBEAT_SECONDS,
                    )
                    break
                except TimeoutError:
                    yield sse_event("progress", AnswerProgressEvent(phase="saving"))
                except asyncio.CancelledError:
                    cancellation_pending = True
                    commit = await _finish_persistence_task(commit_task)
                    break
            if commit.saved and commit.replayed:
                done = _done_from_committed_turn(
                    commit,
                    downloadable_workspaces=downloadable_workspaces,
                    visual_workspaces=visual_workspaces,
                )
            else:
                summary = _conversation_summary(commit.summary)
                done = payload.done.model_copy(
                    update={
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
                    "conversation_saved": False,
                    "conversation_save_reason": "storage_unavailable",
                }
            )
            save_reason = "storage_unavailable"
        except Exception:
            logger.exception("Conversation persistence failed after answer completion")
            done = payload.done.model_copy(
                update={
                    "conversation_saved": False,
                    "conversation_save_reason": "persistence_failed",
                }
            )
            save_reason = "persistence_failed"

        if cancellation_pending:
            raise asyncio.CancelledError

        yield sse_event("done", done)

        # ── Post-done enrichment (trace, highlights) ───────────────
        try:
            trace = getattr(token_iter, "trace", None)
            if observation is not None:
                observation.update(
                    output=answer_trace_output(done.answer, payload.sources, contexts),
                    metadata=_answer_transport_metrics(trace if isinstance(trace, dict) else {}),
                )
            highlighted_internal_sources = await enrich_semantic_highlights(
                payload.internal_sources,
                answer_text=done.answer,
                config=cfg,
            )
            highlighted_sources = project_source_payloads(
                highlighted_internal_sources,
                resolver=SourceDownloadLinkBuilder(base_url="/web/files/raw"),
                downloadable_workspaces=downloadable_workspaces,
                visual_workspaces=visual_workspaces,
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
                        answer_sources=dump_answer_snapshot(highlighted_internal_sources),
                    )
        except asyncio.CancelledError, GeneratorExit:
            raise
        except Exception:
            logger.exception("Post-done answer enrichment failed")

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
        message = (
            exc.public_message
            if isinstance(exc, AnswerInputError | InvalidToolConfigurationError)
            else "Service error. Please try again."
        )
        yield sse_event(
            "error",
            AnswerErrorEvent(message=message, error_kind=error_kind),
        )
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
