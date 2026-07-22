# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Principal-scoped service adapter for durable Web conversations."""

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

import asyncpg

from dlightrag.api.auth import UserContext
from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.core.answer.errors import CurrentDocumentParseError
from dlightrag.core.answer.turn import (
    HISTORICAL_DOCUMENT_LOAD_FAILED,
    HISTORICAL_DOCUMENT_PARSE_FAILED,
    HISTORICAL_DOCUMENT_UNAVAILABLE,
    DocumentWarning,
    PreparedAnswerTurn,
)
from dlightrag.core.request.attachment_digest import build_attachment_planner_digests
from dlightrag.core.request.attachments import (
    AttachmentCacheKey,
    AttachmentContextChunk,
    AttachmentRequestVector,
    ParsedAttachmentBundle,
)
from dlightrag.sourcing.source_contract import safe_source_filename
from dlightrag.storage.pool import POSTGRES_UNAVAILABLE_EXCEPTIONS
from dlightrag.storage.web_conversations import (
    CommitTurnResult,
    ConversationSnapshot,
    PendingConversationAttachment,
    PendingConversationImage,
    PGWebConversationStore,
    StoredConversationAttachment,
    StoredConversationImage,
)
from dlightrag.utils.images import (
    image_bytes_to_data_uri,
    thumbnail_bytes,
)
from dlightrag.web.attachment_models import ValidatedWebDocument, ValidatedWebImage
from dlightrag.web.conversation_models import (
    ConversationDocumentReference,
    ConversationHistory,
    ConversationImageReference,
    ConversationSummary,
    ConversationTurn,
)
from dlightrag.web.principal import principal_id_from_user
from dlightrag.web.safe_html import safe_answer_done

if TYPE_CHECKING:
    from dlightrag.core.service import ComposerProcessingResources
    from dlightrag.core.servicemanager import RAGServiceManager

logger = logging.getLogger(__name__)
T = TypeVar("T")
_COMMIT_ATTEMPT_TIMEOUT_SECONDS = 45.0
_RECONCILE_ATTEMPT_TIMEOUT_SECONDS = 10.0
_RECONCILE_ATTEMPTS = 2
_RECONCILE_BACKOFF_SECONDS = 0.25
_AMBIGUOUS_COMMIT_EXCEPTIONS = (
    *POSTGRES_UNAVAILABLE_EXCEPTIONS,
    asyncpg.exceptions.InterfaceError,
)
_HISTORY_THUMBNAIL_MAX_PX = 320
_HISTORY_THUMBNAIL_MAX_BYTES = 128 * 1024
_HISTORY_THUMBNAIL_QUALITY = 82
_HISTORY_THUMBNAIL_MIN_QUALITY = 50
_HISTORY_THUMBNAIL_MIN_PX = 64
_MAX_HISTORY_ATTACHMENTS = 3


class WebConversationUnavailableError(RuntimeError):
    """Raised when durable Web conversation storage cannot be reached."""

    detail = "Web conversation storage is unavailable"


@dataclass(frozen=True, slots=True)
class PreparedWebConversation:
    principal_id: str
    conversation_id: str
    content_revision: int
    text_history: tuple[dict[str, Any], ...]
    committed_submission: CommitTurnResult | None = None
    # Caption-only catalog of prior Composer documents for the Web planner to
    # scope which history documents a follow-up refers to (never bytes).
    attachment_catalog: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class _DocumentParseFailure:
    attachment_id: str
    filename: str
    error_type: str


class WebConversationService:
    """Map authenticated browser operations onto the scoped persistence store."""

    def __init__(
        self,
        *,
        store: PGWebConversationStore,
        max_turns: int,
        ttl_days: int,
    ) -> None:
        self._store = store
        self._max_turns = max_turns
        self._ttl_days = ttl_days

    async def initialize(self) -> None:
        """Apply schema migrations and best-effort global startup retention."""
        await self._store.initialize()
        try:
            await self._store.prune_expired(ttl_days=self._ttl_days)
        except Exception:
            logger.exception("Failed to prune expired Web conversations at startup")

    async def create(self, user: UserContext | None) -> ConversationSummary:
        principal_id = principal_id_from_user(user)
        row = await self._store_call(self._store.create_conversation(principal_id))
        return _conversation_summary(row)

    async def list(self, user: UserContext | None) -> list[ConversationSummary]:
        principal_id = principal_id_from_user(user)
        rows = await self._store_call(
            self._store.list_conversations(
                principal_id,
                ttl_days=self._ttl_days,
            )
        )
        return [_conversation_summary(row) for row in rows]

    async def history(
        self,
        user: UserContext | None,
        conversation_id: str,
    ) -> ConversationHistory | None:
        principal_id = principal_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None

        return ConversationHistory(
            conversation=_snapshot_summary(snapshot),
            turns=[_conversation_turn(conversation_id, row) for row in snapshot.history],
        )

    async def rename(
        self,
        user: UserContext | None,
        conversation_id: str,
        title: str,
    ) -> ConversationSummary | None:
        principal_id = principal_id_from_user(user)
        row = await self._store_call(
            self._store.rename_conversation(
                principal_id,
                conversation_id,
                title=title,
                ttl_days=self._ttl_days,
            )
        )
        return _conversation_summary(row) if row is not None else None

    async def delete(
        self,
        user: UserContext | None,
        conversation_id: str,
    ) -> bool:
        principal_id = principal_id_from_user(user)
        return await self._store_call(
            self._store.delete_conversation(
                principal_id,
                conversation_id,
                ttl_days=self._ttl_days,
            )
        )

    async def prepare_answer(
        self,
        user: UserContext | None,
        conversation_id: str,
        submission_id: str | None = None,
    ) -> PreparedWebConversation | None:
        principal_id = principal_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None
        text_history: list[dict[str, Any]] = []
        attachment_catalog: list[dict[str, Any]] = []
        for turn in snapshot.history:
            text_history.extend(
                (
                    {"role": "user", "content": str(turn["user_text"])},
                    {"role": "assistant", "content": str(turn["assistant_text"])},
                )
            )
            for attachment in turn.get("attachments") or []:
                attachment_catalog.append(
                    {
                        "attachment_id": str(attachment["attachment_id"]),
                        "turn_number": turn.get("turn_number"),
                        "ordinal": attachment.get("ordinal"),
                        "filename": attachment.get("filename"),
                        "parse_summary": attachment.get("parse_summary") or "",
                    }
                )
        committed_submission = None
        if submission_id is not None:
            try:
                async with asyncio.timeout(_RECONCILE_ATTEMPT_TIMEOUT_SECONDS):
                    committed_submission = await self._store.find_committed_turn(
                        principal_id,
                        conversation_id,
                        submission_id,
                        ttl_days=self._ttl_days,
                        retry=False,
                    )
            except _AMBIGUOUS_COMMIT_EXCEPTIONS as exc:
                raise WebConversationUnavailableError from exc
        return PreparedWebConversation(
            principal_id=principal_id,
            conversation_id=snapshot.conversation_id,
            content_revision=snapshot.content_revision,
            text_history=tuple(text_history),
            committed_submission=committed_submission,
            attachment_catalog=tuple(attachment_catalog),
        )

    async def prepare_answer_turn(
        self,
        *,
        manager: RAGServiceManager,
        prepared: PreparedWebConversation,
        query: str,
        current_images: list[dict[str, Any]],
        current_documents: list[ValidatedWebDocument] | None = None,
        workspaces: list[str] | None = None,
    ) -> PreparedAnswerTurn:
        """Plan the turn and materialize referenced history images/documents.

        The Web-variant planner rewrites the query and selects scoped history
        query images and Composer documents in one call; the selected ids are
        re-validated and materialized here (web owns the store), and the
        finished plan is injected so core skips re-planning. Current images
        always come first and are never displaced by history selection.
        Current-turn documents are parsed/chunked/budgeted into plain-dict
        context rows that core merges into retrieval before generation.
        """
        documents = list(current_documents or [])
        resources = await manager.aget_composer_processing_resources(workspaces)
        document_service = (
            self._get_composer_document_service(resources, prepared) if documents else None
        )
        capability = manager.answer_image_capability
        effective = capability.effective_max_images if capability is not None else 0
        remaining = max(0, effective - len(current_images))

        catalog: list[dict[str, Any]] = []
        if remaining > 0:
            catalog = await self._store_call(
                self._store.list_image_catalog(
                    prepared.principal_id,
                    prepared.conversation_id,
                    max_turns=self._max_turns,
                    ttl_days=self._ttl_days,
                )
            )

        document_catalog = list(prepared.attachment_catalog)
        # Parse current-turn documents BEFORE planning so the planner sees their
        # content and can make the standalone query document-aware. Parsing is
        # content-addressed and cached, so this is the only parse of these
        # documents; the resulting chunks are reused for the answer context below.
        (
            current_chunks,
            current_failures,
            current_bundles,
        ) = await self._parse_composer_documents(
            document_service=document_service,
            prepared=prepared,
            documents=documents,
        )
        if current_failures:
            failed = current_failures[0]
            raise CurrentDocumentParseError(failed.filename)
        current_digests, digest_trace = build_attachment_planner_digests(current_bundles)
        if current_bundles:
            logger.info(
                "[AttachmentDigest] documents=%d input_tokens=%d output_tokens=%d "
                "strategy=%s budgets=%s",
                digest_trace["attachment_digest_documents"],
                digest_trace["attachment_digest_input_tokens"],
                digest_trace["attachment_digest_output_tokens"],
                digest_trace["attachment_digest_strategy"],
                digest_trace["attachment_digest_document_budgets"],
            )
        current_document_catalog = [
            {
                "attachment_id": document.attachment_id,
                "filename": document.filename,
                "parse_summary": current_digests.get(document.attachment_id, ""),
            }
            for document in documents
        ]
        described = await manager.adescribe_query_images(current_images)
        plan = await manager.aplan_web_conversation_query(
            query,
            text_history=list(prepared.text_history),
            image_catalog=catalog or None,
            attachment_catalog=document_catalog or None,
            current_attachment_catalog=current_document_catalog or None,
            allowed_history_image_count=remaining,
            allowed_history_attachment_count=_MAX_HISTORY_ATTACHMENTS,
            current_image_descriptions=list(described.values()) or None,
            workspaces=workspaces,
        )

        history_blocks: list[dict[str, Any]] = []
        if plan.selected_history_image_ids:
            owned = await self._store_call(
                self._store.fetch_images_by_ids(
                    prepared.principal_id,
                    prepared.conversation_id,
                    list(plan.selected_history_image_ids),
                    ttl_days=self._ttl_days,
                )
            )
            by_id = {image.image_id: image for image in owned}
            for image_id in plan.selected_history_image_ids:  # preserve relevance order
                image = by_id.get(image_id)
                if image is None:
                    continue
                history_blocks.append(_history_image_block(image))

        # History-image resolution is degraded when the planner selected images
        # that could not all be materialized (fetch/ownership loss); the turn
        # still continues text-only.
        selected_count = len(plan.selected_history_image_ids)
        resolution_status = "degraded" if selected_count > len(history_blocks) else "ok"

        # Resolve the Composer lane independently from workspace RAG. Current
        # uploads and planner-selected historical documents retain distinct
        # scope metadata and are selected against the standalone query only.
        composer_rows: list[dict[str, Any]] = []
        composer_trace: dict[str, Any] = {}
        warning_by_id: dict[str, DocumentWarning] = {}
        history_chunks: list[AttachmentContextChunk] = []
        history_bundles: list[tuple[str, ParsedAttachmentBundle]] = []
        selected_attachment_ids = tuple(plan.selected_history_attachment_ids)
        if selected_attachment_ids:
            catalog_filenames = {
                str(item.get("attachment_id")): safe_source_filename(str(item["filename"]))
                for item in document_catalog
                if item.get("attachment_id") and item.get("filename")
            }
            try:
                fetched_history_docs = await self._store_call(
                    self._store.fetch_documents_by_ids(
                        prepared.principal_id,
                        prepared.conversation_id,
                        list(selected_attachment_ids),
                        ttl_days=self._ttl_days,
                    )
                )
            except WebConversationUnavailableError:
                logger.warning(
                    "[HistoricalDocument] fetch unavailable; continuing without %d documents",
                    len(selected_attachment_ids),
                    exc_info=True,
                )
                fetched_history_docs = []
                for attachment_id in selected_attachment_ids:
                    filename = catalog_filenames.get(attachment_id, "A referenced document")
                    warning_by_id[attachment_id] = DocumentWarning(
                        code=HISTORICAL_DOCUMENT_LOAD_FAILED,
                        filename=filename,
                        message=(
                            f"{filename} could not be loaded for this answer. "
                            "The answer will continue without it."
                        ),
                    )
            history_docs_by_id = {
                document.attachment_id: document for document in fetched_history_docs
            }
            for attachment_id in selected_attachment_ids:
                if attachment_id in history_docs_by_id or attachment_id in warning_by_id:
                    continue
                filename = catalog_filenames.get(attachment_id, "A referenced document")
                warning_by_id[attachment_id] = DocumentWarning(
                    code=HISTORICAL_DOCUMENT_UNAVAILABLE,
                    filename=filename,
                    message=(
                        f"{filename} is no longer available. The answer will continue without it."
                    ),
                )
            history_docs = [
                history_docs_by_id[attachment_id]
                for attachment_id in selected_attachment_ids
                if attachment_id in history_docs_by_id
            ]
            if history_docs and document_service is None:
                document_service = self._get_composer_document_service(resources, prepared)
            (
                history_chunks,
                history_failures,
                history_bundles,
            ) = await self._parse_composer_documents(
                document_service=document_service,
                prepared=prepared,
                documents=history_docs,
            )
            for failure in history_failures:
                warning_by_id.setdefault(
                    failure.attachment_id,
                    DocumentWarning(
                        code=HISTORICAL_DOCUMENT_PARSE_FAILED,
                        filename=failure.filename,
                        message=(
                            f"Could not read {failure.filename}. "
                            "The answer will continue without it."
                        ),
                    ),
                )
        if current_chunks or history_chunks:
            if document_service is None:
                raise RuntimeError("document service missing for parsed Composer chunks")
            retrieval_attachment_ids = {
                attachment_id
                for attachment_id, bundle in [*current_bundles, *history_bundles]
                if bundle.evidence_mode == "retrieval"
            }
            current_rows = _composer_context_rows(current_chunks, scope="current")
            history_rows = _composer_context_rows(history_chunks, scope="history")
            retrieval_rows = [
                row
                for row in [*current_rows, *history_rows]
                if str(row.get("full_doc_id") or "") in retrieval_attachment_ids
            ]
            retrieval_chunks = [
                chunk
                for chunk in [*current_chunks, *history_chunks]
                if chunk.attachment_id in retrieval_attachment_ids
            ]
            request_vectors = _composer_request_vectors(retrieval_chunks)
            dense_rankings, dense_trace = await document_service.adense_rankings(
                plan.standalone_query,
                retrieval_rows,
                request_vectors=request_vectors,
            )
            composer_rows, composer_trace = await manager._aselect_web_composer_evidence(
                query=plan.standalone_query,
                current_rows=current_rows,
                history_rows=history_rows,
                dense_rankings=dense_rankings,
                retrieval_attachment_ids=retrieval_attachment_ids,
                rerank_func=resources.rerank_func,
            )
            composer_rows = _assign_composer_reference_ids(composer_rows)
            composer_trace = {**dense_trace, **composer_trace}
        composer_document_processing = [
            {"attachment_id": attachment_id, **bundle.trace}
            for attachment_id, bundle in [*current_bundles, *history_bundles]
            if bundle.trace
        ]
        if composer_document_processing:
            composer_trace = {
                **composer_trace,
                "composer_document_processing": composer_document_processing,
            }

        return PreparedAnswerTurn(
            current_query=query,
            retrieval_query=query,
            text_history=tuple(prepared.text_history),
            current_query_images=tuple(current_images),
            history_query_images=tuple(history_blocks),
            current_image_descriptions=described,
            plan=plan,
            history_image_catalog_count=len(catalog),
            history_image_resolution_status=resolution_status,
            composer_context_chunks=tuple(composer_rows),
            composer_evidence_trace=composer_trace,
            web_composer_visuals=True,
            current_document_digests=current_digests,
            document_warnings=tuple(
                warning_by_id[attachment_id]
                for attachment_id in selected_attachment_ids
                if attachment_id in warning_by_id
            ),
        )

    def _get_composer_document_service(
        self,
        resources: ComposerProcessingResources,
        prepared: PreparedWebConversation,
    ) -> Any:
        """Construct the Web-only Composer document service with its injected store."""
        from dlightrag.core.request.attachments import ComposerDocumentService

        return ComposerDocumentService(
            lightrag=resources.lightrag,
            store=self._store,
            parser_rules=resources.config.parser.rules,
            ttl_days=self._ttl_days,
            robust_document_embedder=resources.robust_document_embedder,
            direct_image_embedding_enabled=resources.direct_image_embedding_enabled,
            model_bundle=resources.model_bundle,
            config=resources.config,
            principal_id=prepared.principal_id,
            conversation_id=prepared.conversation_id,
        )

    async def _parse_composer_documents(
        self,
        *,
        document_service: Any | None,
        prepared: PreparedWebConversation,
        documents: list[Any],
    ) -> tuple[
        list[AttachmentContextChunk],
        list[_DocumentParseFailure],
        list[tuple[str, ParsedAttachmentBundle]],
    ]:
        """Parse/chunk documents and retain their bundles for planner digests.

        Parsing is content-addressed and cached by the injected store, so a
        document parsed here (current uploads, before planning) is a cache hit if
        looked up again. Returns parsed chunks (in document order), neutral parse
        failures, and bundles that the caller may feed to the deterministic
        planner-digest selector without reparsing. Failures are neutral internal
        results; callers decide whether they are fatal or nonfatal.
        """
        if not documents:
            return [], [], []
        if document_service is None:
            raise RuntimeError("document service is required for document parsing")
        chunks: list[AttachmentContextChunk] = []
        failures: list[_DocumentParseFailure] = []
        bundles: list[tuple[str, ParsedAttachmentBundle]] = []
        for document in documents:
            bundle, meta = await document_service.achunks_for_attachment(
                principal_id=prepared.principal_id,
                conversation_id=prepared.conversation_id,
                attachment_id=document.attachment_id,
                filename=document.filename,
                document_bytes=document.document_bytes,
                content_sha256=document.content_sha256,
            )
            bundle = replace(bundle, trace=dict(meta))
            if error_type := meta.get("attachment_parse_error"):
                failures.append(
                    _DocumentParseFailure(
                        attachment_id=document.attachment_id,
                        filename=safe_source_filename(document.filename),
                        error_type=str(error_type),
                    )
                )
            chunks.extend(bundle.chunks)
            bundles.append((document.attachment_id, bundle))
        return chunks, failures, bundles

    async def image(
        self,
        user: UserContext | None,
        conversation_id: str,
        image_id: str,
    ) -> StoredConversationImage | None:
        principal_id = principal_id_from_user(user)
        return await self._store_call(
            self._store.get_image(
                principal_id,
                conversation_id,
                image_id,
                ttl_days=self._ttl_days,
            )
        )

    async def document(
        self,
        user: UserContext | None,
        conversation_id: str,
        attachment_id: str,
    ) -> StoredConversationAttachment | None:
        principal_id = principal_id_from_user(user)
        return await self._store_call(
            self._store.get_attachment(
                principal_id,
                conversation_id,
                attachment_id,
                ttl_days=self._ttl_days,
            )
        )

    async def thumbnail(
        self,
        user: UserContext | None,
        conversation_id: str,
        image_id: str,
    ) -> StoredConversationImage | None:
        """Derive one bounded UI thumbnail after the scoped original lookup."""
        image = await self.image(user, conversation_id, image_id)
        if image is None:
            return None
        try:
            payload, mime_type = await asyncio.to_thread(
                thumbnail_bytes,
                image.image_bytes,
                max_px=_HISTORY_THUMBNAIL_MAX_PX,
                max_bytes=_HISTORY_THUMBNAIL_MAX_BYTES,
                quality=_HISTORY_THUMBNAIL_QUALITY,
                min_quality=_HISTORY_THUMBNAIL_MIN_QUALITY,
                min_px=_HISTORY_THUMBNAIL_MIN_PX,
            )
        except Exception:
            logger.warning("Failed to derive Web conversation thumbnail", exc_info=True)
            return None
        return StoredConversationImage(
            image_id=image.image_id,
            mime_type=mime_type,
            image_bytes=payload,
        )

    async def commit_answer(
        self,
        prepared: PreparedWebConversation,
        *,
        submission_id: str,
        user_text: str,
        assistant_text: str,
        answer_sources: dict[str, Any],
        queried_workspaces: list[str],
        images: tuple[ValidatedWebImage, ...],
        image_descriptions: dict[str, str],
        documents: tuple[ValidatedWebDocument, ...] = (),
        document_parse_summaries: dict[str, str] | None = None,
    ) -> CommitTurnResult:
        """Atomically append a completed answer against its captured revision."""

        def description_for(image: ValidatedWebImage) -> str | None:
            return image_descriptions.get(image.image_id) or image_descriptions.get(
                str(image.ordinal)
            )

        pending = [
            PendingConversationImage(
                image_id=image.image_id,
                ordinal=image.ordinal,
                mime_type=image.mime_type,
                image_bytes=image.image_bytes,
                content_sha256=image.content_sha256,
                vlm_description=description_for(image),
            )
            for image in images
        ]
        pending_documents = [
            PendingConversationAttachment(
                attachment_id=document.attachment_id,
                ordinal=document.ordinal,
                filename=document.filename,
                mime_type=document.mime_type,
                suffix=document.suffix,
                attachment_bytes=document.document_bytes,
                content_sha256=document.content_sha256,
                parse_summary=(document_parse_summaries or {}).get(document.attachment_id),
            )
            for document in documents
        ]
        try:
            async with asyncio.timeout(_COMMIT_ATTEMPT_TIMEOUT_SECONDS):
                return await self._store.commit_turn(
                    principal_id=prepared.principal_id,
                    conversation_id=prepared.conversation_id,
                    submission_id=submission_id,
                    expected_revision=prepared.content_revision,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    answer_sources=answer_sources,
                    queried_workspaces=queried_workspaces,
                    images=pending,
                    attachments=pending_documents,
                    max_turns=self._max_turns,
                    ttl_days=self._ttl_days,
                )
        except _AMBIGUOUS_COMMIT_EXCEPTIONS:
            return await self._reconcile_commit(prepared, submission_id)

    async def update_answer_highlights(
        self,
        prepared: PreparedWebConversation,
        *,
        submission_id: str,
        answer_sources: dict[str, Any],
    ) -> None:
        """Persist semantic highlights into a committed turn's stored sources.

        Best-effort: highlights are a display enhancement computed after the
        turn is committed, so a failure here must never affect the answer.
        """
        try:
            async with asyncio.timeout(_COMMIT_ATTEMPT_TIMEOUT_SECONDS):
                await self._store.update_turn_sources(
                    principal_id=prepared.principal_id,
                    conversation_id=prepared.conversation_id,
                    submission_id=submission_id,
                    answer_sources=answer_sources,
                )
        except Exception:
            logger.warning("Failed to persist semantic highlights", exc_info=True)

    async def _reconcile_commit(
        self,
        prepared: PreparedWebConversation,
        submission_id: str,
    ) -> CommitTurnResult:
        """Resolve an ambiguous mutation through a short, one-shot lookup budget."""
        for attempt in range(_RECONCILE_ATTEMPTS):
            try:
                async with asyncio.timeout(_RECONCILE_ATTEMPT_TIMEOUT_SECONDS):
                    committed = await self._store.find_committed_turn(
                        prepared.principal_id,
                        prepared.conversation_id,
                        submission_id,
                        ttl_days=self._ttl_days,
                        retry=False,
                    )
            except _AMBIGUOUS_COMMIT_EXCEPTIONS:
                if attempt + 1 < _RECONCILE_ATTEMPTS:
                    await asyncio.sleep(_RECONCILE_BACKOFF_SECONDS)
                continue
            if committed is not None:
                return committed
            return CommitTurnResult(False, "commit_not_found", None, None)
        return CommitTurnResult(False, "commit_outcome_unknown", None, None)

    async def _snapshot(
        self,
        principal_id: str,
        conversation_id: str,
    ) -> ConversationSnapshot | None:
        return await self._store_call(
            self._store.snapshot(
                principal_id,
                conversation_id,
                ttl_days=self._ttl_days,
                max_turns=self._max_turns,
            )
        )

    async def _store_call(self, operation: Awaitable[T]) -> T:
        try:
            return await operation
        except POSTGRES_UNAVAILABLE_EXCEPTIONS as exc:
            raise WebConversationUnavailableError from exc


def _composer_context_rows(
    chunks: list[AttachmentContextChunk],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    """Project parsed chunks into scoped Composer-only evidence rows."""
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        row = chunk.to_context_row()
        row["metadata"] = {
            **(row.get("metadata") or {}),
            "attachment_scope": scope,
            "chunk_index": chunk.chunk_index,
            "sidecar_type": chunk.sidecar_type,
        }
        rows.append(row)
    return rows


def _assign_composer_reference_ids(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy final Composer rows and assign compact ids by document appearance."""
    reference_ids: dict[str, str] = {}
    assigned_rows: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("full_doc_id")
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("Composer row is missing full_doc_id")
        full_doc_id = value
        reference_id = reference_ids.get(full_doc_id)
        if reference_id is None:
            reference_id = f"att-{len(reference_ids) + 1}"
            reference_ids[full_doc_id] = reference_id

        assigned_rows.append({**row, "reference_id": reference_id})
    return assigned_rows


def _composer_request_vectors(
    chunks: list[AttachmentContextChunk],
) -> dict[AttachmentCacheKey, AttachmentRequestVector]:
    """Extract private in-request vectors without adding them to context rows."""
    return {
        chunk.cache_key: AttachmentRequestVector(
            cache_key=chunk.cache_key,
            embedding_signature=chunk.embedding_signature,
            embedding_vector=chunk.embedding_vector,
        )
        for chunk in chunks
        if chunk.cache_key is not None
        and chunk.embedding_signature is not None
        and chunk.embedding_vector is not None
    }


def _history_image_block(image: StoredConversationImage) -> dict[str, Any]:
    data_uri = image_bytes_to_data_uri(image.image_bytes, fallback_mime=image.mime_type)
    return {"type": "image_url", "image_url": {"url": data_uri}}


def _conversation_summary(row: dict[str, Any]) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=str(row["conversation_id"]),
        title=row.get("title"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _snapshot_summary(snapshot: ConversationSnapshot) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=snapshot.conversation_id,
        title=snapshot.title,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
    )


def _answer_snapshot(value: Any) -> tuple[dict[str, Any], list[SourceReferencePayload], list[Any]]:
    raw = dict(value) if isinstance(value, dict) else {}
    sources_value = raw.get("sources", [])
    sources = []
    for source_value in sources_value:
        source = SourceReferencePayload.model_validate(source_value)
        if source.chunks is None:
            source = source.model_copy(update={"chunks": []})
        sources.append(source)
    answer_images = raw.get("answer_images", [])
    if not isinstance(answer_images, list):
        answer_images = []
    raw["sources"] = [source.model_dump(mode="json") for source in sources]
    raw["answer_images"] = answer_images
    return raw, sources, answer_images


def _conversation_turn(conversation_id: str, row: dict[str, Any]) -> ConversationTurn:
    turn_number = int(row["turn_number"])
    images = [
        _image_reference(conversation_id, turn_number, image) for image in row.get("images", [])
    ]
    documents = [
        _document_reference(conversation_id, turn_number, attachment)
        for attachment in row.get("attachments", [])
    ]
    answer_sources, sources, answer_images = _answer_snapshot(row.get("answer_sources"))
    assistant_text = str(row["assistant_text"])
    return ConversationTurn(
        turn_id=str(row["turn_id"]),
        turn_number=turn_number,
        user_text=str(row["user_text"]),
        assistant_text=assistant_text,
        user_images=images,
        user_documents=documents,
        answer_sources=answer_sources,
        answer_html=safe_answer_done(
            answer=assistant_text,
            sources=sources,
            answer_images=answer_images,
        ),
        queried_workspaces=[str(workspace) for workspace in row.get("queried_workspaces", [])],
        created_at=row["created_at"],
    )


def _image_reference(
    conversation_id: str,
    turn_number: int,
    image: dict[str, Any],
) -> ConversationImageReference:
    image_id = str(image["image_id"])
    ordinal = int(image["ordinal"])
    url = f"/web/conversations/{conversation_id}/images/{image_id}"
    return ConversationImageReference(
        image_id=image_id,
        ordinal=ordinal,
        mime_type=str(image["mime_type"]),
        url=url,
        thumbnail_url=url + "/thumbnail",
        label=f"Turn {turn_number}, image {ordinal}",
    )


def _document_reference(
    conversation_id: str,
    turn_number: int,
    attachment: dict[str, Any],
) -> ConversationDocumentReference:
    attachment_id = str(attachment["attachment_id"])
    ordinal = int(attachment["ordinal"])
    parse_summary = attachment.get("parse_summary")
    return ConversationDocumentReference(
        attachment_id=attachment_id,
        ordinal=ordinal,
        filename=str(attachment["filename"]),
        mime_type=str(attachment["mime_type"]),
        byte_size=int(attachment["byte_size"]),
        url=f"/web/conversations/{conversation_id}/documents/{attachment_id}",
        label=f"Turn {turn_number}, document {ordinal}",
        parse_summary=str(parse_summary) if parse_summary is not None else None,
    )


__all__ = [
    "PreparedWebConversation",
    "WebConversationService",
    "WebConversationUnavailableError",
]
