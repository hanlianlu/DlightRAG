# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-context packing after retrieval and rerank."""

from dataclasses import dataclass, field
from typing import Any

from dlightrag_rag.retrieval import ContextRow

from dlightrag.citations.utils import context_chunk_key, split_source_ids
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import RetrievalContexts


@dataclass
class PackedAnswerContext:
    """Contexts and image blocks that are actually sent to the answer model."""

    contexts: RetrievalContexts
    image_blocks_by_context_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


class AnswerContextPacker:
    """Pack reranked retrieval contexts into answer-model prompt material.

    Retrieval can return more visual chunks than the answer model image budget
    can carry. Pure visual chunks whose image cannot be sent are removed from
    the answer context; mixed text+image chunks keep their text. This keeps
    citation indexes aligned with what the answer model actually saw.
    """

    def pack(
        self,
        contexts: RetrievalContexts,
        *,
        image_budget: AnswerImageBudget,
    ) -> PackedAnswerContext:
        chunks = contexts.get("chunks", [])
        image_blocks: dict[str, dict[str, Any]] = {}
        images_sent = 0
        images_skipped = 0

        packed_chunks: list[ContextRow] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            chunk_key = context_chunk_key(chunk_id, workspace=chunk.get("_workspace"))
            content = str(chunk.get("content") or "").strip()
            image_data = chunk.get("image_data")
            image_block: dict[str, Any] | None = None
            if image_data:
                image_block = image_budget.add_base64(
                    str(image_data),
                    label=chunk_id or str(chunk.get("file_path") or "chunk_image"),
                )
                if image_block is not None:
                    images_sent += 1
                else:
                    images_skipped += 1

            if content or image_block is not None:
                packed_chunk = dict(chunk)
                if image_block is not None and chunk_key:
                    packed_chunk["_answer_image_sent"] = True
                    image_blocks[chunk_key] = image_block
                elif image_data:
                    packed_chunk["_answer_image_sent"] = False
                packed_chunks.append(packed_chunk)

        included_chunk_ids = {
            context_chunk_key(c.get("chunk_id"), workspace=c.get("_workspace"))
            for c in packed_chunks
            if c.get("chunk_id")
        }
        packed_contexts: RetrievalContexts = {
            key: [dict(item) for item in value]
            for key, value in contexts.items()
            if key not in {"chunks", "entities", "relationships"}
        }
        packed_contexts.update(
            {
                "chunks": packed_chunks,
                "entities": _filter_by_source_ids(
                    contexts.get("entities", []),
                    included_chunk_ids,
                ),
                "relationships": _filter_by_source_ids(
                    contexts.get("relationships", []),
                    included_chunk_ids,
                ),
            }
        )
        trace = {
            "answer_context_input_chunks": len(chunks),
            "answer_context_chunks": len(packed_chunks),
            "answer_context_images_sent": images_sent,
            "answer_context_images_skipped": images_skipped,
        }
        return PackedAnswerContext(
            contexts=packed_contexts,
            image_blocks_by_context_key=image_blocks,
            trace=trace,
        )


def _filter_by_source_ids(
    items: list[ContextRow],
    included_chunk_ids: set[str],
) -> list[ContextRow]:
    """Keep KG items sourced by chunks included in the final answer context."""
    if not items:
        return []
    filtered: list[ContextRow] = []
    for item in items:
        source_ids = split_source_ids(item.get("source_id"))
        if any(
            context_chunk_key(source, workspace=item.get("_workspace")) in included_chunk_ids
            for source in source_ids
        ):
            filtered.append(item)
    return filtered


__all__ = ["AnswerContextPacker", "PackedAnswerContext"]
