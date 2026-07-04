# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-context packing after retrieval and rerank."""

from dataclasses import dataclass, field
from typing import Any

from dlightrag.citations.utils import split_source_ids
from dlightrag.core.answer_images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import ContextRow, RetrievalContexts


@dataclass
class PackedAnswerContext:
    """Contexts and image blocks that are actually sent to the answer model."""

    contexts: RetrievalContexts
    image_blocks_by_chunk_id: dict[str, dict[str, Any]] = field(default_factory=dict)
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
        context_top_k: int | None = None,
    ) -> PackedAnswerContext:
        chunks = contexts.get("chunks", [])
        target_chunks = context_top_k if context_top_k and context_top_k > 0 else None
        packed_chunks: list[ContextRow] = []
        image_blocks: dict[str, dict[str, Any]] = {}
        skipped_images = 0
        sent_images = 0

        for chunk in chunks:
            if target_chunks is not None and len(packed_chunks) >= target_chunks:
                break

            chunk_id = str(chunk.get("chunk_id") or "")
            content = str(chunk.get("content") or "").strip()
            image_data = chunk.get("image_data")
            image_block: dict[str, Any] | None = None
            if image_data:
                image_block = image_budget.add_base64(
                    str(image_data),
                    label=chunk_id or str(chunk.get("file_path") or "chunk_image"),
                )
                if image_block is not None:
                    sent_images += 1
                else:
                    skipped_images += 1

            if content or image_block is not None:
                packed_chunk = dict(chunk)
                if image_block is not None and chunk_id:
                    packed_chunk["_answer_image_sent"] = True
                    image_blocks[chunk_id] = image_block
                elif image_data:
                    packed_chunk["_answer_image_sent"] = False
                packed_chunks.append(packed_chunk)

        included_chunk_ids = {str(c.get("chunk_id")) for c in packed_chunks if c.get("chunk_id")}
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
            "answer_context_candidate_chunks": len(chunks),
            "answer_context_target_chunks": target_chunks,
            "answer_context_chunks": len(packed_chunks),
            "answer_context_images_sent": sent_images,
            "answer_context_images_skipped": skipped_images,
        }
        return PackedAnswerContext(
            contexts=packed_contexts,
            image_blocks_by_chunk_id=image_blocks,
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
        if any(source in included_chunk_ids for source in source_ids):
            filtered.append(item)
    return filtered


__all__ = ["AnswerContextPacker", "PackedAnswerContext"]
