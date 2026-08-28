# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-context packing after retrieval and rerank."""

from dataclasses import dataclass, field
from typing import Any

from dlightrag.engine.answer.citations.utils import context_chunk_key
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.rag.retrieval import ContextRow, RetrievalContexts


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
        """Pack images and rows without mutating the retrieved contexts.

        LightRAG mix-mode entities and relationships are independent graph
        context. Chunk image admission must not remove them based on
        ``source_id`` provenance.
        """
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

        packed_contexts: RetrievalContexts = {
            key: [dict(item) for item in value]
            for key, value in contexts.items()
            if key not in {"chunks", "entities", "relationships"}
        }
        entities = contexts.get("entities", [])
        relationships = contexts.get("relationships", [])
        packed_contexts.update(
            {
                "chunks": packed_chunks,
                "entities": [dict(item) for item in entities],
                "relationships": [dict(item) for item in relationships],
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


__all__ = ["AnswerContextPacker", "PackedAnswerContext"]
