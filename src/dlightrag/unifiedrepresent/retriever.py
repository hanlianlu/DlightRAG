# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual retrieval pipeline for unified representational RAG.

Handles query processing: LightRAG KG retrieval -> visual chunk resolution ->
visual reranking -> VLM answer generation.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, cast

import httpx
from lightrag import LightRAG, QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from PIL import Image

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.unifiedrepresent.multimodal_query import enhance_query_with_images

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]

logger = logging.getLogger(__name__)


class VisualRetriever:
    """Query pipeline for unified representational RAG (Phases 2-4)."""

    def __init__(
        self,
        lightrag: LightRAG,
        visual_chunks: Any,  # BaseKVStorage instance
        config: Any,  # DlightragConfig
        vision_model_func: Callable | None = None,
        rerank_model: str | None = None,
        rerank_base_url: str | None = None,
        rerank_api_key: str | None = None,
        rerank_backend: str | None = None,
        path_resolver: PathResolver | None = None,
        embedder: Any | None = None,
        provider: str = "openai",
    ) -> None:
        self.lightrag = lightrag
        self.visual_chunks = visual_chunks
        self.config = config
        self.vision_model_func = vision_model_func
        self.rerank_model = rerank_model
        self.rerank_base_url = rerank_base_url
        self.rerank_api_key = rerank_api_key
        self.rerank_backend = rerank_backend
        self.provider = provider
        self.path_resolver = path_resolver
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        images: list[bytes] | None = None,
        conversation_context: str | None = None,
    ) -> dict:
        """Run Phase 1-3 with optional dual-path for multimodal queries.

        The returned dict always contains an ``enhanced_query`` key holding the
        query string that was actually used for text-path retrieval.  For pure
        text queries this equals *query*; when *images* are supplied it includes
        VLM-generated image descriptions.
        """
        # Dual-path: if images provided, run VLM-enhanced text query + visual embedding
        if images:
            if self.vision_model_func:
                enhanced_query = await enhance_query_with_images(
                    query=query,
                    images=images,
                    vision_model_func=self.vision_model_func,
                    conversation_context=conversation_context,
                )
            else:
                enhanced_query = query
                logger.warning("No vision_model_func; using original query for text path")

            # Text path: standard retrieval with enhanced query
            text_result = await self._text_retrieve(
                enhanced_query,
                mode,
                top_k,
                chunk_top_k,
            )

            # Image path: visual embedding direct query
            visual_chunks = await self.query_by_visual_embedding(images, top_k=chunk_top_k)

            # Round-robin merge text chunks + visual chunks
            text_chunks = text_result.get("contexts", {}).get("chunks", [])
            merged_chunks: list[dict[str, Any]] = []
            max_len = max(len(text_chunks), len(visual_chunks))
            for i in range(max_len):
                if i < len(text_chunks):
                    merged_chunks.append(text_chunks[i])
                if i < len(visual_chunks):
                    merged_chunks.append(visual_chunks[i])

            # Cap at chunk_top_k to maintain consistent result count with text-only queries
            text_result["contexts"]["chunks"] = merged_chunks[:chunk_top_k]
            text_result["enhanced_query"] = enhanced_query
            return text_result

        # Text-only: unchanged existing path
        result = await self._text_retrieve(query, mode, top_k, chunk_top_k)
        result["enhanced_query"] = query
        return result

    async def _text_retrieve(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
    ) -> dict:
        """Run Phase 1-3: LightRAG retrieval -> visual resolution -> reranking.

        Returns dict with keys:
            contexts: {entities, relationships, chunks}
        """
        # Phase 1: LightRAG retrieval
        param = QueryParam(
            mode=cast(QueryMode, mode),
            only_need_context=True,
            top_k=top_k,
            enable_rerank=False,  # We handle reranking ourselves
        )
        result = await self.lightrag.aquery_data(query, param=param)

        # Extract chunk_ids, text, and reference metadata from result
        data = result.get("data", {})
        chunk_text: dict[str, str] = {}
        chunk_meta: dict[str, dict[str, str]] = {}  # chunk_id -> {reference_id, file_path}
        chunk_ids: set[str] = set()

        # From chunks section
        for chunk in data.get("chunks", []):
            cid = chunk.get("chunk_id")
            if cid:
                chunk_ids.add(cid)
                chunk_text[cid] = chunk.get("content", "")
                chunk_meta[cid] = {
                    "reference_id": str(chunk.get("reference_id", "")),
                    "file_path": chunk.get("file_path", ""),
                }

        # From entities source_id
        for entity in data.get("entities", []):
            source_id = entity.get("source_id", "")
            if source_id:
                for cid in source_id.split(GRAPH_FIELD_SEP):
                    cid = cid.strip()
                    if cid:
                        chunk_ids.add(cid)

        # From relationships source_id
        for rel in data.get("relationships", []):
            source_id = rel.get("source_id", "")
            if source_id:
                for cid in source_id.split(GRAPH_FIELD_SEP):
                    cid = cid.strip()
                    if cid:
                        chunk_ids.add(cid)

        # Backfill content and metadata for chunks discovered via entities/relationships.
        # These chunk_ids came from source_id fields and lack text/reference metadata.
        missing_cids = chunk_ids - set(chunk_text.keys())
        if missing_cids:
            # Build reverse map: file_path -> reference_id from known chunks
            path_to_ref: dict[str, str] = {}
            for meta in chunk_meta.values():
                fp = meta.get("file_path", "")
                rid = meta.get("reference_id", "")
                if fp and rid:
                    path_to_ref[fp] = rid

            missing_list = list(missing_cids)
            text_data = await self.lightrag.text_chunks.get_by_ids(missing_list)
            recovered = 0
            for cid, td in zip(missing_list, text_data, strict=False):
                if td is None:
                    continue
                if isinstance(td, str):
                    try:
                        td = json.loads(td)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if isinstance(td, dict):
                    content = td.get("content", "")
                    file_path = td.get("file_path", "")
                    if content:
                        chunk_text[cid] = content
                        recovered += 1
                    if cid not in chunk_meta:
                        chunk_meta[cid] = {
                            "reference_id": path_to_ref.get(file_path, ""),
                            "file_path": file_path,
                        }
            logger.info(
                "[Backfill] Recovered text for %d/%d entity/relationship chunks",
                recovered,
                len(missing_cids),
            )

        # Phase 2: Visual resolution
        chunk_id_list = list(chunk_ids)
        logger.info("[Visual Resolve] Looking up %d chunk_ids in visual_chunks", len(chunk_id_list))
        visual_data = await self.visual_chunks.get_by_ids(chunk_id_list)
        # visual_data is a list (same order as input); filter out None/missing.
        # Some KV backends (e.g., PG) may return JSONB as raw strings — parse them.
        resolved: dict[str, dict] = {}
        for cid, vd in zip(chunk_id_list, visual_data, strict=False):
            if vd is None:
                continue
            if isinstance(vd, str):
                try:
                    vd = json.loads(vd)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping unparseable visual_chunk %s", cid)
                    continue
            if isinstance(vd, dict):
                resolved[cid] = vd
        logger.info("[Visual Resolve] Resolved %d/%d chunks", len(resolved), len(chunk_id_list))

        # Phase 3: Visual reranking (optional)
        if self.rerank_backend == "llm" and self.vision_model_func and resolved:
            resolved = await self._llm_visual_rerank(query, resolved, chunk_top_k, chunk_text)
        elif self.rerank_base_url and self.rerank_model and resolved:
            resolved = await self._visual_rerank(query, resolved, chunk_top_k, chunk_text)
        else:
            resolved = dict(list(resolved.items())[:chunk_top_k])
            logger.info(
                "[Visual Rerank] Skipped (backend=%s, vision=%s, resolved=%d)",
                self.rerank_backend,
                self.vision_model_func is not None,
                len(resolved),
            )

        # Filter low-score chunks after reranking (only affects scored chunks)
        threshold = getattr(self.config, "rerank_score_threshold", 0.5)
        before = len(resolved)
        resolved = {
            cid: vd
            for cid, vd in resolved.items()
            if vd.get("relevance_score") is None or vd["relevance_score"] >= threshold
        }
        filtered = before - len(resolved)
        if filtered:
            logger.info(
                "[Rerank Filter] Removed %d/%d chunks below %.2f threshold",
                filtered,
                before,
                threshold,
            )

        # Build return dict — all data lives in contexts, no separate "raw"
        return {
            "contexts": {
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "chunks": [
                    {
                        "chunk_id": cid,
                        "reference_id": chunk_meta.get(cid, {}).get("reference_id", ""),
                        "file_path": chunk_meta.get(cid, {}).get(
                            "file_path", vd.get("file_path", "")
                        ),
                        "content": chunk_text.get(cid, ""),
                        "image_data": vd.get("image_data"),
                        "page_idx": (vd.get("page_index", 0) or 0) + 1,  # 0-based -> 1-based
                        "relevance_score": vd.get("relevance_score"),
                    }
                    for cid, vd in resolved.items()
                ],
            },
        }

    async def answer(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        images: list[bytes] | None = None,
        conversation_context: str | None = None,
    ) -> dict:
        """Run Phase 1-4: retrieve + VLM answer generation.

        Returns dict with keys: answer, contexts
        """
        retrieval = await self.retrieve(
            query,
            mode,
            top_k,
            chunk_top_k,
            images=images,
            conversation_context=conversation_context,
        )

        if not self.vision_model_func:
            return {"answer": None, "references": [], **retrieval}

        # Phase 4: Build multimodal prompt and call VLM
        from dlightrag.models.llm import provider_supports_structured_vision
        from dlightrag.models.schemas import StructuredAnswer
        from dlightrag.unifiedrepresent.prompts import (
            FREETEXT_REMINDER,
            get_answer_system_prompt,
        )

        contexts = retrieval["contexts"]
        answer_query = retrieval.get("enhanced_query", query)
        structured = provider_supports_structured_vision(self.provider)
        system_prompt = get_answer_system_prompt(structured=structured)

        kg_context = self._format_kg_context(contexts)
        indexer = self._build_citation_indexer(contexts)
        ref_list = indexer.format_reference_list()
        prompt_parts = [
            f"Knowledge Graph Context:\n{kg_context}",
            f"Reference Document List:\n{ref_list}",
        ]
        if conversation_context:
            prompt_parts.append(f"Conversation History:\n{conversation_context}")
        prompt_parts.append(f"Question: {answer_query}")
        if not structured:
            prompt_parts.append(FREETEXT_REMINDER)
        user_prompt = "\n\n".join(prompt_parts)

        messages = self._build_vlm_messages(system_prompt, user_prompt, contexts["chunks"])

        if structured:
            raw = await self.vision_model_func(
                user_prompt,
                messages=messages,
                response_schema=StructuredAnswer,
            )
            try:
                result = StructuredAnswer.model_validate_json(raw)
            except Exception:
                logger.warning("Structured answer parse failed, degrading to raw text")
                result = StructuredAnswer(answer=raw, references=[])
        else:
            answer_text = await self.vision_model_func(user_prompt, messages=messages)
            result = StructuredAnswer(answer=answer_text, references=[])

        return {"answer": result.answer, "references": result.references, **retrieval}

    async def answer_stream(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        images: list[bytes] | None = None,
        conversation_context: str | None = None,
    ) -> tuple[dict, AsyncIterator[str] | None]:
        """Run Phase 1-3 batch + Phase 4 streaming.

        Returns (contexts, token_iterator).
        All data lives in contexts; there is no separate raw dict.
        """
        retrieval = await self.retrieve(
            query,
            mode,
            top_k,
            chunk_top_k,
            images=images,
            conversation_context=conversation_context,
        )

        contexts = retrieval["contexts"]
        answer_query = retrieval.get("enhanced_query", query)

        if not self.vision_model_func:
            return contexts, None

        from dlightrag.models.llm import provider_supports_structured_vision
        from dlightrag.models.schemas import StructuredAnswer
        from dlightrag.models.streaming import AnswerStream, StreamingAnswerParser
        from dlightrag.unifiedrepresent.prompts import (
            FREETEXT_REMINDER,
            get_answer_system_prompt,
        )

        structured = provider_supports_structured_vision(self.provider)
        system_prompt = get_answer_system_prompt(structured=structured)

        kg_context = self._format_kg_context(contexts)
        indexer = self._build_citation_indexer(contexts)
        ref_list = indexer.format_reference_list()
        prompt_parts = [
            f"Knowledge Graph Context:\n{kg_context}",
            f"Reference Document List:\n{ref_list}",
        ]
        if conversation_context:
            prompt_parts.append(f"Conversation History:\n{conversation_context}")
        prompt_parts.append(f"Question: {answer_query}")
        if not structured:
            prompt_parts.append(FREETEXT_REMINDER)
        user_prompt = "\n\n".join(prompt_parts)

        messages = self._build_vlm_messages(system_prompt, user_prompt, contexts["chunks"])
        token_iterator = await self.vision_model_func(
            user_prompt,
            messages=messages,
            stream=True,
            response_schema=StructuredAnswer if structured else None,
        )

        if structured and hasattr(token_iterator, "__aiter__"):
            parser = StreamingAnswerParser()
            token_iterator = AnswerStream(token_iterator, parser)

        return contexts, token_iterator

    async def query_by_visual_embedding(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query chunks_vdb using visual embeddings of uploaded images.

        Each image is embedded via VisualEmbedder and queried against the
        chunks vector database directly. Results from multiple images are
        merged via round-robin interleaving.

        Args:
            images: List of raw image bytes.
            top_k: Number of results per image.

        Returns:
            Round-robin merged list of chunk dicts from all images.
        """
        if not images or self.embedder is None:
            return []

        per_image_results: list[list[dict[str, Any]]] = []
        for img_bytes in images:
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                vec = await self.embedder.embed_pages([pil_img])
                embedding = vec[0].tolist()

                raw_chunks = await self.lightrag.chunks_vdb.query(
                    query="",
                    top_k=top_k,
                    query_embedding=embedding,
                )
                # Normalize VDB results to match text chunk schema
                normalized = [
                    {
                        "chunk_id": c.get("id", ""),
                        "reference_id": "",  # Not available from VDB; set by merge
                        "file_path": c.get("file_path", ""),
                        "content": c.get("content", ""),
                        "page_idx": (c.get("chunk_order_index") or 0) + 1,  # 0→1-based
                        "relevance_score": c.get("distance", 0.0),
                    }
                    for c in (raw_chunks or [])
                ]
                per_image_results.append(normalized)
            except Exception:
                logger.warning("Visual embedding query failed for image", exc_info=True)
                per_image_results.append([])

        # Round-robin merge across images
        merged: list[dict[str, Any]] = []
        max_len = max((len(r) for r in per_image_results), default=0)
        for i in range(max_len):
            for img_results in per_image_results:
                if i < len(img_results):
                    merged.append(img_results[i])
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _llm_visual_rerank(
        self,
        query: str,
        resolved: dict[str, dict],
        top_k: int,
        chunk_text: dict[str, str] | None = None,
    ) -> dict[str, dict]:
        """Rerank visual chunks using VLM pointwise scoring.

        Sends each page image to vision_model_func with a scoring prompt.
        Pages scored 0-1 by VLM, sorted descending.
        Falls back to text prompt when image is unavailable.
        """
        import asyncio

        from dlightrag.models.llm import provider_supports_structured_vision
        from dlightrag.models.schemas import VisualRerankScore
        from dlightrag.unifiedrepresent.prompts import VISUAL_RERANK_PROMPT

        use_schema = provider_supports_structured_vision(self.provider)

        if not resolved or not self.vision_model_func:
            return dict(list(resolved.items())[:top_k])

        vision_model_func = self.vision_model_func
        prompt = VISUAL_RERANK_PROMPT.format(query=query)
        sem = asyncio.Semaphore(4)
        chunk_ids = list(resolved.keys())
        logger.info("[Visual Rerank] Scoring %d chunks (pointwise VLM)", len(chunk_ids))

        async def _score_one(cid: str) -> tuple[str, float]:
            vd = resolved[cid]
            img_data = vd.get("image_data")
            text = chunk_text.get(cid, "") if chunk_text else ""
            if not img_data and not text:
                return cid, 0.0
            async with sem:
                try:
                    if img_data:
                        if use_schema:
                            messages = self._build_vlm_messages(
                                system_prompt="",
                                user_prompt=prompt,
                                chunks=[vd],
                            )
                            resp = await vision_model_func(
                                "",
                                messages=messages,
                                response_schema=VisualRerankScore,
                            )
                        else:
                            img_bytes = base64.b64decode(img_data)
                            resp = await vision_model_func(prompt, image_data=img_bytes)
                    else:
                        text_prompt = f"{prompt}\n\nDocument text:\n{text}"
                        if use_schema:
                            resp = await vision_model_func(
                                "",
                                messages=[{"role": "user", "content": text_prompt}],
                                response_schema=VisualRerankScore,
                            )
                        else:
                            resp = await vision_model_func(text_prompt)
                    score = self._parse_rerank_score(resp)
                    logger.debug("[Visual Rerank] chunk=%s score=%.2f", cid, score)
                    return cid, score
                except Exception:
                    logger.warning("VLM rerank failed for chunk %s", cid, exc_info=True)
                    return cid, 0.0

        results = await asyncio.gather(*[_score_one(cid) for cid in chunk_ids])
        scored = sorted(results, key=lambda x: x[1], reverse=True)

        reranked: dict[str, dict] = {}
        for cid, score in scored[:top_k]:
            resolved[cid]["relevance_score"] = score
            reranked[cid] = resolved[cid]
        scores_str = ", ".join(f"{s:.1f}" for _, s in scored[:top_k])
        logger.info("[Visual Rerank] Top %d scores: [%s]", top_k, scores_str)
        return reranked

    @staticmethod
    def _parse_rerank_score(response: str) -> float:
        """Parse VLM response to a 0-1 relevance score.

        Handles multiple output formats:
        1. Plain float ("0.82")
        2. JSON from structured output ({"score": 0.82})
        3. Free-form text with trailing score (think-mode models)
        """
        if response is None:
            logger.warning("Could not parse rerank score from: %r", response)
            return 0.0

        text = str(response).strip()

        # 1. Plain float
        try:
            return max(0.0, min(1.0, float(text)))
        except (ValueError, TypeError):
            pass

        # Strip chat wrappers from local servers
        text = text.replace("<|im_end|>", "").strip()

        # 2. JSON (e.g. {"score": 0.82} from response_schema)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ("score", "relevance_score"):
                    if key in parsed:
                        return max(0.0, min(1.0, float(parsed[key])))
        except Exception:
            pass

        # 3. Last number in [0, 1] range from free-form text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        for token in reversed(numbers):
            try:
                val = float(token)
            except ValueError:
                continue
            if 0.0 <= val <= 1.0:
                return val

        logger.warning("Could not parse rerank score from: %r", response)
        return 0.0

    async def _visual_rerank(
        self,
        query: str,
        resolved: dict[str, dict],
        top_k: int,
        chunk_text: dict[str, str] | None = None,
    ) -> dict[str, dict]:
        """Rerank resolved visual chunks using multimodal reranker API.

        Calls OpenAI-compatible rerank endpoint with query + page images.
        """
        if not resolved:
            return resolved

        chunk_ids = list(resolved.keys())
        documents: list[dict | str] = []
        for cid in chunk_ids:
            vd = resolved[cid]
            img_data = vd.get("image_data")
            if img_data:
                documents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    }
                )
            else:
                # Fallback to text content if no image
                documents.append(chunk_text.get(cid, "") if chunk_text else "")

        if self.rerank_base_url is None:
            raise RuntimeError("rerank_base_url is required for visual reranking")
        url = f"{self.rerank_base_url.rstrip('/')}/rerank"
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }
        headers: dict[str, str] = {}
        if self.rerank_api_key:
            headers["Authorization"] = f"Bearer {self.rerank_api_key}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            results = resp.json().get("results", [])

            # Sort by relevance_score descending
            results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            reranked: dict[str, dict] = {}
            for item in results[:top_k]:
                idx = item["index"]
                cid = chunk_ids[idx]
                resolved[cid]["relevance_score"] = item.get("relevance_score")
                reranked[cid] = resolved[cid]
            return reranked

        except Exception:
            logger.warning(
                "Visual reranking failed, returning unranked results",
                exc_info=True,
            )
            return dict(list(resolved.items())[:top_k])

    @staticmethod
    def _build_vlm_messages(system_prompt: str, user_prompt: str, chunks: list[dict]) -> list[dict]:
        """Build OpenAI-format multimodal messages with inline base64 images from chunks."""
        content: list[dict] = []
        for item in chunks:
            img_data = item.get("image_data")
            if img_data:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    }
                )
        content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _format_kg_context(self, contexts: dict) -> str:
        """Format KG context (entities + relationships) as text for VLM prompt."""
        parts: list[str] = []

        entities = contexts.get("entities", [])
        if entities:
            parts.append("## Entities")
            for e in entities[:20]:  # Limit to avoid prompt overflow
                name = e.get("entity_name", "")
                etype = e.get("entity_type", "")
                desc = e.get("description", "")
                parts.append(f"- **{name}** ({etype}): {desc}")

        rels = contexts.get("relationships", [])
        if rels:
            parts.append("\n## Relationships")
            for r in rels[:20]:
                src = r.get("src_id", "")
                tgt = r.get("tgt_id", "")
                desc = r.get("description", "")
                parts.append(f"- {src} -> {tgt}: {desc}")

        return "\n".join(parts) if parts else "No knowledge graph context available."

    @staticmethod
    def _build_citation_indexer(contexts: dict) -> CitationIndexer:
        """Build a CitationIndexer from the retrieval contexts dict.

        Flattens all context types (chunks, entities, relationships) into
        a single list, matching the shape expected by
        :meth:`CitationIndexer.build_index`.
        """
        flat: list[dict[str, Any]] = []
        for items in contexts.values():
            if isinstance(items, list):
                flat.extend(items)
        indexer = CitationIndexer()
        indexer.build_index(flat)
        return indexer
