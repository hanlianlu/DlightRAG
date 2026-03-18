# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual retrieval pipeline for unified representational RAG.

Handles query processing: LightRAG KG retrieval -> visual chunk resolution ->
visual reranking.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Callable
from typing import Any, Literal, cast

import httpx
from lightrag import LightRAG, QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from PIL import Image

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
    ) -> None:
        self.lightrag = lightrag
        self.visual_chunks = visual_chunks
        self.config = config
        self.vision_model_func = vision_model_func
        self.rerank_model = rerank_model
        self.rerank_base_url = rerank_base_url
        self.rerank_api_key = rerank_api_key
        self.rerank_backend = rerank_backend
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
                )
            else:
                enhanced_query = query
                logger.warning("No vision_model_func; using original query for text path")

            # Text path: standard retrieval with enhanced query
            text_result = await self._retrieve(
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
        result = await self._retrieve(query, mode, top_k, chunk_top_k)
        result["enhanced_query"] = query
        return result

    async def _retrieve(
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

        # Phase 2: Visual resolution
        chunk_id_list = list(chunk_ids)
        resolved = await self._resolve_visual_chunks(chunk_id_list)

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

        # Build unified candidate set: visual + text-only
        all_candidates: dict[str, dict] = {}
        for cid, vd in resolved.items():
            all_candidates[cid] = vd
        # Add text-only chunks (not in visual_chunks but have content)
        for cid in chunk_ids:
            if cid not in all_candidates and cid in chunk_text:
                all_candidates[cid] = {}  # No visual data

        # Cap total candidates
        if len(all_candidates) > chunk_top_k:
            # Visual-resolved chunks first (already reranked), then text-only
            ordered_ids: list[str] = list(resolved.keys())
            for cid in chunk_ids:
                if cid not in resolved and cid in all_candidates:
                    ordered_ids.append(cid)
            ordered_ids = ordered_ids[:chunk_top_k]
            all_candidates = {
                cid: all_candidates[cid] for cid in ordered_ids if cid in all_candidates
            }

        # Build return dict
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
                        "page_idx": (vd.get("page_index", 0) or 0) + 1,
                        "relevance_score": vd.get("relevance_score"),
                    }
                    for cid, vd in all_candidates.items()
                ],
            },
        }

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
                # Resolve visual_chunks to get image_data
                raw_ids = [c.get("id", "") for c in (raw_chunks or []) if c.get("id")]
                visual_data = await self._resolve_visual_chunks(raw_ids)

                # Normalize VDB results to match text chunk schema
                normalized = [
                    {
                        "chunk_id": c.get("id", ""),
                        "reference_id": "",  # Not available from VDB; set by merge
                        "file_path": c.get("file_path", ""),
                        "content": c.get("content", ""),
                        "image_data": visual_data.get(c.get("id", ""), {}).get("image_data"),
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

    async def _resolve_visual_chunks(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Look up chunk_ids in visual_chunks KV store, return resolved dict.

        Handles None results, JSONB-as-string parsing, and logs progress.
        """
        if not chunk_ids:
            return {}
        raw = await self.visual_chunks.get_by_ids(chunk_ids)
        resolved: dict[str, dict] = {}
        for cid, vd in zip(chunk_ids, raw, strict=False):
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
        logger.info("[Visual Resolve] Resolved %d/%d chunks", len(resolved), len(chunk_ids))
        return resolved

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

        from dlightrag.unifiedrepresent.prompts import VISUAL_RERANK_PROMPT

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
                    content: list[dict] = []
                    if img_data:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_data}"},
                            }
                        )
                    text_content = prompt if img_data else f"{prompt}\n\nDocument text:\n{text}"
                    content.append({"type": "text", "text": text_content})
                    messages = [{"role": "user", "content": content}]
                    resp = await vision_model_func(
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
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
