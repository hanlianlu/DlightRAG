"""Semantic highlight extraction — uses LLM to find relevant phrases in chunks.

Ported from sandbox_agent. LangChain replaced with raw async LLM function call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .schemas import SourceReference

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_HIGHLIGHTS = 15
_MAX_INPUT_CHARS = 4096

_HIGHLIGHT_SYSTEM_PROMPT = (
    "You are a precise text analysis assistant. Given a citing sentence and a "
    "chunk of source text, identify 1-3 short phrases (3-12 words each) from "
    "the chunk that most directly support the citing sentence. Return ONLY "
    "phrases that appear verbatim in the chunk text.\n\n"
    'Return JSON: {"phrases": ["phrase1", "phrase2"], "confidence": 0.0-1.0}'
)

_HIGHLIGHT_USER_PROMPT = (
    "Citing sentence: {citing_sentence}\n\n"
    "Source chunk:\n{chunk_content}\n\n"
    "Extract 1-3 supporting phrases from the source chunk (must be exact substrings)."
)


class HighlightPhrases(BaseModel):
    phrases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|(?<=[.!?\u3002\uff01\uff1f])(?=\S)")
_CITATION_RE = re.compile(r"\[\w+-\d+\]")


def extract_all_citing_sentences(answer_text: str) -> dict[str, list[str]]:
    """Extract all sentences that contain each citation."""
    sentences = _SENTENCE_SPLIT_RE.split(answer_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    result: dict[str, list[str]] = {}
    for sentence in sentences:
        for m in _CITATION_RE.finditer(sentence):
            key = m.group(0)[1:-1]
            result.setdefault(key, [])
            if sentence not in result[key]:
                result[key].append(sentence)

    return result


class HighlightExtractor:
    """Extract semantic highlight phrases using LLM with LRU cache."""

    def __init__(
        self,
        llm_func: Callable[..., Awaitable[str]],
        cache_size: int = 500,
    ) -> None:
        self._llm_func = llm_func
        self._cache: OrderedDict[str, HighlightPhrases] = OrderedDict()
        self._cache_size = cache_size

    def _cache_key(self, citing_sentence: str, chunk_id: str) -> str:
        raw = f"{citing_sentence}||{chunk_id}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> HighlightPhrases | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: HighlightPhrases) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    async def extract_highlights(
        self,
        citing_sentence: str,
        chunk_content: str,
        chunk_id: str,
    ) -> HighlightPhrases:
        cache_key = self._cache_key(citing_sentence, chunk_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        citing_sentence = citing_sentence[:_MAX_INPUT_CHARS]
        chunk_content = chunk_content[:_MAX_INPUT_CHARS]

        prompt = (
            f"System: {_HIGHLIGHT_SYSTEM_PROMPT}\n\n"
            + _HIGHLIGHT_USER_PROMPT.format(
                citing_sentence=citing_sentence,
                chunk_content=chunk_content,
            )
        )

        try:
            raw_response = await self._llm_func(prompt)
            text = raw_response.strip()
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            parsed = json.loads(text)
            result = HighlightPhrases(**parsed)
        except Exception:
            logger.debug(
                "Highlight extraction failed for chunk %s", chunk_id, exc_info=True
            )
            result = HighlightPhrases()

        chunk_lower = chunk_content.lower()
        validated = []
        for phrase in result.phrases:
            words = phrase.split()
            if len(words) < 1 or len(words) > 25:
                continue
            if phrase in chunk_content:
                validated.append(phrase)
            elif phrase.lower() in chunk_lower:
                idx = chunk_lower.index(phrase.lower())
                validated.append(chunk_content[idx : idx + len(phrase)])
        result.phrases = validated

        self._cache_put(cache_key, result)
        return result


async def extract_highlights_for_sources(
    sources: list[SourceReference],
    answer_text: str,
    llm_func: Callable[..., Awaitable[str]],
) -> list[SourceReference]:
    """Extract highlights for all sources in parallel."""
    extractor = HighlightExtractor(llm_func=llm_func)
    citing_sentences = extract_all_citing_sentences(answer_text)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_HIGHLIGHTS)

    async def _extract_one(
        chunk_id: str, chunk_content: str, sentence: str
    ) -> tuple[str, HighlightPhrases]:
        async with semaphore:
            return chunk_id, await extractor.extract_highlights(
                sentence, chunk_content, chunk_id
            )

    tasks = []
    for src in sources:
        if not src.chunks:
            continue
        for chunk in src.chunks:
            key = f"{src.id}-{chunk.chunk_idx}"
            for sentence in citing_sentences.get(key, []):
                tasks.append(_extract_one(chunk.chunk_id, chunk.content, sentence))

    if not tasks:
        return sources

    results = await asyncio.gather(*tasks, return_exceptions=True)

    chunk_phrases: dict[str, set[str]] = {}
    for res in results:
        if isinstance(res, Exception):
            logger.debug("Highlight task failed: %s", res)
            continue
        chunk_id, hp = res
        if hp.phrases:
            chunk_phrases.setdefault(chunk_id, set()).update(hp.phrases)

    for src in sources:
        if not src.chunks:
            continue
        for chunk in src.chunks:
            phrases = chunk_phrases.get(chunk.chunk_id)
            if phrases:
                chunk.highlight_phrases = list(phrases)

    return sources
