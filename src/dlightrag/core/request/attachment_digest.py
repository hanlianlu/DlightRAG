# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Structure-aware planner digests for temporary Web Composer documents."""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any

from dlightrag.core.request.attachments import ParsedAttachmentBundle
from dlightrag.utils.tokens import estimate_tokens, truncate_to_estimated_tokens

ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET = 8_192

_DOCUMENT_FLOOR = 1_536
_ANCHOR_MAX_TOKENS = 1_536
_ANCHOR_RATIO = 0.35
_SAMPLE_TARGET_TOKENS = 384
_MIN_SAMPLE_SLOTS = 4
_MAX_SAMPLE_SLOTS = 12

_JSON_TABLE_RE = re.compile(
    r'<table\b[^>]*\bformat=["\']json["\'][^>]*>(.*?)</table>',
    flags=re.IGNORECASE | re.DOTALL,
)
_FILL_IN_LINE_RE = re.compile(
    r"^(?:answer|name|date|score|答案|姓名|日期|得分)\s*[:：]\s*[_\W]{3,}$",
    flags=re.IGNORECASE,
)
_STRUCTURE_MARKER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+|\[(?:table|image|equation) name\]|\[table\])"
)
_STRUCTURAL_SIDECAR_TYPES = frozenset({"table", "drawing", "image", "figure", "equation"})


def build_attachment_planner_digests(
    documents: list[tuple[str, ParsedAttachmentBundle]],
    *,
    token_budget: int = ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build structure-aware document digests for the Web query planner.

    This consumes normalized chunks already assembled by the selected
    native/MinerU parser and iTeP chunker; it never reparses source files. Short
    documents pass through in full. Longer documents reserve space for existing
    structural signals and sample the remaining chunks across the whole
    document. Multiple documents share one global budget with a per-document
    floor and square-root weighting so one large file cannot starve the rest.
    """
    cleaned: list[tuple[str, ParsedAttachmentBundle, list[str], int]] = []
    for attachment_id, bundle in documents:
        chunks = [_clean_digest_text(chunk.content) for chunk in bundle.chunks]
        chunks = [content for content in chunks if content]
        input_tokens = estimate_tokens("\n\n".join(chunks))
        cleaned.append((attachment_id, bundle, chunks, input_tokens))

    budgets = _allocate_document_digest_budgets(
        [(attachment_id, input_tokens) for attachment_id, _, _, input_tokens in cleaned],
        token_budget=max(0, token_budget),
    )
    digests: dict[str, str] = {}
    output_tokens = 0
    anchor_tokens = 0
    sample_tokens = 0
    sampled = False
    for attachment_id, bundle, chunks, input_tokens in cleaned:
        digest, document_trace = _build_document_planner_digest(
            bundle=bundle,
            cleaned_chunks=chunks,
            input_tokens=input_tokens,
            token_budget=budgets.get(attachment_id, 0),
        )
        digests[attachment_id] = digest
        output_tokens += estimate_tokens(digest)
        anchor_tokens += int(document_trace["anchor_tokens"])
        sample_tokens += int(document_trace["sample_tokens"])
        sampled = sampled or bool(document_trace["sampled"])

    input_tokens = sum(item[3] for item in cleaned)
    return digests, {
        "attachment_digest_strategy": "sampled" if sampled else "full",
        "attachment_digest_documents": len(cleaned),
        "attachment_digest_budget_tokens": max(0, token_budget),
        "attachment_digest_document_budgets": budgets,
        "attachment_digest_input_tokens": input_tokens,
        "attachment_digest_output_tokens": output_tokens,
        "attachment_digest_anchor_tokens": anchor_tokens,
        "attachment_digest_sample_tokens": sample_tokens,
        "attachment_digest_truncated": output_tokens < input_tokens,
    }


def _allocate_document_digest_budgets(
    demands: list[tuple[str, int]], *, token_budget: int
) -> dict[str, int]:
    """Allocate one global digest budget with floors and sqrt weighting."""
    demand_by_id = {attachment_id: max(0, demand) for attachment_id, demand in demands}
    allocations = dict.fromkeys(demand_by_id, 0)
    budget = max(0, token_budget)
    floors = {
        attachment_id: min(demand, _DOCUMENT_FLOOR)
        for attachment_id, demand in demand_by_id.items()
    }
    if sum(floors.values()) <= budget:
        allocations.update(floors)
        budget -= sum(floors.values())
    else:
        _distribute_digest_budget(allocations, floors, token_budget=budget)
        return allocations

    _distribute_digest_budget(allocations, demand_by_id, token_budget=budget)
    return allocations


def _distribute_digest_budget(
    allocations: dict[str, int],
    caps: dict[str, int],
    *,
    token_budget: int,
) -> None:
    """Distribute a token pool simultaneously using sqrt-weighted water filling."""
    remaining = max(0, token_budget)
    while remaining:
        active = [
            attachment_id for attachment_id, cap in caps.items() if allocations[attachment_id] < cap
        ]
        if not active:
            break
        weights = {
            attachment_id: math.sqrt(caps[attachment_id] - allocations[attachment_id])
            for attachment_id in active
        }
        weight_total = sum(weights.values())
        round_budget = remaining
        grants: dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        for attachment_id in active:
            ideal = round_budget * weights[attachment_id] / weight_total
            capacity = caps[attachment_id] - allocations[attachment_id]
            grants[attachment_id] = min(int(ideal), capacity)
            remainders.append((ideal - int(ideal), attachment_id))

        leftover = min(round_budget - sum(grants.values()), remaining)
        for _, attachment_id in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if leftover <= 0:
                break
            capacity = caps[attachment_id] - allocations[attachment_id] - grants[attachment_id]
            if capacity <= 0:
                continue
            grants[attachment_id] += 1
            leftover -= 1

        progressed = sum(grants.values())
        if progressed == 0:
            break
        for attachment_id, grant in grants.items():
            allocations[attachment_id] += grant
        remaining -= progressed


def _build_document_planner_digest(
    *,
    bundle: ParsedAttachmentBundle,
    cleaned_chunks: list[str],
    input_tokens: int,
    token_budget: int,
) -> tuple[str, dict[str, int | bool]]:
    if token_budget <= 0 or not cleaned_chunks:
        return "", {"anchor_tokens": 0, "sample_tokens": 0, "sampled": False}

    full_text = "\n\n".join(cleaned_chunks)
    if input_tokens <= token_budget and estimate_tokens(full_text) <= token_budget:
        return full_text, {
            "anchor_tokens": 0,
            "sample_tokens": estimate_tokens(full_text),
            "sampled": False,
        }

    anchor_budget = min(_ANCHOR_MAX_TOKENS, int(token_budget * _ANCHOR_RATIO))
    anchor_indices = _structural_chunk_indices(bundle, cleaned_chunks)
    anchor_slots = min(len(anchor_indices), max(1, anchor_budget // 256))
    selected_anchor_indices = _uniform_indices(anchor_indices, anchor_slots)
    anchor_text = _fit_digest_segments(
        [cleaned_chunks[index] for index in selected_anchor_indices],
        token_budget=anchor_budget,
    )
    used_anchor_tokens = estimate_tokens(anchor_text)

    sample_budget = max(0, token_budget - used_anchor_tokens - estimate_tokens("[Structure]\n"))
    sample_slots = max(
        _MIN_SAMPLE_SLOTS,
        min(_MAX_SAMPLE_SLOTS, round(sample_budget / _SAMPLE_TARGET_TOKENS)),
    )
    anchor_set = set(selected_anchor_indices)
    sample_candidates = [index for index in range(len(cleaned_chunks)) if index not in anchor_set]
    selected_sample_indices = _uniform_token_indices(
        cleaned_chunks,
        sample_candidates,
        sample_slots,
    )

    label_tokens = estimate_tokens("[Structure]\n\n[Coverage]\n")
    sample_text = _fit_digest_segments(
        [cleaned_chunks[index] for index in selected_sample_indices],
        token_budget=max(0, token_budget - used_anchor_tokens - label_tokens),
    )
    sections: list[str] = []
    if anchor_text:
        sections.append(f"[Structure]\n{anchor_text}")
    if sample_text:
        sections.append(f"[Coverage]\n{sample_text}")
    digest = "\n\n".join(sections)
    if estimate_tokens(digest) > token_budget:
        digest = truncate_to_estimated_tokens(digest, token_budget)
    return digest, {
        "anchor_tokens": used_anchor_tokens,
        "sample_tokens": estimate_tokens(sample_text),
        "sampled": True,
    }


def _clean_digest_text(text: str) -> str:
    """Compact parser markup and low-information form lines for planning."""
    if not text:
        return ""
    compacted = _JSON_TABLE_RE.sub(_compact_json_table, text)
    lines: list[str] = []
    seen_short_lines: set[str] = set()
    for raw_line in compacted.splitlines():
        line = " ".join(html.unescape(raw_line).split())
        if not line or _FILL_IN_LINE_RE.fullmatch(line):
            continue
        normalized = line.casefold()
        if len(line) <= 120 and normalized in seen_short_lines:
            continue
        if len(line) <= 120:
            seen_short_lines.add(normalized)
        lines.append(line)
    return "\n".join(lines)


def _compact_json_table(match: re.Match[str]) -> str:
    """Keep representative rows from LightRAG's inline JSON table markup."""
    raw_body = html.unescape(match.group(1).strip())
    try:
        value = json.loads(raw_body)
    except json.JSONDecodeError, TypeError:
        return f"[Table] {truncate_to_estimated_tokens(raw_body, 256)}".strip()
    if not isinstance(value, list):
        return f"[Table] {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
    sampled = value if len(value) <= 4 else [value[0], value[1], value[len(value) // 2], value[-1]]
    payload = json.dumps(sampled, ensure_ascii=False, separators=(",", ":"))
    return f"[Table] {payload}"


def _structural_chunk_indices(
    bundle: ParsedAttachmentBundle, cleaned_chunks: list[str]
) -> list[int]:
    indices: list[int] = [0] if cleaned_chunks else []
    cleaned_index = 0
    for chunk in bundle.chunks:
        cleaned = _clean_digest_text(chunk.content)
        if not cleaned:
            continue
        metadata = chunk.metadata or {}
        metadata_is_structural = any(
            metadata.get(key)
            for key in (
                "heading",
                "title",
                "caption",
                "parent_headings",
                "content_type",
                "block_type",
            )
        )
        if (
            (chunk.sidecar_type or "").casefold() in _STRUCTURAL_SIDECAR_TYPES
            or metadata_is_structural
            or _STRUCTURE_MARKER_RE.search(cleaned)
        ):
            indices.append(cleaned_index)
        cleaned_index += 1
    return list(dict.fromkeys(indices))


def _uniform_indices(indices: list[int], count: int) -> list[int]:
    """Choose deterministic first/middle/last-aware positions."""
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)
    positions = {0, len(indices) // 2, len(indices) - 1}
    if count == 1:
        positions = {len(indices) // 2}
    elif count == 2:
        positions = {0, len(indices) - 1}
    else:
        positions.update(round(slot * (len(indices) - 1) / (count - 1)) for slot in range(count))
    if len(positions) > count:
        required = {0, len(indices) // 2, len(indices) - 1}
        positions = required | set(sorted(positions - required)[: max(0, count - len(required))])
    if len(positions) < count:
        for position in range(len(indices)):
            positions.add(position)
            if len(positions) == count:
                break
    return [indices[position] for position in sorted(positions)]


def _uniform_token_indices(contents: list[str], indices: list[int], count: int) -> list[int]:
    """Choose chunks nearest balanced positions in the document token stream."""
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)

    token_sizes = [max(1, estimate_tokens(content)) for content in contents]
    total_tokens = sum(token_sizes)
    midpoint_by_index: dict[int, float] = {}
    offset = 0
    for index, size in enumerate(token_sizes):
        midpoint_by_index[index] = offset + size / 2
        offset += size

    if count == 1:
        targets = [total_tokens / 2]
    elif count == 2:
        targets = [0.0, float(total_tokens)]
    else:
        normalized_targets = [0.0, 0.5, 1.0]
        while len(normalized_targets) < count:
            intervals = list(zip(normalized_targets, normalized_targets[1:], strict=False))
            start, end = max(
                intervals,
                key=lambda interval: (
                    interval[1] - interval[0],
                    -abs((interval[0] + interval[1]) / 2 - 0.5),
                    interval[0],
                ),
            )
            normalized_targets.append((start + end) / 2)
            normalized_targets.sort()
        targets = [target * total_tokens for target in normalized_targets]
    available = set(indices)
    selected: set[int] = set()
    for target in targets:
        if len(selected) == count or not available:
            break
        nearest = min(
            available,
            key=lambda index: (abs(midpoint_by_index[index] - target), index),
        )
        selected.add(nearest)
        available.remove(nearest)
    if len(selected) < count:
        for index in indices:
            selected.add(index)
            if len(selected) == count:
                break
    return sorted(selected)


def _fit_digest_segments(segments: list[str], *, token_budget: int) -> str:
    if token_budget <= 0 or not segments:
        return ""
    separator = "\n\n"
    content_budget = max(
        0,
        token_budget - estimate_tokens(separator) * max(0, len(segments) - 1),
    )
    remaining = content_budget
    fitted: list[str] = []
    for index, segment in enumerate(segments):
        remaining_segments = len(segments) - index
        piece = truncate_to_estimated_tokens(
            segment,
            max(0, remaining // remaining_segments),
        )
        if piece:
            fitted.append(piece)
            remaining = max(0, remaining - estimate_tokens(piece))
    return separator.join(fitted)


__all__ = [
    "ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET",
    "build_attachment_planner_digests",
]
