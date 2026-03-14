import logging
from collections.abc import Sequence
from typing import Any


def log_answer_llm_output(
    location: str,
    *,
    structured: bool,
    provider: str | None = None,
    query: str,
    raw: str | None = None,
    answer_text: str | None = None,
    parse_error: Exception | None = None,
):
    """
    统一记录LLM answer相关的原始输出、路径分支、异常等。
    """
    logger = logging.getLogger("dlightrag.answer")
    ctx = f"structured={structured} provider={provider or 'N/A'} query={query[:40]}"
    if parse_error is not None:
        logger.warning(
            f"[ANSWER][{location}][parse_fail] {ctx} raw={str(raw)[:100]} exc={parse_error}"
        )
    elif raw is not None:
        logger.info(f"[ANSWER][{location}][llm_raw] {ctx} raw={str(raw)[:100]}")
    elif answer_text is not None:
        logger.info(f"[ANSWER][{location}][llm_raw] {ctx} answer_text={str(answer_text)[:100]}")
    else:
        logger.info(f"[ANSWER][{location}][llm_path] {ctx}")


def log_references(location: str, refs: Sequence[Any], **context):
    """Unified references logging for all layers.
    Args:
        location: str, e.g. 'ragservice.answer', 'manager.federation', 'api.response'
        refs: list or sequence of references (can be empty)
        context: extra info, e.g. query, workspace, mode, etc.
    """
    logger = logging.getLogger("dlightrag.answer")
    level = logging.WARNING if not refs else logging.INFO
    sample = refs[:2] if refs else []
    ctx_str = " ".join(f"{k}={str(v)[:40]}" for k, v in context.items())
    msg = f"[REFERENCES][{location}] count={len(refs)} sample={sample} {ctx_str}"
    logger.log(level, msg)
