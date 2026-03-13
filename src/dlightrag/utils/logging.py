import logging
from collections.abc import Sequence
from typing import Any


def log_references(location: str, refs: Sequence[Any], **context):
    """Unified references logging for all layers.
    Args:
        location: str, e.g. 'ragservice.answer', 'manager.federation', 'api.response'
        refs: list or sequence of references (can be empty)
        context: extra info, e.g. query, workspace, mode, etc.
    """
    logger = logging.getLogger("dlightrag.references")
    level = logging.WARNING if not refs else logging.INFO
    sample = refs[:2] if refs else []
    ctx_str = " ".join(f"{k}={str(v)[:40]}" for k, v in context.items())
    msg = f"[REFERENCES][{location}] count={len(refs)} sample={sample} {ctx_str}"
    logger.log(level, msg)
