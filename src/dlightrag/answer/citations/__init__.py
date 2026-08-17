"""Citation processing and semantic highlighting for DlightRAG.

Everything else in this package is imported from its own module; only the
answer-finalizing entry point is re-exported here.
"""

from .finalization import finalize_answer

__all__ = ["finalize_answer"]
