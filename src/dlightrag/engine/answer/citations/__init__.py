"""Citation processing and semantic highlighting for DlightRAG.

Everything else in this package is imported from its own module; only the
answer-finalizing entry point and the publication projection are re-exported
here.
"""

from .finalization import finalize_answer
from .projection import link_public_citations

__all__ = ["finalize_answer", "link_public_citations"]
