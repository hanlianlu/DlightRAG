"""Video link cards for one Answer.

The Model writes an address, the page declares whether it is a video, and the
card carries what the page said. Import the module's own names; this package
re-exports the two entry points an Answer settlement and a projection need.
"""

from .cards import LinkCard, collect_link_cards, project_link_cards

__all__ = ["LinkCard", "collect_link_cards", "project_link_cards"]
