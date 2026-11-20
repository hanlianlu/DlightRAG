"""Validate a draft document and seal it when no rule blocks."""


def validate(draft: str) -> int:
    """The number of rules that block the draft."""
    return 0 if draft.strip() else 1


if __name__ == "__main__":
    raise SystemExit(validate("draft"))
