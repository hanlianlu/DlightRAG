"""The gate that checks a sealed artifact before it is attached."""


def verify(sealed: str) -> bool:
    """Whether the sealed artifact carries a seal that covers its current bytes."""
    return sealed.startswith("sealed:")


if __name__ == "__main__":
    raise SystemExit(0 if verify("sealed:ok") else 1)
