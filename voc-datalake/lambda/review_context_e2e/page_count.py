"""Small, isolated pagination helper used only by a disposable E2E test PR."""


def page_count(total_items: int, page_size: int) -> int:
    """Return pages required, including a final partial page."""
    if total_items < 0:
        raise ValueError("total_items must be non-negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return total_items // page_size
