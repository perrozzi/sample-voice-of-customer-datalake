from .page_count import page_count


def test_exact_pages() -> None:
    assert page_count(20, 10) == 2


def test_final_partial_page() -> None:
    assert page_count(21, 10) == 3
