"""Tests for search result structure. No DB needed."""

from src.retrieval.search import SearchResult


def test_search_result_fields():
    r = SearchResult(
        chunk_id="aapl_acc_0",
        ticker="aapl",
        filing_date="2024-01-01",
        section="Item 1",
        text="Apple designs iPhones.",
        score=0.85,
    )
    assert r.ticker == "aapl"
    assert 0 <= r.score <= 1
