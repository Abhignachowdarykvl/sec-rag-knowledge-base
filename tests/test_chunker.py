"""Tests for the document chunker. No network or DB needed."""

from pathlib import Path

import pytest

from src.pipeline.chunker import Chunk, chunk_filing, detect_section, extract_text


def test_detect_section_finds_item():
    text = "Item 1. Business\nWe are a technology company."
    assert "item" in detect_section(text).lower()


def test_detect_section_returns_unknown_for_empty():
    assert detect_section("") == "Unknown"


def test_chunk_filing_with_sample_html(tmp_path):
    html = tmp_path / "test-acc.html"
    html.write_text("""<html><body>
    <p>Item 1. Business</p>
    <p>We are a technology company that designs, manufactures, and markets
    smartphones, personal computers, tablets, wearables, and accessories,
    and sells a variety of related services. This is a longer paragraph
    to ensure the text is substantial enough to generate at least one chunk
    from the filing document we are testing here today.</p>
    </body></html>""")

    chunks = chunk_filing(html, ticker="test", form_type="10-K", filing_date="2024-01-01")
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.ticker == "test" for c in chunks)


def test_chunk_filing_skips_empty_html(tmp_path):
    html = tmp_path / "empty.html"
    html.write_text("<html><body></body></html>")
    chunks = chunk_filing(html, ticker="test", form_type="10-K", filing_date="2024-01-01")
    assert chunks == []
