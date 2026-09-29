"""
Run the full chunking + embedding + storage pipeline.

Steps:
1. Read all transcript files from data/raw/
2. Chunk each transcript with metadata
3. Embed all chunks using sentence-transformers
4. Store in pgvector (Postgres) and Chroma
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.common.logging import get_logger
from src.pipeline.chunker import chunk_filing
from src.pipeline.embedder import embed_and_store, load_model

log = get_logger(__name__)

RAW_DIR = Path("data/raw")


def run(ticker: str | None = None) -> int:
    model = load_model()
    total = 0

    tickers = [ticker.upper()] if ticker else [
        p.name.upper() for p in RAW_DIR.iterdir()
        if p.is_dir() and p.name != "_manifest.json"
    ]

    for t in tickers:
        ticker_dir = RAW_DIR / t.lower()
        if not ticker_dir.exists():
            log.warning("no data for ticker %s", t)
            continue

        txt_files = list(ticker_dir.glob("*.txt"))
        log.info("processing %d transcripts for %s", len(txt_files), t)

        all_chunks = []
        for txt_path in txt_files:
            # Extract quarter from filename e.g. 2021-Q3_0.txt
            parts = txt_path.stem.split("_")
            quarter = parts[0] if parts else "unknown"
            chunks = chunk_filing(txt_path, ticker=t.lower(),
                                  form_type="earnings_call",
                                  filing_date=quarter)
            all_chunks.extend(chunks)

        if all_chunks:
            stored = embed_and_store(all_chunks, model)
            total += stored
            log.info("stored %d chunks for %s", stored, t)

    log.info("pipeline complete: %d total chunks stored", total)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="Single ticker to process")
    args = parser.parse_args()
    run(args.ticker)
