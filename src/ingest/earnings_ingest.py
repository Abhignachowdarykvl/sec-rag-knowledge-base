"""
Ingest earnings call transcripts from the lamini/earnings-calls-qa dataset.

Why this dataset:
- Real earnings call transcripts for hundreds of public companies
- Includes Q&A pairs we can use directly as an evaluation set
- No API keys or web scraping needed — Hugging Face hosts it publicly
- Same RAG pipeline skills: chunking, embedding, retrieval, generation

Design:
- Download transcripts for target tickers
- Save raw text to data/raw/<ticker>/<quarter>.txt
- Extract Q&A pairs to eval/eval_questions.json
- Idempotent: skips already-downloaded tickers
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import UTC, datetime

from datasets import load_dataset

from src.common.logging import get_logger

log = get_logger(__name__)

RAW_DIR = Path("data/raw")
EVAL_DIR = Path("eval")

TARGET_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSM", "NVDA"}


def ingest(tickers: set[str] | None = None, max_per_ticker: int = 5) -> dict:
    tickers = tickers or TARGET_TICKERS
    log.info("loading earnings-calls-qa dataset from Hugging Face...")
    ds = load_dataset("lamini/earnings-calls-qa", split="train")
    log.info("dataset loaded: %d rows", len(ds))

    saved = {}
    eval_questions = []

    for row in ds:
        ticker = row.get("ticker", "").upper()
        if ticker not in tickers:
            continue
        if ticker not in saved:
            saved[ticker] = 0
        if saved[ticker] >= max_per_ticker:
            continue

        # Save transcript
        out_dir = RAW_DIR / ticker.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        quarter = row.get("q", "unknown").replace(" ", "_")
        out_path = out_dir / f"{quarter}_{saved[ticker]}.txt"

        if not out_path.exists():
            out_path.write_text(row["transcript"], encoding="utf-8")
            log.info("saved %s (%d chars)", out_path.name, len(row["transcript"]))

        saved[ticker] += 1

        # Collect eval Q&A pairs (only where answer is not "I do not know")
        answer = row.get("answer", "")
        question = row.get("question", "")
        if question and answer and "do not know" not in answer.lower():
            eval_questions.append({
                "question": question,
                "ticker": ticker.lower(),
                "expected_answer": answer[:300],
                "keywords": [w for w in answer.split()[:10] if len(w) > 4],
            })

    # Save eval set
    EVAL_DIR.mkdir(exist_ok=True)
    eval_path = EVAL_DIR / "eval_questions.json"
    eval_path.write_text(json.dumps(eval_questions[:50], indent=2))
    log.info("saved %d eval questions to %s", len(eval_questions[:50]), eval_path)

    manifest = {
        "tickers": list(saved.keys()),
        "transcripts_per_ticker": saved,
        "eval_questions": len(eval_questions[:50]),
        "run_at": datetime.now(UTC).isoformat(),
    }
    (RAW_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("ingestion complete: %s", saved)
    return saved


if __name__ == "__main__":
    ingest()
