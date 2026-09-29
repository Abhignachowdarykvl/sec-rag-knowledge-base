"""Ingest SEC 10-K and 10-Q filings from EDGAR. Fixed URL pattern."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.common.logging import get_logger

log = get_logger(__name__)

RAW_DIR = Path("data/raw")
EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "sec-rag-knowledge-base research@example.com"}

COMPANIES = {
    "aapl": "0000320193",
    "msft": "0000789019",
    "googl": "0001652044",
    "amzn": "0001018724",
    "meta": "0001326801",
}


def build_session() -> requests.Session:
    retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def get_filings(session: requests.Session, cik: str, form_type: str, limit: int = 5) -> list[dict]:
    url = f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])
    results = []
    for form, acc, date, doc in zip(forms, accessions, dates, primary_docs):
        if form == form_type and len(results) < limit:
            results.append({"form": form, "accession": acc, "date": date,
                            "primary_doc": doc, "cik": cik})
    return results


def download_filing(session: requests.Session, filing: dict, out_dir: Path) -> Path | None:
    cik_int = str(int(filing["cik"]))
    acc_clean = filing["accession"].replace("-", "")
    base_url = f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}"
    out_path = out_dir / f"{filing['accession']}.html"
    if out_path.exists():
        log.info("skipping existing %s", out_path.name)
        return out_path
    # Try primary doc and both .htm/.html variants
    candidates = [filing["primary_doc"]]
    doc = filing["primary_doc"]
    if doc.endswith(".htm"):
        candidates.append(doc + "l")
    elif doc.endswith(".html"):
        candidates.append(doc[:-1])
    for candidate in candidates:
        try:
            url = f"{base_url}/{candidate}"
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            log.info("downloaded %s (%d bytes)", out_path.name, len(resp.content))
            time.sleep(0.5)
            return out_path
        except Exception as e:
            log.warning("failed %s: %s", url, e)
    return None


def ingest(ticker: str, form_type: str = "10-K", limit: int = 3) -> list[Path]:
    cik = COMPANIES.get(ticker.lower())
    if not cik:
        raise ValueError(f"Unknown ticker {ticker}.")
    out_dir = RAW_DIR / ticker.lower() / form_type
    out_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()
    filings = get_filings(session, cik, form_type, limit)
    log.info("found %d %s filings for %s", len(filings), form_type, ticker)
    for f in filings:
        log.info("filing: %s %s %s", f["accession"], f["date"], f["primary_doc"])
    paths = []
    for filing in filings:
        path = download_filing(session, filing, out_dir)
        if path:
            paths.append(path)
    manifest = {"ticker": ticker, "cik": cik, "form_type": form_type,
                "filings_downloaded": len(paths), "run_at": datetime.now(UTC).isoformat()}
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("ingested %d filings for %s", len(paths), ticker)
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="aapl")
    parser.add_argument("--form", default="10-K")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    ingest(args.ticker, args.form, args.limit)
