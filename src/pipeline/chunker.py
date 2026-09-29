"""Parse transcript text files and split into chunks with metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.logging import get_logger

log = get_logger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SECTION_PATTERN = re.compile(
    r"(item\s+\d+[a-z]?\.?\s+[\w\s,&-]{3,60})", re.IGNORECASE
)


@dataclass
class Chunk:
    text: str
    ticker: str
    form_type: str
    filing_date: str
    accession: str
    section: str
    chunk_index: int
    char_count: int


def extract_text(file_path: Path) -> str:
    """Read text from .txt or .html files."""
    if file_path.suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")
    else:
        from bs4 import BeautifulSoup
        html = file_path.read_bytes()
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "table"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def detect_section(text_fragment: str) -> str:
    matches = SECTION_PATTERN.findall(text_fragment[-500:])
    return matches[-1].strip() if matches else "Transcript"


def chunk_filing(file_path: Path, ticker: str, form_type: str, filing_date: str) -> list[Chunk]:
    accession = file_path.stem
    raw_text = extract_text(file_path)

    if len(raw_text) < 100:
        log.warning("very short text for %s, skipping", file_path.name)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    texts = splitter.split_text(raw_text)

    chunks = []
    for i, text in enumerate(texts):
        chunks.append(Chunk(
            text=text,
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            accession=accession,
            section=detect_section(text),
            chunk_index=i,
            char_count=len(text),
        ))

    log.info("chunked %s into %d chunks", file_path.name, len(chunks))
    return chunks
