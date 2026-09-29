"""
Embed chunks and store in pgvector (primary) and Chroma (secondary).

Why two stores?
- pgvector: production-grade, lives in Postgres alongside metadata,
  supports hybrid search (vector + SQL filters on ticker/date/section)
- Chroma: fast local development and benchmarking
- Having both shows the ability to evaluate and compare vector stores,
  which is explicitly mentioned in AI Data Engineer job postings.
"""

from __future__ import annotations

import uuid

import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from src.common.config import settings
from src.common.logging import get_logger
from src.pipeline.chunker import Chunk

log = get_logger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality for financial text
BATCH_SIZE = 64


def load_model() -> SentenceTransformer:
    log.info("loading embedding model %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def setup_pgvector(engine) -> None:
    """Create the pgvector extension and chunks table."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sec_chunks (
                id          TEXT PRIMARY KEY,
                ticker      TEXT NOT NULL,
                form_type   TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                accession   TEXT NOT NULL,
                section     TEXT,
                chunk_index INTEGER,
                text        TEXT NOT NULL,
                embedding   vector(384)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS sec_chunks_embedding_idx
            ON sec_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))


def embed_and_store(chunks: list[Chunk], model: SentenceTransformer) -> int:
    """Embed chunks and upsert into pgvector and Chroma."""
    if not chunks:
        return 0

    engine = create_engine(settings.postgres_url)
    setup_pgvector(engine)

    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="sec_filings",
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c.text for c in chunks]
    log.info("embedding %d chunks in batches of %d", len(chunks), BATCH_SIZE)
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    # pgvector upsert
    with engine.begin() as conn:
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{chunk.ticker}_{chunk.accession}_{chunk.chunk_index}"
            conn.execute(text("""
                INSERT INTO sec_chunks
                    (id, ticker, form_type, filing_date, accession,
                     section, chunk_index, text, embedding)
                VALUES
                    (:id, :ticker, :form_type, :filing_date, :accession,
                     :section, :chunk_index, :text, :embedding)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
            """), {
                "id": chunk_id,
                "ticker": chunk.ticker,
                "form_type": chunk.form_type,
                "filing_date": chunk.filing_date,
                "accession": chunk.accession,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": emb.tolist(),
            })

    # Chroma upsert
    collection.upsert(
        ids=[f"{c.ticker}_{c.accession}_{c.chunk_index}" for c in chunks],
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{
            "ticker": c.ticker,
            "form_type": c.form_type,
            "filing_date": c.filing_date,
            "section": c.section,
        } for c in chunks],
    )

    log.info("stored %d chunks in pgvector and Chroma", len(chunks))
    return len(chunks)
