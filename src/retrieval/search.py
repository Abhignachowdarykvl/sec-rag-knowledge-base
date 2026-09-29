from __future__ import annotations
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from src.common.config import settings
from src.common.logging import get_logger

log = get_logger(__name__)

@dataclass
class SearchResult:
    chunk_id: str
    ticker: str
    form_type: str
    filing_date: str
    section: str
    text: str
    score: float

def search(query, model, ticker=None, form_type=None, top_k=5):
    query_embedding = model.encode([query])[0]
    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    engine = create_engine(settings.postgres_url)
    filters = []
    params = {"top_k": top_k}
    if ticker:
        filters.append("ticker = :ticker")
        params["ticker"] = ticker
    if form_type:
        filters.append("form_type = :form_type")
        params["form_type"] = form_type
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    sql = text(f"""
        SELECT id, ticker, form_type, filing_date, section, text,
            1 - (embedding <=> '{embedding_str}'::vector) AS score
        FROM sec_chunks {where}
        ORDER BY embedding <=> '{embedding_str}'::vector
        LIMIT :top_k
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    results = [SearchResult(
        chunk_id=r.id, ticker=r.ticker, form_type=r.form_type,
        filing_date=r.filing_date, section=r.section,
        text=r.text, score=float(r.score)) for r in rows]
    log.info("search done -> %d results", len(results))
    return results
