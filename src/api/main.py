"""
FastAPI Q&A endpoint for the SEC RAG knowledge base.

Endpoints:
  POST /ask    — answer a question using retrieved filing chunks
  GET  /health — health check

Design:
- Each request retrieves top-k chunks, assembles a prompt, and calls
  the Anthropic API for generation.
- The response includes the answer AND the source chunks (citations),
  so the user can verify the answer.
- Every call is traced in Langfuse for observability.
"""

from __future__ import annotations

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.common.config import settings
from src.common.logging import get_logger
from src.retrieval.search import search

log = get_logger(__name__)

app = FastAPI(title="SEC Filings RAG API", version="1.0.0")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class AskRequest(BaseModel):
    question: str
    ticker: str | None = None
    form_type: str | None = None
    top_k: int = 5


class Citation(BaseModel):
    ticker: str
    filing_date: str
    section: str
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    chunks_retrieved: int


SYSTEM_PROMPT = """You are a financial analyst assistant with access to SEC filings.
Answer questions based ONLY on the provided filing excerpts.
If the answer is not in the excerpts, say so clearly.
Always cite which filing and section your answer comes from.
Be concise and accurate."""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    model = get_model()
    results = search(
        request.question,
        model,
        ticker=request.ticker,
        form_type=request.form_type,
        top_k=request.top_k,
    )

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    context = "\n\n---\n\n".join([
        f"[{r.ticker} {r.form_type} {r.filing_date} | {r.section}]\n{r.text}"
        for r in results
    ])

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context from SEC filings:\n\n{context}\n\nQuestion: {request.question}"
        }]
    )

    answer = message.content[0].text
    log.info("answered question '%s' using %d chunks", request.question[:50], len(results))

    return AskResponse(
        question=request.question,
        answer=answer,
        citations=[Citation(
            ticker=r.ticker,
            filing_date=r.filing_date,
            section=r.section,
            text=r.text[:200] + "...",
            score=r.score,
        ) for r in results],
        chunks_retrieved=len(results),
    )
