# SEC Filings RAG Knowledge Base

AI Data Engineering pipeline: EDGAR filings → parse & chunk → embed → pgvector + Chroma → hybrid search → FastAPI Q&A with citations → evaluation.

## Business Problem

SEC 10-K and 10-Q filings contain critical financial and risk information, but they are thousands of pages long. This project builds a knowledge base that lets analysts ask natural-language questions and get cited answers grounded in the actual filing text.

**Example questions:**
- "What are Apple's primary revenue sources?"
- "What risks does Microsoft identify for its cloud business?"
- "How did Amazon's AWS margins change year over year?"

## Architecture

```
EDGAR API ──► Python ingest ──► data/raw/ (HTML filings)
           ──► Parse & chunk (800 token chunks with section metadata)
           ──► sentence-transformers (all-MiniLM-L6-v2 embeddings)
           ──► pgvector (primary, hybrid search)
           ──► Chroma (secondary, benchmarking)
           ──► FastAPI /ask endpoint (retrieval + Anthropic generation)
           ──► Langfuse (observability: latency, tokens, cost)
           ──► Evaluation harness (hit rate, faithfulness)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Python (requests, BeautifulSoup) |
| Chunking | LangChain text splitters |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector stores | pgvector (Postgres) + Chroma |
| Generation | Anthropic Claude API |
| API | FastAPI |
| Observability | Langfuse |
| Evaluation | Custom harness (hit rate + LLM-as-judge faithfulness) |
| Testing | pytest, GitHub Actions CI |

## Quick Start

```bash
git clone https://github.com/Abhignachowdarykvl/sec-rag-knowledge-base.git
cd sec-rag-knowledge-base
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY

# Start pgvector
docker compose up -d

# Ingest filings
python -m src.ingest.edgar_ingest --ticker aapl --form 10-K --limit 3

# Chunk, embed, store
python -m src.pipeline.run_pipeline --ticker aapl

# Start the API
uvicorn src.api.main:app --reload

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Apple primary revenue sources?", "ticker": "aapl"}'

# Run evaluation
python -m src.eval.evaluate
```

## Evaluation Results

| Metric | Score |
|---|---|
| Retrieval hit rate | 92% (23/25 questions) |
| Answer faithfulness | 0.82 / 1.0 (LLM-as-judge) |
| Eval set size | 25 questions across 7 companies |

## Key Design Decisions

- **Chunk size 800 tokens with 100-token overlap** — balances context vs retrieval noise
- **Section metadata on every chunk** — enables filtering by Item 1, Item 7, etc.
- **Hybrid search** — vector similarity + metadata filters for precision
- **Two vector stores** — pgvector for production, Chroma for benchmarking
- **LLM-as-judge evaluation** — scalable faithfulness scoring without human labels

## Skills Demonstrated

`RAG pipelines` `document parsing` `chunking strategy` `embeddings` `pgvector` `Chroma`
`hybrid search` `FastAPI` `Anthropic API` `LLM evaluation` `Langfuse` `Docker` `CI/CD`
`unstructured data` `financial domain`
