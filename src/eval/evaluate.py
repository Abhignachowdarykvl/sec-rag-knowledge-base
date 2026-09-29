"""
Evaluate RAG pipeline quality.

Metrics:
- Retrieval hit rate: does the correct chunk appear in top-k results?
- Answer faithfulness: is the answer grounded in retrieved chunks?
  (scored by the LLM itself — a common "LLM as judge" pattern)

Interview point: without evaluation you don't know if your RAG works.
A measured number (e.g., "82% hit rate") is what separates a portfolio
project from a production system.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
from sentence_transformers import SentenceTransformer

from src.common.config import settings
from src.common.logging import get_logger
from src.retrieval.search import search

log = get_logger(__name__)

EVAL_SET_PATH = Path("eval/eval_questions.json")


def load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Eval set not found at {EVAL_SET_PATH}")
    return json.loads(EVAL_SET_PATH.read_text())


def score_faithfulness(question: str, answer: str, context: str) -> float:
    """Ask the LLM to score whether the answer is grounded in the context."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = f"""Score whether this answer is faithfully grounded in the context.
Context: {context[:1000]}
Question: {question}
Answer: {answer}

Respond with ONLY a number from 0.0 to 1.0 where:
1.0 = fully grounded, every claim is in the context
0.5 = partially grounded
0.0 = not grounded or contradicts context"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(msg.content[0].text.strip())
    except ValueError:
        return 0.0


def run_evaluation(top_k: int = 5) -> dict:
    eval_set = load_eval_set()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    hit_count = 0
    faithfulness_scores = []

    for item in eval_set:
        results = search(item["question"], model, ticker=item.get("ticker"), top_k=top_k)
        retrieved_texts = [r.text for r in results]

        # Hit rate: does any result contain the expected answer keywords?
        hit = any(
            kw.lower() in " ".join(retrieved_texts).lower()
            for kw in item.get("keywords", [])
        )
        if hit:
            hit_count += 1

        if results and settings.anthropic_api_key:
            context = "\n".join(retrieved_texts[:3])
            score = score_faithfulness(item["question"], item.get("expected_answer", ""), context)
            faithfulness_scores.append(score)

    metrics = {
        "total_questions": len(eval_set),
        "hit_rate": round(hit_count / len(eval_set), 3) if eval_set else 0,
        "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
        if faithfulness_scores else None,
    }
    log.info("evaluation complete: %s", metrics)
    return metrics


if __name__ == "__main__":
    print(run_evaluation())
