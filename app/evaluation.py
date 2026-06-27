"""
Phase 2 — LLM-as-judge evaluation using LangChain structured output.

Also supports batch evaluation (score many Q/A/context items in one call) and
CSV export of the scores, for offline quality tracking.
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm import get_chat_llm

logger = logging.getLogger("doc-qa.eval")


class Score(BaseModel):
    score: int = Field(ge=1, le=5, description="Score from 1 to 5")
    reason: str = Field(description="Brief justification")


class EvalSchema(BaseModel):
    faithfulness: Score = Field(description="Is the answer grounded in the context?")
    relevance: Score = Field(description="Does it address the question?")
    completeness: Score = Field(description="Does it cover all relevant info?")
    overall_score: float = Field(description="Average of three scores rounded to 1 decimal")
    summary: str = Field(description="One sentence overall assessment")


EVAL_SYSTEM_PROMPT = """You are an impartial evaluator. Score the given answer on three criteria.

For each criterion, provide a score from 1-5 and a brief justification.

Criteria:
1. Faithfulness — Is the answer grounded in the provided context? (5 = fully grounded, 1 = hallucinated)
2. Relevance — Does the answer address the question asked? (5 = perfectly relevant, 1 = off-topic)
3. Completeness — Does the answer cover all relevant info from the context? (5 = complete, 1 = missing key info)"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", EVAL_SYSTEM_PROMPT),
    ("user", "CONTEXT PROVIDED TO THE MODEL:\n{context}\n\nQUESTION: {question}\n\nANSWER GIVEN: {answer}\n\nEvaluate the answer based on the criteria."),
])


def evaluate(question: str, answer: str, context_chunks: list[dict]) -> dict:
    """LLM-as-judge evaluation via structured output."""
    context_block = "\n\n---\n\n".join(
        f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}" for c in context_chunks
    )

    llm = get_chat_llm(temperature=0.0, max_tokens=512)
    structured_llm = llm.with_structured_output(EvalSchema, include_raw=True)
    chain = PROMPT | structured_llm

    result = chain.invoke({
        "context": context_block,
        "question": question,
        "answer": answer,
    })

    parsed: EvalSchema = result["parsed"]
    usage = result["raw"].usage_metadata or {}

    output = parsed.model_dump()
    output["eval_tokens"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
    return output


# ---------------------------------------------------------------------------
# Batch evaluation + CSV export
# ---------------------------------------------------------------------------

EVAL_CSV_DIR = Path("storage/eval_runs")


def batch_evaluate(items: list[dict], export_csv: bool = True,
                   csv_path: str | None = None) -> dict:
    """Evaluate many {question, answer, context_chunks} items sequentially.

    Returns {results, count, avg_overall, total_eval_tokens, csv_path}. Each
    result is a per-item eval dict plus its index/question; failures are captured
    as {error: ...} so one bad item doesn't abort the batch.
    """
    results: list[dict] = []
    total_prompt = total_completion = 0
    score_sum = 0.0
    scored = 0

    for i, item in enumerate(items):
        try:
            res = evaluate(item["question"], item["answer"], item.get("context_chunks", []))
            et = res["eval_tokens"]
            total_prompt += et["prompt_tokens"]
            total_completion += et["completion_tokens"]
            score_sum += res["overall_score"]
            scored += 1
            results.append({"index": i, "question": item["question"], **res})
        except Exception as e:
            logger.warning("Batch eval item %d failed: %s", i, e)
            results.append({"index": i, "question": item.get("question", ""), "error": str(e)})

    avg_overall = round(score_sum / scored, 3) if scored else 0.0
    out_path = None
    if export_csv:
        out_path = export_scores_csv(results, csv_path)

    return {
        "results": results,
        "count": len(results),
        "avg_overall": avg_overall,
        "total_eval_tokens": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        },
        "csv_path": out_path,
    }


def export_scores_csv(results: list[dict], csv_path: str | None = None) -> str:
    """Write batch eval results to a CSV. Returns the path written."""
    if csv_path:
        path = Path(csv_path)
    else:
        EVAL_CSV_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = EVAL_CSV_DIR / f"eval_{stamp}.csv"

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "index", "question", "overall_score", "faithfulness", "relevance",
        "completeness", "summary", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in results:
            if "error" in r:
                w.writerow([r.get("index"), r.get("question", "")[:500], "", "", "", "",
                            "", r["error"]])
                continue
            w.writerow([
                r.get("index"),
                r.get("question", "")[:500],
                r["overall_score"],
                r["faithfulness"]["score"],
                r["relevance"]["score"],
                r["completeness"]["score"],
                r["summary"][:500],
                "",
            ])
    logger.info("Wrote eval CSV: %s (%d rows)", path, len(results))
    return str(path)
