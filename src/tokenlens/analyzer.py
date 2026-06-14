"""Analyze a log of LLM requests to show where the tokens (and money) go.

Input: a JSONL file where each line is one request, with any of:
  - "text"          : the prompt text (tokens counted for you), OR
  - "input_tokens"  : precounted input tokens (used if no text)
  - "output_tokens" : output tokens (default 0)
  - "model"         : model name (falls back to default_model)
  - "tag"           : a label like "summarize" / "classify" / "chat"
  - "id"            : optional identifier

Output: totals plus breakdowns by model and by tag, and the most expensive
requests - the view that answers "what is actually driving my bill?"
"""

from __future__ import annotations
import json
from collections import defaultdict

from .tokenizer import count_tokens
from .pricing import cost


def _entry_tokens(entry: dict, method: str) -> tuple[int, int]:
    if "input_tokens" in entry:
        inp = int(entry["input_tokens"])
    else:
        inp, _ = count_tokens(entry.get("text", ""), method=method)
    out = int(entry.get("output_tokens", 0))
    return inp, out


def analyze(path: str, prices: dict, default_model: str,
            method: str = "auto") -> dict:
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            model = e.get("model", default_model)
            inp, out = _entry_tokens(e, method)
            c = cost(inp, out, model, prices) if model in prices else 0.0
            rows.append({
                "id": e.get("id", f"req_{len(rows)+1}"),
                "tag": e.get("tag", "untagged"),
                "model": model,
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out,
                "cost": c,
            })

    total_cost = sum(r["cost"] for r in rows)
    total_tokens = sum(r["total_tokens"] for r in rows)

    by_model = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
    by_tag = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
    for r in rows:
        for key, bucket in ((r["model"], by_model), (r["tag"], by_tag)):
            bucket[key]["requests"] += 1
            bucket[key]["tokens"] += r["total_tokens"]
            bucket[key]["cost"] += r["cost"]

    top = sorted(rows, key=lambda r: r["cost"], reverse=True)[:5]

    return {
        "n_requests": len(rows),
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_cost": total_cost / len(rows) if rows else 0.0,
        "by_model": dict(by_model),
        "by_tag": dict(by_tag),
        "top_requests": top,
        "rows": rows,
    }
