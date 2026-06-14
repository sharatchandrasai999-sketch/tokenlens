"""Token optimization - three independent, measurable levers.

The design rule (the honest one): optimize, but measure each lever *separately*
so you can say "trimming saved X, routing saved Y" instead of one mystery total.

Levers:
  1. trim    - remove structural waste (extra whitespace, duplicate lines, filler
               phrases). Conservative and meaning-preserving by construction, so it
               needs no quality check. Semantic/LLM compression is deliberately NOT
               here - that DOES change meaning and would require a quality eval.
  2. route   - pick the cheapest adequate model for the task instead of defaulting
               everything to the priciest one. Config-driven: tiers come from the
               price ranking in prices.json, not hard-coded model names.
  3. cap     - recommend a sane max output length (output tokens cost ~4-5x input).
               Advisory: the caller enforces it on the real API call.
"""

from __future__ import annotations
import re

from .tokenizer import count_tokens
from .pricing import cost

# Safe, meaning-preserving filler phrases to drop.
_FILLER = [
    r"\bplease note that\b", r"\bit is important to note that\b",
    r"\bas (?:previously |already )?mentioned\b", r"\bkindly\b",
    r"\bplease be aware that\b", r"\bi would like you to\b",
    r"\bi want you to\b", r"\bin order to\b",
]
_FILLER_REPL = {r"\bin order to\b": "to", r"\bi would like you to\b": "",
                r"\bi want you to\b": ""}

# Task -> price tier (low = cheapest model, high = priciest).
_TASK_TIER = {
    "classify": "low", "extract": "low", "tag": "low",
    "summarize": "mid", "translate": "mid",
    "chat": "high", "reason": "high", "code": "high",
}

# Suggested output cap (tokens) by task. Conservative starting points.
_OUTPUT_CAP = {
    "classify": 10, "extract": 120, "tag": 10,
    "summarize": 250, "translate": 400,
    "chat": 800, "reason": 900, "code": 1200,
}


def trim_prompt(text: str) -> str:
    """Remove structural waste without changing meaning."""
    # drop filler phrases
    for pat in _FILLER:
        text = re.sub(pat, _FILLER_REPL.get(pat, ""), text, flags=re.IGNORECASE)
    # collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # remove duplicate consecutive lines
    lines, out, prev = text.splitlines(), [], None
    for ln in lines:
        s = ln.strip()
        if s and s == prev:
            continue
        out.append(ln.rstrip())
        prev = s
    # collapse 3+ blank lines to one
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return cleaned


def _price_tiers(prices: dict) -> dict:
    """Map low/mid/high tiers to actual model names by price ranking."""
    ranked = sorted(prices, key=lambda m: prices[m]["input"] + prices[m]["output"])
    return {"low": ranked[0], "mid": ranked[len(ranked) // 2], "high": ranked[-1]}


def route(task: str, prices: dict, input_tokens: int = 0) -> tuple[str, str]:
    """Recommend the cheapest adequate model. Returns (model, reason)."""
    tier = _TASK_TIER.get(task, "mid")
    tiers = _price_tiers(prices)
    reason = f"task '{task}' -> {tier} tier"
    # very long inputs bump a low-tier task up one, since long context often
    # signals a more involved request.
    if tier == "low" and input_tokens > 4000:
        tier, reason = "mid", f"task '{task}' but long input ({input_tokens} tok) -> mid tier"
    return tiers[tier], reason


def recommend_output_cap(task: str) -> int:
    return _OUTPUT_CAP.get(task, 300)


def optimize(text: str, task: str, output_tokens: int, prices: dict,
             default_model: str, method: str = "auto") -> dict:
    """Run all three levers and report each one's separate contribution."""
    tok_before, _ = count_tokens(text, method=method)
    trimmed = trim_prompt(text)
    tok_after, _ = count_tokens(trimmed, method=method)

    rec_model, reason = route(task, prices, input_tokens=tok_after)
    cap = recommend_output_cap(task)
    capped_output = min(output_tokens, cap) if output_tokens else cap

    # baseline: original prompt, original output, default model
    cost_before = cost(tok_before, output_tokens or cap, default_model, prices) \
        if default_model in prices else 0.0
    # after all levers: trimmed prompt, capped output, routed model
    cost_after = cost(tok_after, capped_output, rec_model, prices)

    # isolate each lever's individual saving (hold the others at baseline)
    save_trim = (cost(tok_before, output_tokens or cap, default_model, prices)
                 - cost(tok_after, output_tokens or cap, default_model, prices)) \
        if default_model in prices else 0.0
    save_route = (cost(tok_after, output_tokens or cap, default_model, prices)
                  - cost(tok_after, output_tokens or cap, rec_model, prices)) \
        if default_model in prices else 0.0
    save_cap = cost(tok_after, (output_tokens or cap), rec_model, prices) \
        - cost(tok_after, capped_output, rec_model, prices)

    return {
        "task": task,
        "trim": {"tokens_before": tok_before, "tokens_after": tok_after,
                 "saved_tokens": tok_before - tok_after, "saved_cost": save_trim,
                 "text": trimmed},
        "routing": {"default_model": default_model, "recommended_model": rec_model,
                    "reason": reason, "saved_cost": save_route},
        "output_cap": {"suggested_max_tokens": cap, "saved_cost": save_cap},
        "totals": {"cost_before": cost_before, "cost_after": cost_after,
                   "saved_cost": cost_before - cost_after,
                   "saved_pct": ((cost_before - cost_after) / cost_before * 100)
                   if cost_before else 0.0},
    }


def fit_context(text: str, max_tokens: int, method: str = "auto") -> tuple[str, bool]:
    """Shrink text to fit a token budget. Returns (text, was_truncated).

    Strategy: keep the head and tail (where instructions and the question usually
    live) and drop the middle, marked with an ellipsis note. This is deliberate,
    honest truncation - NOT semantic compression, which would need a quality check.
    """
    tokens, _ = count_tokens(text, method=method)
    if tokens <= max_tokens:
        return text, False
    # rough char budget from the token budget (~4 chars/token), minus the marker
    marker = "\n...[trimmed to fit context]...\n"
    char_budget = max(max_tokens * 4 - len(marker), 0)
    head = char_budget // 2
    tail = char_budget - head
    fitted = text[:head] + marker + (text[-tail:] if tail else "")
    return fitted, True
