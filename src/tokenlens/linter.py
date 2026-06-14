"""Prompt linter - catch waste BEFORE you spend, with a concrete fix each time.

This is the proactive side of precaution: instead of reporting cost after the
fact, it inspects a request and flags patterns that waste tokens or money, each
with an actionable suggestion. Findings have a severity so it can act as a
pre-send check or a CI gate (high-severity findings -> non-zero exit).
"""

from __future__ import annotations
import re

from .tokenizer import count_tokens
from .optimizer import route, recommend_output_cap, _FILLER

HIGH, MED, LOW = "high", "med", "low"

# Tasks where a big model is overkill.
_EASY = {"classify", "extract", "tag"}


def lint(text: str, task: str, output_tokens: int, model: str,
         prices: dict, method: str = "auto") -> list[dict]:
    """Return a list of findings: {severity, code, message, suggestion}."""
    findings = []
    tokens, _ = count_tokens(text, method=method)

    # 1. Bloated input
    if tokens > 2000:
        findings.append({
            "severity": MED, "code": "large_input",
            "message": f"input is {tokens} tokens",
            "suggestion": "trim background/examples or retrieve only what's needed"})

    # 2. Filler phrases present
    if any(re.search(p, text, re.IGNORECASE) for p in _FILLER):
        findings.append({
            "severity": LOW, "code": "filler",
            "message": "contains filler phrases (e.g. 'please note that')",
            "suggestion": "remove filler - run `tokenlens optimize`"})

    # 3. Duplicate lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) != len(set(lines)):
        findings.append({
            "severity": LOW, "code": "duplicate_lines",
            "message": "repeated lines detected",
            "suggestion": "deduplicate repeated instructions"})

    # 4. Uncapped / oversized output
    cap = recommend_output_cap(task)
    if output_tokens == 0:
        findings.append({
            "severity": MED, "code": "uncapped_output",
            "message": "no output limit set (output costs ~4-5x input)",
            "suggestion": f"set max output around {cap} tokens for a '{task}' task"})
    elif output_tokens > cap * 2:
        findings.append({
            "severity": MED, "code": "large_output",
            "message": f"output budget {output_tokens} is high for a '{task}' task",
            "suggestion": f"consider ~{cap} tokens unless you need long output"})

    # 5. Expensive model on an easy task
    if task in _EASY and model in prices:
        cheap, _ = route(task, prices, input_tokens=tokens)
        if cheap != model:
            dear = prices[model]["input"] + prices[model]["output"]
            low = prices[cheap]["input"] + prices[cheap]["output"]
            if low < dear:
                findings.append({
                    "severity": HIGH, "code": "overpowered_model",
                    "message": f"using '{model}' for a simple '{task}' task",
                    "suggestion": f"route to '{cheap}' - same job, much cheaper"})

    return findings


def worst_severity(findings: list[dict]) -> str | None:
    for sev in (HIGH, MED, LOW):
        if any(f["severity"] == sev for f in findings):
            return sev
    return None
