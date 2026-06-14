"""Token counting.

Two methods, same idea as a real production setup:
  - tiktoken : exact BPE counts (GPT cl100k_base reference). Needs the tiktoken
               package, which downloads its vocab once. Use this when you can.
  - approx   : a pure-Python estimate (no install, works offline). Calibrated to
               typical English; expect roughly +/-15% vs exact BPE.

"auto" uses tiktoken if it's importable and loads, otherwise falls back to approx
and tells you which it used - so the tool always runs, and is honest about it.

Note on model families: tiktoken's cl100k_base is a widely-used *reference*
tokenizer (GPT-4/3.5). Claude and other families tokenize slightly differently,
so treat counts as a close estimate for cost planning, not a billing guarantee.
"""

from __future__ import annotations
import re

_WORD_RE = re.compile(r"\S+")


def approx_tokens(text: str) -> int:
    """Estimate token count without any model files.

    Blends a character-based estimate (~4 chars/token) with a word-based estimate
    (~1.3 tokens/word), which tracks real BPE on typical English reasonably well.
    """
    if not text:
        return 0
    chars = len(text)
    words = len(_WORD_RE.findall(text))
    by_chars = chars / 4.0
    by_words = words * 1.33
    # never fewer tokens than words; round to a whole number
    return max(words, round((by_chars + by_words) / 2))


def count_tokens(text: str, method: str = "auto") -> tuple[int, str]:
    """Return (token_count, method_used). method in {auto, tiktoken, approx}."""
    if method in ("auto", "tiktoken"):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text)), "tiktoken"
        except Exception:
            if method == "tiktoken":
                raise
            # fall through to approx
    return approx_tokens(text), "approx"
