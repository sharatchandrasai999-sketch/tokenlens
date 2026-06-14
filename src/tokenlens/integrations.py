"""Drop-in token saver for your own LLM code.

Point it at the function you already use to call your model, and every call now
gets optimized (trimmed prompt, right-sized model, capped output), budget-guarded,
and cached - automatically.

Two ways to use it:

    # 1) wrap an existing call
    from tokenlens import TokenSaver

    def my_llm(prompt, model, max_tokens):
        ...  # your real Anthropic/OpenAI call -> returns the answer string
        return answer

    saver = TokenSaver(my_llm, default_model="claude-sonnet",
                       per_request_limit=0.05)
    answer = saver("Summarize this contract.", task="summarize")
    print(saver.stats())   # calls, cache hits, total saved

    # 2) decorator
    from tokenlens import save_tokens

    @save_tokens(default_model="claude-sonnet", per_request_limit=0.05)
    def my_llm(prompt, model, max_tokens):
        return answer

    answer = my_llm("Summarize this contract.", task="summarize")

Your function can take (prompt), (prompt, model) or (prompt, model, max_tokens) -
whatever it already takes. Return a string, or a (string, output_tokens) tuple if
you want exact output-token accounting.
"""

from __future__ import annotations
import inspect
from functools import wraps

from .tokenizer import count_tokens
from .pricing import load_prices
from .pipeline import Pipeline


class BudgetExceeded(Exception):
    """Raised by the simple call form when the guard blocks a request."""


def _adapt(user_call):
    """Turn a flexible user function into pipeline's (prompt, model, max_tokens)
    -> (text, output_tokens) shape."""
    n_params = len(inspect.signature(user_call).parameters)

    def call_model(prompt, model, max_tokens):
        if n_params >= 3:
            out = user_call(prompt, model, max_tokens)
        elif n_params == 2:
            out = user_call(prompt, model)
        else:
            out = user_call(prompt)
        if isinstance(out, tuple):
            return out                       # (text, output_tokens)
        return out, count_tokens(out)[0]     # estimate output tokens from text
    return call_model


class TokenSaver:
    def __init__(self, call, *, prices: dict | None = None,
                 default_model: str = "claude-sonnet", task: str = "chat",
                 optimize: bool = True, use_cache: bool = True,
                 cache_path: str | None = None, threshold: float = 0.92,
                 per_request_limit: float | None = None,
                 daily_limit: float | None = None, method: str = "auto"):
        from .cache import SemanticCache
        self.default_task = task
        self._pipe = Pipeline(
            prices or load_prices(),
            default_model=default_model,
            call_model=_adapt(call),
            cache=SemanticCache(threshold=threshold, store_path=cache_path),
            per_request_limit=per_request_limit, daily_limit=daily_limit,
            optimize=optimize, use_cache=use_cache, method=method)

    def complete(self, prompt: str, task: str | None = None,
                 output_tokens: int | None = None) -> dict:
        """Full result dict (status, response, cache_hit, cost, saved, ...)."""
        return self._pipe.complete(prompt, task=task or self.default_task,
                                   output_tokens=output_tokens)

    def __call__(self, prompt: str, task: str | None = None,
                 output_tokens: int | None = None) -> str:
        """Convenience: return just the answer string; raise if budget-blocked."""
        r = self.complete(prompt, task=task, output_tokens=output_tokens)
        if r["status"] == "blocked":
            raise BudgetExceeded(r["reason"])
        return r["response"]

    def stats(self) -> dict:
        return self._pipe.stats()


def save_tokens(**config):
    """Decorator form of TokenSaver. Wraps your model-calling function."""
    def deco(fn):
        saver = TokenSaver(fn, **config)

        @wraps(fn)
        def wrapper(prompt, task=None, output_tokens=None):
            return saver(prompt, task=task, output_tokens=output_tokens)

        wrapper.saver = saver   # access .saver.stats()
        return wrapper
    return deco
