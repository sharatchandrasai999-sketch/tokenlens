"""The live pipeline - this is what makes TokenLens a *system*, not just advice.

Every request flows through the same guarded, optimized, cached path:

    complete(prompt)
      -> optimize   (trim the prompt, route to the right model, cap output)
      -> guard      (estimate cost; BLOCK if it busts the budget - no call made)
      -> cache      (HIT: return the stored answer, pay nothing)
      -> call model (MISS: actually call the LLM, store the answer, record spend)

The model call is injectable:
  - mock_call       : offline, deterministic - the repo runs and tests with no key.
  - make_anthropic_call : real Anthropic API call when you have a key.

Swapping the caller is the only difference between the demo and production; the
guard/optimize/cache logic around it is identical.
"""

from __future__ import annotations

from .tokenizer import count_tokens
from .pricing import cost
from .optimizer import trim_prompt, route, recommend_output_cap
from .budget import check_budget, SpendLedger, BLOCK
from .cache import SemanticCache


# --------------------------------------------------------------------------
# Model callers: (prompt, model, max_tokens) -> (response_text, output_tokens)
# --------------------------------------------------------------------------
def mock_call(prompt: str, model: str, max_tokens: int):
    """Offline stand-in so the pipeline runs with no API key."""
    answer = f"[mock:{model}] answer to: {prompt[:60].strip()}"
    out_tokens = min(max_tokens, count_tokens(answer)[0])
    return answer, out_tokens


def make_anthropic_call(model_map: dict | None = None):
    """Return a real caller backed by the Anthropic API (needs a key)."""
    import anthropic
    client = anthropic.Anthropic()
    model_map = model_map or {
        "claude-haiku": "claude-haiku-4-5-20251001",
        "claude-sonnet": "claude-sonnet-4-6",
    }

    def _call(prompt: str, model: str, max_tokens: int):
        api_model = model_map.get(model, model)
        resp = client.messages.create(
            model=api_model, max_tokens=max(max_tokens, 1),
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.output_tokens

    return _call


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class Pipeline:
    def __init__(self, prices: dict, *, default_model: str = "claude-sonnet",
                 call_model=None, cache: SemanticCache | None = None,
                 ledger: SpendLedger | None = None,
                 per_request_limit: float | None = None,
                 daily_limit: float | None = None,
                 optimize: bool = True, use_cache: bool = True,
                 method: str = "auto"):
        self.prices = prices
        self.default_model = default_model
        self.call_model = call_model or mock_call
        self.cache = cache if cache is not None else SemanticCache()
        self.ledger = ledger
        self.per_request_limit = per_request_limit
        self.daily_limit = daily_limit
        self.optimize = optimize
        self.use_cache = use_cache
        self.method = method
        # cumulative stats
        self.calls = 0
        self.cache_hits = 0
        self.total_cost = 0.0
        self.total_saved = 0.0

    def complete(self, prompt: str, task: str = "chat",
                 output_tokens: int | None = None) -> dict:
        self.calls += 1

        # 1. optimize
        if self.optimize:
            prompt_used = trim_prompt(prompt)
            in_tokens, _ = count_tokens(prompt_used, method=self.method)
            model, _ = route(task, self.prices, input_tokens=in_tokens)
            cap = recommend_output_cap(task)
            max_out = min(output_tokens, cap) if output_tokens else cap
        else:
            prompt_used = prompt
            in_tokens, _ = count_tokens(prompt_used, method=self.method)
            model = self.default_model
            max_out = output_tokens or recommend_output_cap(task)

        est_cost = cost(in_tokens, max_out, model, self.prices) \
            if model in self.prices else 0.0

        # 2. guard
        spent = self.ledger.spent_today() if self.ledger else 0.0
        guard = check_budget(est_cost, per_request_limit=self.per_request_limit,
                             spent_today=spent, daily_limit=self.daily_limit)
        if guard["decision"] == BLOCK:
            return {"status": "blocked", "reason": guard["reason"],
                    "model": model, "estimated_cost": est_cost,
                    "response": None, "cache_hit": False}

        # 3. cache
        if self.use_cache:
            hit, cached, sim = self.cache.get(prompt_used)
            if hit:
                self.cache_hits += 1
                self.total_saved += est_cost
                return {"status": "ok", "response": cached, "cache_hit": True,
                        "similarity": round(sim, 3), "model": model,
                        "cost": 0.0, "saved": est_cost, "guard": guard["decision"]}

        # 4. real (or mock) call
        response, out_tokens = self.call_model(prompt_used, model, max_out)
        real_cost = cost(in_tokens, out_tokens, model, self.prices) \
            if model in self.prices else 0.0
        self.total_cost += real_cost
        if self.use_cache:
            self.cache.put(prompt_used, response)
        if self.ledger:
            self.ledger.record(real_cost)

        return {"status": "ok", "response": response, "cache_hit": False,
                "model": model, "input_tokens": in_tokens,
                "output_tokens": out_tokens, "cost": real_cost,
                "saved": 0.0, "guard": guard["decision"]}

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "hit_rate": round(self.cache_hits / self.calls, 3) if self.calls else 0,
                "total_cost": round(self.total_cost, 6),
                "total_saved": round(self.total_saved, 6)}
