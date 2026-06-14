from tokenlens.pipeline import Pipeline
from tokenlens.cache import SemanticCache

PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet": {"input": 3.0, "output": 15.0},
}

def _counting_caller():
    state = {"n": 0}
    def call(prompt, model, max_tokens):
        state["n"] += 1
        return f"answer {state['n']}", 5
    return call, state

def test_miss_calls_model_then_hit_does_not():
    call, state = _counting_caller()
    pipe = Pipeline(PRICES, call_model=call, cache=SemanticCache(threshold=0.9))
    r1 = pipe.complete("what is the capital of france", task="chat")
    assert r1["cache_hit"] is False and state["n"] == 1
    r2 = pipe.complete("what is the capital of france", task="chat")
    assert r2["cache_hit"] is True and state["n"] == 1   # NOT called again
    assert r2["saved"] > 0

def test_guard_blocks_and_skips_call():
    call, state = _counting_caller()
    pipe = Pipeline(PRICES, default_model="claude-sonnet", call_model=call,
                    optimize=False, per_request_limit=0.0)
    r = pipe.complete("a very expensive long prompt", task="chat",
                      output_tokens=1000)
    assert r["status"] == "blocked" and state["n"] == 0   # never called the model

def test_stats_accumulate():
    call, _ = _counting_caller()
    pipe = Pipeline(PRICES, call_model=call, cache=SemanticCache(threshold=0.9))
    pipe.complete("hello there", task="chat")
    pipe.complete("hello there", task="chat")
    s = pipe.stats()
    assert s["calls"] == 2 and s["cache_hits"] == 1 and s["total_saved"] > 0
