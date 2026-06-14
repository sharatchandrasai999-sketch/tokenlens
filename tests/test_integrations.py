from tokenlens import TokenSaver, save_tokens, BudgetExceeded, fit_context
from tokenlens.tokenizer import approx_tokens

PRICES = {"gpt-4o-mini": {"input": 0.15, "output": 0.6},
          "claude-sonnet": {"input": 3.0, "output": 15.0}}

def test_saver_caches_repeat_call():
    calls = {"n": 0}
    def llm(prompt, model, max_tokens):
        calls["n"] += 1
        return "answer"
    saver = TokenSaver(llm, prices=PRICES, threshold=0.9)
    saver("what is the capital of france", task="chat")
    saver("what is the capital of france", task="chat")
    assert calls["n"] == 1                      # second served from cache
    assert saver.stats()["cache_hits"] == 1

def test_saver_accepts_simple_one_arg_function():
    def llm(prompt):
        return "ok"
    saver = TokenSaver(llm, prices=PRICES)
    assert saver("hello", task="chat") == "ok"

def test_budget_block_raises():
    def llm(prompt, model, max_tokens):
        return "x"
    saver = TokenSaver(llm, prices=PRICES, default_model="claude-sonnet",
                       optimize=False, per_request_limit=0.0)
    try:
        saver("an expensive prompt", task="chat", output_tokens=1000)
        assert False, "should have raised"
    except BudgetExceeded:
        pass

def test_decorator_form_works():
    @save_tokens(prices=PRICES, threshold=0.9)
    def llm(prompt, model, max_tokens):
        return "hi"
    assert llm("hey there", task="chat") == "hi"
    assert llm.saver.stats()["calls"] == 1

def test_fit_context_shrinks_long_text():
    long = "word " * 1000
    fitted, trimmed = fit_context(long, max_tokens=100, method="approx")
    assert trimmed and approx_tokens(fitted) <= 160   # near budget (+marker slack)

def test_fit_context_leaves_short_text():
    fitted, trimmed = fit_context("short text", max_tokens=100, method="approx")
    assert not trimmed and fitted == "short text"
