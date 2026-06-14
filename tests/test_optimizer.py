from tokenlens.optimizer import trim_prompt, route, recommend_output_cap, optimize

PRICES = {
    "cheap": {"input": 0.15, "output": 0.60},
    "mid":   {"input": 2.50, "output": 10.0},
    "dear":  {"input": 3.00, "output": 15.0},
}

def test_trim_removes_whitespace_and_dupes():
    messy = "please note that  the   sky\nthe sky\n\n\n\nis blue"
    out = trim_prompt(messy)
    assert "  " not in out                 # collapsed spaces
    assert out.count("the sky") == 1       # duplicate line removed
    assert "please note that" not in out   # filler removed

def test_trim_is_idempotent():
    once = trim_prompt("in order to  test\ntest")
    assert trim_prompt(once) == once

def test_route_easy_task_picks_cheapest():
    model, _ = route("classify", PRICES)
    assert model == "cheap"

def test_route_hard_task_picks_priciest():
    model, _ = route("chat", PRICES)
    assert model == "dear"

def test_long_input_bumps_low_tier_up():
    model, _ = route("classify", PRICES, input_tokens=10_000)
    assert model == "mid"

def test_output_cap_lookup():
    assert recommend_output_cap("classify") < recommend_output_cap("chat")

def test_optimize_savings_non_negative_and_separated():
    text = "please note that " + "summarize this document in detail. " * 5
    o = optimize(text, task="classify", output_tokens=500, prices=PRICES,
                 default_model="dear", method="approx")
    assert o["trim"]["saved_tokens"] >= 0
    assert o["routing"]["recommended_model"] == "cheap"
    assert o["totals"]["saved_cost"] >= 0
    # each lever reported separately
    assert "saved_cost" in o["trim"] and "saved_cost" in o["routing"]
