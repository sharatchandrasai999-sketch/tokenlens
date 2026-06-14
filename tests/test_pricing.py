import pytest
from tokenlens.pricing import cost, cost_table

PRICES = {
    "cheap": {"input": 1.0, "output": 2.0},
    "dear":  {"input": 10.0, "output": 20.0},
}

def test_cost_math():
    # 1,000,000 input tokens at $1/1M = $1.00 ; 500,000 output at $2/1M = $1.00
    assert cost(1_000_000, 500_000, "cheap", PRICES) == pytest.approx(2.0)

def test_zero_tokens_zero_cost():
    assert cost(0, 0, "cheap", PRICES) == 0.0

def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cost(10, 10, "nope", PRICES)

def test_cost_table_sorted_cheapest_first():
    table = cost_table(1_000_000, 0, PRICES)
    assert table[0]["model"] == "cheap"
    assert table[-1]["model"] == "dear"
