"""Pricing and cost math.

Prices live in an editable JSON file (prices.json), NOT hard-coded, because
provider rates change. Costs are computed per 1,000,000 tokens, the unit most
providers quote.
"""

from __future__ import annotations
import os
import json

DEFAULT_PRICES_PATH = os.environ.get(
    "TOKENLENS_PRICES",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prices.json"),
)


def load_prices(path: str | None = None) -> dict:
    path = path or DEFAULT_PRICES_PATH
    with open(path) as fh:
        data = json.load(fh)
    return data["models"]


def cost(input_tokens: int, output_tokens: int, model: str, prices: dict) -> float:
    """Cost in USD for a single request given a model's per-1M rates."""
    if model not in prices:
        raise KeyError(f"Unknown model '{model}'. Known: {sorted(prices)}")
    rate = prices[model]
    return (input_tokens / 1_000_000) * rate["input"] + \
           (output_tokens / 1_000_000) * rate["output"]


def cost_table(input_tokens: int, output_tokens: int, prices: dict) -> list[dict]:
    """Cost of the same request across every known model, cheapest first."""
    rows = []
    for model in prices:
        c = cost(input_tokens, output_tokens, model, prices)
        rows.append({
            "model": model,
            "input_cost": (input_tokens / 1_000_000) * prices[model]["input"],
            "output_cost": (output_tokens / 1_000_000) * prices[model]["output"],
            "total": c,
        })
    rows.sort(key=lambda r: r["total"])
    return rows
