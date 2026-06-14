"""Budget guard - turn "here's the cost" into "stop, that's too expensive."

Two kinds of limit:
  - per-request : block any single call over a ceiling
  - daily       : block once the running daily total would cross a ceiling

A small JSON ledger tracks spend per day so the daily limit actually persists
across CLI calls. Everything is local and offline.

Decisions: OK (under thresholds), WARN (in the warning band), BLOCK (over a hard
limit). Safe-by-default: when a limit is set and a call would exceed it, the guard
says BLOCK - the caller decides whether to honor it.
"""

from __future__ import annotations
import os
import json
from datetime import date

OK, WARN, BLOCK = "OK", "WARN", "BLOCK"

# Warn once you cross this fraction of a limit (e.g. 0.8 = warn at 80%).
WARN_FRACTION = 0.8

DEFAULT_LEDGER = os.environ.get(
    "TOKENLENS_LEDGER", os.path.join(os.path.expanduser("~"), ".tokenlens_ledger.json"))


def check_budget(cost: float, *, per_request_limit: float | None = None,
                 spent_today: float = 0.0,
                 daily_limit: float | None = None) -> dict:
    """Return {decision, reason} for a request of the given cost."""
    # Hard per-request ceiling.
    if per_request_limit is not None and cost > per_request_limit:
        return {"decision": BLOCK,
                "reason": f"request ${cost:.5f} exceeds per-request limit "
                          f"${per_request_limit:.5f}"}
    # Daily ceiling (projected).
    if daily_limit is not None:
        projected = spent_today + cost
        if projected > daily_limit:
            return {"decision": BLOCK,
                    "reason": f"would reach ${projected:.5f} today, over daily "
                              f"limit ${daily_limit:.5f}"}
        if projected > daily_limit * WARN_FRACTION:
            return {"decision": WARN,
                    "reason": f"${projected:.5f} of ${daily_limit:.5f} daily "
                              f"budget used ({projected/daily_limit*100:.0f}%)"}
    # Per-request warning band.
    if per_request_limit is not None and cost > per_request_limit * WARN_FRACTION:
        return {"decision": WARN,
                "reason": f"request ${cost:.5f} is near the per-request limit "
                          f"${per_request_limit:.5f}"}
    return {"decision": OK, "reason": "within budget"}


class SpendLedger:
    """Tiny per-day spend tracker persisted to a JSON file."""

    def __init__(self, path: str | None = None):
        self.path = path or DEFAULT_LEDGER
        self._data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path) as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def spent_today(self) -> float:
        return float(self._data.get(str(date.today()), 0.0))

    def record(self, cost: float) -> float:
        today = str(date.today())
        self._data[today] = self.spent_today() + cost
        with open(self.path, "w") as fh:
            json.dump(self._data, fh)
        return self._data[today]
