import json, tempfile, os
from tokenlens.analyzer import analyze

PRICES = {"m": {"input": 1.0, "output": 2.0}}

def _log(lines):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    return path

def test_totals_and_buckets():
    path = _log([
        {"id": "a", "tag": "x", "model": "m", "input_tokens": 1_000_000, "output_tokens": 0},
        {"id": "b", "tag": "y", "model": "m", "input_tokens": 0, "output_tokens": 1_000_000},
    ])
    a = analyze(path, PRICES, default_model="m")
    os.remove(path)
    assert a["n_requests"] == 2
    assert a["total_cost"] == 3.0          # $1 input + $2 output
    assert a["by_tag"]["y"]["cost"] == 2.0
    assert a["top_requests"][0]["id"] == "b"  # most expensive first

def test_text_is_counted_when_no_tokens_given():
    path = _log([{"id": "c", "model": "m", "text": "hello world foo bar"}])
    a = analyze(path, PRICES, default_model="m", method="approx")
    os.remove(path)
    assert a["rows"][0]["input_tokens"] > 0
