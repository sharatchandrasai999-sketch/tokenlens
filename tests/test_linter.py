from tokenlens.linter import lint, worst_severity

PRICES = {"cheap": {"input": 0.15, "output": 0.6}, "dear": {"input": 3.0, "output": 15.0}}

def test_flags_overpowered_model():
    f = lint("classify this", task="classify", output_tokens=10, model="dear",
             prices=PRICES, method="approx")
    assert any(x["code"] == "overpowered_model" and x["severity"] == "high" for x in f)

def test_flags_uncapped_output():
    f = lint("hello", task="chat", output_tokens=0, model="dear",
             prices=PRICES, method="approx")
    assert any(x["code"] == "uncapped_output" for x in f)

def test_flags_filler_and_dupes():
    text = "please note that do it\ndo it\ndo it"
    f = lint(text, task="chat", output_tokens=100, model="dear",
             prices=PRICES, method="approx")
    codes = {x["code"] for x in f}
    assert "filler" in codes and "duplicate_lines" in codes

def test_clean_prompt_no_high():
    f = lint("translate to french", task="summarize", output_tokens=100,
             model="cheap", prices=PRICES, method="approx")
    assert worst_severity(f) != "high"
