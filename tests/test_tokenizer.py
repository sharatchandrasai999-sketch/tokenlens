from tokenlens.tokenizer import approx_tokens, count_tokens

def test_empty_is_zero():
    assert approx_tokens("") == 0

def test_longer_text_more_tokens():
    short = approx_tokens("hello world")
    long = approx_tokens("hello world " * 50)
    assert long > short

def test_at_least_one_token_per_word():
    assert approx_tokens("a b c d e") >= 5

def test_count_tokens_returns_method():
    n, method = count_tokens("some text here", method="approx")
    assert n > 0 and method == "approx"

def test_approx_in_reasonable_range():
    # ~9 words; real BPE is ~11-13 tokens. Estimate should be in a sane band.
    text = "The quick brown fox jumps over the lazy dog."
    n = approx_tokens(text)
    assert 8 <= n <= 18
