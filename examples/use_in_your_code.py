"""Example: drop TokenLens into your own LLM code.

Run:  python examples/use_in_your_code.py
(Uses a fake model so it runs with no API key. Swap in your real call to go live.)
"""
from tokenlens import TokenSaver, save_tokens, fit_context


# --- your existing model call (replace the body with a real API call) ---------
def my_llm(prompt, model, max_tokens):
    # e.g. anthropic / openai call here; must return the answer string
    return f"[{model}] pretend answer to: {prompt[:40]}"


# --- 1) wrap it -------------------------------------------------------------
saver = TokenSaver(my_llm, default_model="claude-sonnet", per_request_limit=1.0)

print(saver("What is the capital of France?", task="chat"))   # calls the model
print(saver("What is the capital of France?", task="chat"))   # served from cache
print("stats:", saver.stats())


# --- 2) decorator form ------------------------------------------------------
@save_tokens(default_model="claude-sonnet")
def chat(prompt, model, max_tokens):
    return f"[{model}] answer: {prompt[:40]}"

print(chat("Classify this ticket as billing or technical.", task="classify"))
print("decorator stats:", chat.saver.stats())


# --- 3) fit an over-long prompt into a context budget -----------------------
long_prompt = "background. " * 1000 + "QUESTION: summarize the above."
fitted, was_trimmed = fit_context(long_prompt, max_tokens=200)
print(f"fit_context trimmed={was_trimmed}, now ~fits 200 tokens")
