# 🔎 TokenLens

**Count tokens and see what an LLM request costs — *before* you spend.** Plus a
log analyzer that answers the question every team eventually asks: *where is our
token budget actually going?*

```bash
tokenlens count    --text "Summarize this contract in plain English." --output-tokens 200
tokenlens optimize --text "..." --task classify --default-model claude-sonnet
tokenlens lint     --text "..." --task classify --model claude-sonnet   # flag waste before sending
tokenlens guard    --text "..." --model gpt-4o --per-request-limit 0.01  # block over-budget calls
tokenlens analyze  data/sample_log.jsonl
tokenlens complete --text "What is the capital of France?"   # full live pipeline (mock by default)
tokenlens replay   data/sample_log.jsonl                     # show the cache saving money
tokenlens models
```

Runs with **no API key and no downloads** — it ships with a fast offline token
estimator and plugs into the exact `tiktoken` tokenizer with one optional install.

---

## Why this exists

Tokens are an invisible cost. You don't see what a prompt will cost until after
you've sent it, and at scale the bill creeps up with no clear culprit. TokenLens
makes the invisible visible in two ways:

- **Before a call:** paste text, get the token count and the cost on every model,
  side by side, so you can see (for example) that the cheapest option is 25x
  cheaper than the priciest for the same request.
- **After many calls:** point it at a log of requests and it breaks the spend down
  by model and by *type of work* — revealing things like "chat is 2 of 10 requests
  but 75% of the cost." That's the insight that actually saves money.

## What it does

- **Token counting** — exact via `tiktoken` (GPT cl100k reference) when installed,
  or a pure-Python estimate that works offline. It always tells you which it used.
- **Cost-per-model table** — the same request priced across every model in your
  config, cheapest first, with the cost multiplier called out.
- **Token optimization** — `optimize` applies three independent, separately-measured
  levers: **trim** structural waste, **route** to the cheapest adequate model, and
  **cap** output length. Each lever's saving is reported on its own, so you know
  *what* worked, not just that the total went down.
- **Prompt linter** — `lint` flags wasteful patterns *before* you send (oversized
  input, uncapped output, an expensive model on a trivial task) with a concrete fix
  for each. High-severity findings exit non-zero, so it can gate a pipeline.
- **Budget guard** — `guard` estimates a call's cost and **warns or blocks** when it
  would cross a per-request or daily limit, with a local ledger that tracks the
  day's running spend. This is the piece that makes precaution real, not implied.
- **Semantic cache** — skip paying for a request you've effectively answered before.
  Swappable embedder (real sentence-transformers, or an offline fallback), and a
  threshold eval that measures hit rate **and** false-hit rate — because serving a
  confidently-wrong cached answer is the failure that matters.
- **Live pipeline** — `complete` runs a real request through the whole chain:
  optimize the prompt, **guard** the budget (block if it busts a limit), check the
  **semantic cache** (a hit costs nothing), and only call the model on a miss. The
  model call is swappable: an offline mock by default, or the real Anthropic API
  with `--provider anthropic`. `replay` runs a whole log through it and shows the
  cache turning repeat traffic into $0 calls. This is what makes it a *system*, not
  just advice.
- **Log analysis** — totals plus breakdowns by model and by tag, and the most
  expensive requests, so you know exactly what to optimize.
- **Editable pricing** — prices live in `prices.json`, not the code, because rates
  change. Update one file and every number updates.
- **CLI + dashboard + tests + CI.**

## Quickstart

```bash
pip install -e ".[dev,app]"        # or: make install
make test                          # 11 passing tests

tokenlens count --text "your prompt here" --output-tokens 200
tokenlens analyze data/sample_log.jsonl
streamlit run src/tokenlens/app.py
```

For exact token counts (recommended on your own machine, needs internet once):

```bash
pip install -e ".[exact]"
tokenlens count --text "..." --method tiktoken
```

## The request log format

`analyze` reads JSONL where each line is one request. Any of these fields work:

```json
{"id": "r06", "tag": "chat", "model": "claude-sonnet", "text": "...", "output_tokens": 900}
{"id": "r03", "tag": "classify", "model": "gpt-4o-mini", "input_tokens": 12, "output_tokens": 3}
```

Give it `text` and it counts the tokens for you; give it `input_tokens` and it uses
them directly. `tag` lets you group spend by kind of work.

## Use it in your own LLM code

TokenLens isn't just a CLI — it's importable. Wrap the function you already use to
call your model, and every call is optimized, budget-guarded, and cached:

```python
from tokenlens import TokenSaver

def my_llm(prompt, model, max_tokens):
    ...  # your real Anthropic/OpenAI call -> returns the answer string
    return answer

saver = TokenSaver(my_llm, default_model="claude-sonnet", per_request_limit=0.05)

answer = saver("Summarize this contract.", task="summarize")
print(saver.stats())   # {'calls':1, 'cache_hits':0, 'total_saved':..., ...}
```

Or as a decorator:

```python
from tokenlens import save_tokens

@save_tokens(default_model="claude-sonnet", per_request_limit=0.05)
def my_llm(prompt, model, max_tokens):
    return answer
```

And if the *shortage* is context length (prompt too long for the model), shrink it
to fit a token budget:

```python
from tokenlens import fit_context
prompt, trimmed = fit_context(long_prompt, max_tokens=8000)
```

A runnable example lives in [`examples/use_in_your_code.py`](examples/use_in_your_code.py).

## Project layout

```
tokenlens/
├── prices.json                 # editable per-model prices (USD per 1M tokens)
├── data/sample_log.jsonl       # example request log to analyze
├── src/tokenlens/
│   ├── tokenizer.py            # token counting: tiktoken + offline estimate
│   ├── pricing.py              # load prices, cost math
│   ├── analyzer.py             # aggregate a request log
│   ├── optimizer.py            # trim / route / cap - measured per lever
│   ├── linter.py               # flag wasteful patterns before sending
│   ├── budget.py               # warn/block over-budget calls + spend ledger
│   ├── cache.py                # semantic cache (disk-persistent) + threshold eval
│   ├── pipeline.py             # live pipeline: guard -> optimize -> cache -> model
│   ├── integrations.py         # TokenSaver / @save_tokens - drop into your own code
│   ├── report.py               # terminal tables (ASCII)
│   ├── cli.py                  # count / analyze / models
│   └── app.py                  # Streamlit dashboard
├── tests/                      # pytest (tokenizer, pricing, analyzer)
└── .github/workflows/ci.yml    # tests + CLI smoke test on every push
```

## Honest notes

- The offline estimator is an **approximation** (a character-and-word blend
  calibrated to typical English; expect roughly ±15% vs exact BPE). For real
  numbers, install `tiktoken` and use `--method tiktoken`.
- `tiktoken`'s `cl100k_base` is a widely-used **reference** tokenizer (GPT-4/3.5).
  Other model families tokenize a little differently, so treat counts as close
  estimates for cost planning, not a billing guarantee.
- The prices in `prices.json` are **examples and will go stale** — update them to
  current provider rates before trusting the dollar figures. The tooling is the
  point; the numbers are yours to keep current.
- The semantic cache's **offline embedder** (hashing) catches near-identical and
  lightly-varied prompts, not deep paraphrases. Install `sentence-transformers` and
  use the real embedder for semantic matching. Either way, the threshold eval is
  there so you can pick a cutoff from measured hit/false-hit rates rather than guess.
- The live pipeline runs end-to-end **offline with a mock model** (so the repo is
  clonable and testable with no key). The real `--provider anthropic` path is wired
  to the correct API shape; run it once with your own key to confirm your setup —
  "I tested it myself" beats "it should work."

## Roadmap

- A savings dashboard: cumulative cost avoided from cache hits and optimizations.
- TTL expiry on cached answers so stale responses age out.
- More per-model exact tokenizers (not just the GPT reference).
- A FastAPI service exposing the pipeline as an HTTP endpoint other apps can call.

## License

MIT — see [LICENSE](LICENSE).
