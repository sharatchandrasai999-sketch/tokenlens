"""TokenLens CLI.

    tokenlens count   --text "..." [--output-tokens 200] [--method auto|approx|tiktoken]
    tokenlens count   --file prompt.txt
    tokenlens analyze data/sample_log.jsonl [--default-model gpt-4o-mini]
    tokenlens models
"""

from __future__ import annotations
import argparse

from . import __version__
from .tokenizer import count_tokens
from .pricing import load_prices, cost_table, cost
from .analyzer import analyze
from .optimizer import optimize
from .budget import check_budget, SpendLedger
from .linter import lint, worst_severity
from .pipeline import Pipeline, mock_call, make_anthropic_call
from .report import print_count, print_analysis, print_optimize


def cmd_count(args):
    text = _text_from(args)
    tokens, method = count_tokens(text, method=args.method)
    prices = load_prices(args.prices)
    table = cost_table(tokens, args.output_tokens, prices)
    print_count(tokens, method, args.output_tokens, table)


def cmd_analyze(args):
    prices = load_prices(args.prices)
    a = analyze(args.log, prices, default_model=args.default_model, method=args.method)
    print_analysis(a)


def cmd_optimize(args):
    text = _text_from(args)
    prices = load_prices(args.prices)
    o = optimize(text, task=args.task, output_tokens=args.output_tokens,
                 prices=prices, default_model=args.default_model, method=args.method)
    print_optimize(o)


def _text_from(args):
    import sys
    if getattr(args, "file", None):
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = args.text or ""
    if not text.strip():
        print("warning: empty input - nothing to analyze "
              "(did you forget --text or --file?)", file=sys.stderr)
    return text


def cmd_guard(args):
    import sys
    text = _text_from(args)
    prices = load_prices(args.prices)
    tokens, _ = count_tokens(text, method=args.method)
    c = cost(tokens, args.output_tokens, args.model, prices)
    ledger = SpendLedger()
    spent = ledger.spent_today()
    res = check_budget(c, per_request_limit=args.per_request_limit,
                       spent_today=spent, daily_limit=args.daily_limit)
    print(f"\n  estimated cost: ${c:.5f}   spent today: ${spent:.5f}")
    print(f"  [{res['decision']}] {res['reason']}\n")
    if res["decision"] != "BLOCK" and args.record:
        new_total = ledger.record(c)
        print(f"  recorded - daily total now ${new_total:.5f}\n")
    if res["decision"] == "BLOCK":
        sys.exit(2)


def cmd_lint(args):
    import sys
    text = _text_from(args)
    prices = load_prices(args.prices)
    findings = lint(text, task=args.task, output_tokens=args.output_tokens,
                    model=args.model, prices=prices, method=args.method)
    print(f"\n  linting a '{args.task}' request for '{args.model}'")
    print("  " + "-" * 52)
    if not findings:
        print("  no issues - looks lean.\n")
        return
    tag = {"high": "[HIGH]", "med": "[MED ]", "low": "[LOW ]"}
    for f in findings:
        print(f"  {tag[f['severity']]} {f['message']}")
        print(f"           -> {f['suggestion']}")
    print()
    if worst_severity(findings) == "high":
        sys.exit(1)   # let it act as a pre-send / CI gate


def _make_caller(provider):
    if provider == "anthropic":
        return make_anthropic_call()
    return mock_call


def cmd_complete(args):
    import os
    from .cache import SemanticCache
    text = _text_from(args)
    prices = load_prices(args.prices)
    cache_path = None if args.no_cache else os.path.join(
        os.path.expanduser("~"), ".tokenlens_cache.json")
    pipe = Pipeline(prices, default_model=args.default_model,
                    call_model=_make_caller(args.provider),
                    cache=SemanticCache(store_path=cache_path),
                    per_request_limit=args.per_request_limit,
                    optimize=not args.no_optimize, use_cache=not args.no_cache)
    r = pipe.complete(text, task=args.task, output_tokens=args.output_tokens or None)
    print()
    if r["status"] == "blocked":
        print(f"  [BLOCKED] {r['reason']}\n")
        return
    tag = "CACHE HIT" if r["cache_hit"] else "called model"
    print(f"  [{tag}]  model: {r['model']}  cost: ${r.get('cost', 0):.5f}"
          + (f"  saved: ${r['saved']:.5f}" if r["cache_hit"] else ""))
    print(f"  response: {r['response']}\n")


def cmd_replay(args):
    import json
    prices = load_prices(args.prices)
    pipe = Pipeline(prices, default_model=args.default_model,
                    call_model=_make_caller(args.provider))
    entries = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    print(f"\n  replaying {len(entries)} requests x {args.passes} pass(es) "
          f"through guard -> optimize -> cache -> {args.provider}")
    print("  " + "-" * 52)
    for p in range(args.passes):
        hits_before = pipe.cache_hits
        for e in entries:
            pipe.complete(e.get("text", ""), task=e.get("tag", "chat"),
                          output_tokens=e.get("output_tokens"))
        s = pipe.stats()
        print(f"  after pass {p+1}: cache hits {s['cache_hits']}  "
              f"(+{s['cache_hits']-hits_before})  spent ${s['total_cost']:.5f}  "
              f"saved ${s['total_saved']:.5f}")
    s = pipe.stats()
    print("  " + "-" * 52)
    print(f"  total: {s['calls']} calls, {s['hit_rate']*100:.0f}% served from cache, "
          f"${s['total_saved']:.5f} saved\n")


def cmd_models(args):
    prices = load_prices(args.prices)
    print("\n  known models (USD per 1M tokens)")
    print("  " + "-" * 40)
    for m, r in prices.items():
        print(f"  {m:<16} in ${r['input']:<7} out ${r['output']}")
    print("\n  edit prices.json to update these.\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tokenlens",
                                description="Count tokens and see LLM cost before you spend.")
    p.add_argument("--prices", default=None, help="path to prices.json")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count", help="count tokens + cost for a piece of text")
    c.add_argument("--text", default=None)
    c.add_argument("--file", default=None, help="read text from a file instead")
    c.add_argument("--output-tokens", type=int, default=0,
                   help="assumed output length, for output cost")
    c.add_argument("--method", default="auto", choices=["auto", "approx", "tiktoken"])
    c.set_defaults(func=cmd_count)

    a = sub.add_parser("analyze", help="analyze a JSONL request log")
    a.add_argument("log")
    a.add_argument("--default-model", default="gpt-4o-mini")
    a.add_argument("--method", default="auto", choices=["auto", "approx", "tiktoken"])
    a.set_defaults(func=cmd_analyze)

    m = sub.add_parser("models", help="list known models and prices")
    m.set_defaults(func=cmd_models)

    g = sub.add_parser("guard", help="warn/block before an over-budget request")
    g.add_argument("--text", default=None)
    g.add_argument("--file", default=None)
    g.add_argument("--model", default="gpt-4o")
    g.add_argument("--output-tokens", type=int, default=300)
    g.add_argument("--per-request-limit", type=float, default=None)
    g.add_argument("--daily-limit", type=float, default=None)
    g.add_argument("--record", action="store_true",
                   help="add this request's cost to today's running total")
    g.add_argument("--method", default="auto", choices=["auto", "approx", "tiktoken"])
    g.set_defaults(func=cmd_guard)

    ln = sub.add_parser("lint", help="flag wasteful patterns before you send")
    ln.add_argument("--text", default=None)
    ln.add_argument("--file", default=None)
    ln.add_argument("--task", default="chat")
    ln.add_argument("--model", default="gpt-4o")
    ln.add_argument("--output-tokens", type=int, default=0)
    ln.add_argument("--method", default="auto", choices=["auto", "approx", "tiktoken"])
    ln.set_defaults(func=cmd_lint)

    cp = sub.add_parser("complete", help="run a prompt through the full live pipeline")
    cp.add_argument("--text", default=None)
    cp.add_argument("--file", default=None)
    cp.add_argument("--task", default="chat")
    cp.add_argument("--default-model", default="claude-sonnet")
    cp.add_argument("--output-tokens", type=int, default=0)
    cp.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    cp.add_argument("--per-request-limit", type=float, default=None)
    cp.add_argument("--no-optimize", action="store_true")
    cp.add_argument("--no-cache", action="store_true")
    cp.set_defaults(func=cmd_complete)

    rp = sub.add_parser("replay", help="run a log through the pipeline to show cache savings")
    rp.add_argument("log")
    rp.add_argument("--default-model", default="claude-sonnet")
    rp.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    rp.add_argument("--passes", type=int, default=2)
    rp.set_defaults(func=cmd_replay)

    o = sub.add_parser("optimize", help="trim, route, and cap a request to cut cost")
    o.add_argument("--text", default=None)
    o.add_argument("--file", default=None, help="read text from a file instead")
    o.add_argument("--task", default="chat",
                   help="classify, extract, summarize, chat, reason, code, ...")
    o.add_argument("--output-tokens", type=int, default=0)
    o.add_argument("--default-model", default="claude-sonnet",
                   help="the model you'd use today (the baseline to beat)")
    o.add_argument("--method", default="auto", choices=["auto", "approx", "tiktoken"])
    o.set_defaults(func=cmd_optimize)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
