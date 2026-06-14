"""Terminal output. All ASCII so it renders on any console."""

from __future__ import annotations


def print_count(text_tokens: int, method: str, output_tokens: int, table: list[dict]):
    print(f"\n  input tokens: {text_tokens}  (counted via {method})")
    print(f"  assumed output tokens: {output_tokens}")
    print("  " + "-" * 52)
    print(f"  {'model':<16}{'input $':>10}{'output $':>11}{'total $':>11}")
    print("  " + "-" * 52)
    for r in table:
        print(f"  {r['model']:<16}{r['input_cost']:>10.5f}"
              f"{r['output_cost']:>11.5f}{r['total']:>11.5f}")
    print("  " + "-" * 52)
    cheapest, dearest = table[0], table[-1]
    if dearest["total"] > 0:
        x = dearest["total"] / cheapest["total"] if cheapest["total"] else 0
        print(f"  cheapest: {cheapest['model']} (${cheapest['total']:.5f})   "
              f"dearest: {dearest['model']} (${dearest['total']:.5f})"
              + (f"  -> {x:.1f}x" if x else ""))
    print()


def _bucket_table(title: str, bucket: dict, total_cost: float):
    print(f"  {title}")
    print("  " + "-" * 52)
    print(f"  {'name':<16}{'reqs':>6}{'tokens':>10}{'cost $':>11}{'%':>7}")
    print("  " + "-" * 52)
    for name, b in sorted(bucket.items(), key=lambda kv: kv[1]["cost"], reverse=True):
        pct = (b["cost"] / total_cost * 100) if total_cost else 0
        print(f"  {name:<16}{b['requests']:>6}{b['tokens']:>10}"
              f"{b['cost']:>11.5f}{pct:>6.0f}%")
    print()


def print_analysis(a: dict):
    print(f"\n  Analyzed {a['n_requests']} requests")
    print("  " + "=" * 52)
    print(f"  total tokens: {a['total_tokens']:,}")
    print(f"  total cost:   ${a['total_cost']:.5f}")
    print(f"  avg cost/req: ${a['avg_cost']:.5f}")
    print("  " + "=" * 52 + "\n")
    _bucket_table("by tag (what kind of work spends the most)", a["by_tag"], a["total_cost"])
    _bucket_table("by model", a["by_model"], a["total_cost"])
    print("  most expensive requests")
    print("  " + "-" * 52)
    for r in a["top_requests"]:
        print(f"  {r['id']:<12} {r['tag']:<12} {r['model']:<14} "
              f"{r['total_tokens']:>7} tok  ${r['cost']:.5f}")
    print()


def print_optimize(o: dict):
    t = o["trim"]
    print(f"\n  Optimizing a '{o['task']}' request")
    print("  " + "=" * 52)
    print("  1) trim waste")
    print(f"       {t['tokens_before']} -> {t['tokens_after']} input tokens "
          f"(saved {t['saved_tokens']}, ${t['saved_cost']:.5f})")
    r = o["routing"]
    print("  2) route to right-sized model")
    print(f"       {r['default_model']} -> {r['recommended_model']}  ({r['reason']})")
    print(f"       saved ${r['saved_cost']:.5f}")
    c = o["output_cap"]
    print("  3) cap output length")
    print(f"       suggested max {c['suggested_max_tokens']} tokens  "
          f"(saved ${c['saved_cost']:.5f})")
    tot = o["totals"]
    print("  " + "-" * 52)
    print(f"  cost before: ${tot['cost_before']:.5f}   "
          f"after: ${tot['cost_after']:.5f}")
    print(f"  total saved: ${tot['saved_cost']:.5f}  ({tot['saved_pct']:.1f}%)")
    print("\n  note: each saving is measured independently; trim is structural")
    print("  (meaning-preserving), so no quality check is needed.\n")
