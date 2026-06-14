"""TokenLens dashboard.

    streamlit run src/tokenlens/app.py

Tab 1: paste text, see token count + cost across models live.
Tab 2: analyze a request log (uses the bundled sample by default).
"""

import os
import streamlit as st

from tokenlens.tokenizer import count_tokens
from tokenlens.pricing import load_prices, cost_table
from tokenlens.analyzer import analyze
from tokenlens.optimizer import optimize

st.set_page_config(page_title="TokenLens", page_icon="🔎", layout="centered")
st.title("🔎 TokenLens")
st.caption("Count tokens and see what a request costs - before you spend.")

prices = load_prices()

tab_count, tab_analyze, tab_optimize = st.tabs(
    ["Estimate a prompt", "Analyze a log", "Optimize a request"])

with tab_count:
    text = st.text_area("Prompt text", height=180,
                        value="Summarize this 40-page contract into one "
                              "plain-English paragraph for a non-lawyer.")
    out_tokens = st.slider("Assumed output tokens", 0, 2000, 200, 50)
    import importlib.util
    _has_tiktoken = importlib.util.find_spec("tiktoken") is not None
    options = ["auto", "approx"] + (["tiktoken"] if _has_tiktoken else [])
    method = st.radio("Counting method", options, horizontal=True,
                      help="auto uses tiktoken if installed, else a fast estimate."
                           + ("" if _has_tiktoken else
                              " (tiktoken not installed - `pip install tiktoken` to enable)"))
    try:
        tokens, used = count_tokens(text, method=method)
    except Exception:
        tokens, used = count_tokens(text, method="approx")
        st.warning("tiktoken isn't available here - used the offline estimate instead.")
    c1, c2 = st.columns(2)
    c1.metric("Input tokens", f"{tokens:,}")
    c2.metric("Counted via", used)
    if used == "approx":
        st.caption("Estimate (+/- ~15%). Install the `exact` extra for tiktoken counts.")
    table = cost_table(tokens, out_tokens, prices)
    st.markdown("**Cost per model** (cheapest first)")
    st.table([{"model": r["model"],
               "input $": round(r["input_cost"], 6),
               "output $": round(r["output_cost"], 6),
               "total $": round(r["total"], 6)} for r in table])
    cheap, dear = table[0], table[-1]
    if cheap["total"]:
        st.info(f"Cheapest is **{cheap['model']}** at ${cheap['total']:.5f} - "
                f"that's {dear['total']/cheap['total']:.1f}x cheaper than "
                f"{dear['model']} for the same request.")

with tab_analyze:
    default_log = "data/sample_log.jsonl"
    st.write("Analyzing a request log answers: *where is my budget actually going?*")
    path = st.text_input("Log path (JSONL)", value=default_log)
    default_model = st.selectbox("Default model (for entries without one)", list(prices))
    if os.path.exists(path):
        a = analyze(path, prices, default_model=default_model)
        m1, m2, m3 = st.columns(3)
        m1.metric("Requests", a["n_requests"])
        m2.metric("Total tokens", f"{a['total_tokens']:,}")
        m3.metric("Total cost", f"${a['total_cost']:.4f}")
        st.markdown("**By tag** - what kind of work spends the most")
        st.bar_chart({k: v["cost"] for k, v in a["by_tag"].items()})
        st.markdown("**By model**")
        st.table([{"model": k, "requests": v["requests"],
                   "tokens": v["tokens"], "cost $": round(v["cost"], 5)}
                  for k, v in sorted(a["by_model"].items(),
                                     key=lambda kv: kv[1]["cost"], reverse=True)])
        st.markdown("**Most expensive requests**")
        st.table([{"id": r["id"], "tag": r["tag"], "model": r["model"],
                   "tokens": r["total_tokens"], "cost $": round(r["cost"], 5)}
                  for r in a["top_requests"]])
    else:
        st.warning(f"No file at {path}")


with tab_optimize:
    st.write("Three independent levers to cut a request's cost - each measured separately.")
    otext = st.text_area("Request text", height=140,
                         value="Please note that I would like you to classify the "
                               "following support message in order to route it: "
                               "where is my order?")
    oc1, oc2 = st.columns(2)
    task = oc1.selectbox("Task type",
                         ["classify", "extract", "summarize", "chat", "reason", "code"])
    default_model = oc2.selectbox("Model you'd use today", list(prices),
                                  index=len(prices) - 1)
    out_t = st.slider("Assumed output tokens", 0, 2000, 300, 50, key="opt_out")
    if st.button("Optimize", type="primary"):
        o = optimize(otext, task=task, output_tokens=out_t, prices=prices,
                     default_model=default_model)
        t, r, c, tot = o["trim"], o["routing"], o["output_cap"], o["totals"]
        st.markdown(f"**1. Trim waste** - {t['tokens_before']} -> "
                    f"{t['tokens_after']} tokens (saved ${t['saved_cost']:.5f})")
        st.markdown(f"**2. Route model** - {r['default_model']} -> "
                    f"`{r['recommended_model']}` ({r['reason']}); saved ${r['saved_cost']:.5f}")
        st.markdown(f"**3. Cap output** - suggest max {c['suggested_max_tokens']} "
                    f"tokens (saved ${c['saved_cost']:.5f})")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cost before", f"${tot['cost_before']:.5f}")
        m2.metric("Cost after", f"${tot['cost_after']:.5f}")
        m3.metric("Saved", f"{tot['saved_pct']:.1f}%")
        st.caption("Trim is structural (meaning-preserving), so no quality check "
                   "is needed. Semantic compression would need one - that's roadmap.")
