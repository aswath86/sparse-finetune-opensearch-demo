#!/usr/bin/env python3
"""
Step 6: Side-by-side comparison via OpenSearch Predict API.

Calls the deployed base and fine-tuned models through OpenSearch and
displays a formatted comparison showing all activated tokens with
★ NEW, ▲ BOOSTED, and ▼ DROPPED markers.

Usage:
    python demo_compare.py
    python demo_compare.py "my whole family got sick after a party"

What it does:
    - Sends each query to both base and fine-tuned models via _predict API
    - Shows top-28 tokens side by side in columns
    - Marks terms that are NEW (not in base) with ★
    - Marks terms that are BOOSTED (>1.1x) with ▲
    - Marks terms that are DROPPED (<1/1.1x) with ▼
    - Pulls notable terms from below top-N with · · · separator
"""
import json, sys, urllib.request

BASE_ID = "A83I3ZwBkywyXWfOmQhz"
FT_ID = "B83J3ZwBkywyXWfO2whD"
OS_URL = "http://localhost:9202"
TOP_N = 28

THRESHOLD = 0.01
BOOST_RATIO = 1.1

def predict(model_id, text):
    url = f"{OS_URL}/_plugins/_ml/_predict/sparse_encoding/{model_id}"
    data = json.dumps({"text_docs": [text]}).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["inference_results"][0]["output"][0]["dataAsMap"]["response"][0]

def show(query):
    base = predict(BASE_ID, query)
    ft = predict(FT_ID, query)

    base_sorted = sorted(base.items(), key=lambda x: -x[1])[:TOP_N]
    ft_sorted = sorted(ft.items(), key=lambda x: -x[1])[:TOP_N]

    new_terms = {t: ft[t] for t in ft if ft[t] > THRESHOLD and t not in base}
    boosted = {t: (base[t], ft[t]) for t in ft if t in base and ft[t] > base[t] * BOOST_RATIO and ft[t] > THRESHOLD}
    dropped = {t: (base[t], ft[t]) for t in ft if t in base and ft[t] < base[t] / BOOST_RATIO and base[t] > THRESHOLD}

    # Pull new/boosted terms not in top-N into the FT column
    ft_shown = set(t for t, _ in ft_sorted)
    extra = sorted([(t, ft[t]) for t in (set(new_terms) | set(boosted)) - ft_shown], key=lambda x: -x[1])
    if extra:
        ft_sorted = ft_sorted + extra
    ft_sorted = ft_sorted[:TOP_N]

    print(f"\n{'='*70}")
    print(f"  Query: \"{query}\"")
    print(f"{'='*70}")
    print(f"  {'BASE (' + str(len(base)) + ' tokens)':34s} {'FINE-TUNED (' + str(len(ft)) + ' tokens)':34s}")
    print(f"  {'─'*33} {'─'*33}")
    rows = max(len(base_sorted), len(ft_sorted))
    for i in range(rows):
        left = f"  {base_sorted[i][0]:20s} {base_sorted[i][1]:.3f}" if i < len(base_sorted) else " " * 34
        right = ""
        if i < len(ft_sorted):
            t, s = ft_sorted[i]
            marker = " ★" if t in new_terms else " ▲" if t in boosted else " ▼" if t in dropped else ""
            right = f"  {t:20s} {s:.3f}{marker}"
        if i == len(base_sorted) and extra:
            left = "  " + "· " * 16
        print(f"{left:34s}{right}")

    if new_terms:
        top_new = sorted(new_terms.items(), key=lambda x:-x[1])[:10]
        terms = ', '.join(f"{t}({v:.2f})" for t,v in top_new)
        print(f"\n  ★ NEW terms: {terms}")
    if boosted:
        top_boost = sorted(boosted.items(), key=lambda x:-x[1][1])[:10]
        terms = ', '.join(f"{t}({a:.2f}→{b:.2f})" for t,(a,b) in top_boost)
        print(f"  ▲ BOOSTED terms: {terms}")

queries = sys.argv[1:] or [
    "my whole family got sick after a party",
    "when should i get my booster shot",
    "i have a fever and dry cough",
    "still tired months after being sick",
    "how does the virus spread",
    "how to configure nginx reverse proxy",
]
for q in queries:
    show(q)
print()
