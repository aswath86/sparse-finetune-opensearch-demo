#!/usr/bin/env python3
"""
Step 6: Side-by-side comparison via OpenSearch Predict API.

Calls the deployed base and fine-tuned models through OpenSearch and
displays a compact comparison showing the most interesting tokens:
★ NEW terms, ▲ BOOSTED terms, plus top base terms for context.

Usage:
    python demo_compare.py
    python demo_compare.py "my whole family got sick after a party"
"""
import json, sys, urllib.request

BASE_ID = "A83I3ZwBkywyXWfOmQhz"  # TODO: replace with your base model ID
FT_ID = "B83J3ZwBkywyXWfO2whD"    # TODO: replace with your fine-tuned model ID
OS_URL = "http://localhost:9202"
SHOW_ROWS = 15

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

    new_terms = {t: ft[t] for t in ft if ft[t] > THRESHOLD and t not in base}
    sig_boosted = {t: (base[t], ft[t]) for t in ft if t in base and ft[t] > base[t] * BOOST_RATIO and ft[t] > THRESHOLD}

    def marker_for(t):
        if t in new_terms: return " ★"
        if t not in base: return ""
        ratio = ft[t] / base[t] if base[t] > THRESHOLD else 999
        if ratio > BOOST_RATIO: return " ▲"
        if ratio < 1 / BOOST_RATIO: return " ▼"
        return ""

    # Build FT column: prioritize ★ and ▲ terms, fill rest from top tokens
    notable = sorted(
        [(t, ft[t]) for t in (set(new_terms) | set(sig_boosted))],
        key=lambda x: -x[1]
    )
    ft_top = sorted(ft.items(), key=lambda x: -x[1])
    seen = set(t for t, _ in notable)
    for t, s in ft_top:
        if t not in seen:
            notable.append((t, s))
            seen.add(t)
        if len(notable) >= SHOW_ROWS:
            break
    ft_display = sorted(notable[:SHOW_ROWS], key=lambda x: -x[1])

    base_sorted = sorted(base.items(), key=lambda x: -x[1])

    print(f"\n{'='*70}")
    print(f"  Query: \"{query}\"")
    print(f"{'='*70}")
    print(f"  {'BASE (' + str(len(base)) + ' tokens)':34s} {'FINE-TUNED (' + str(len(ft)) + ' tokens)':34s}")
    print(f"  {'─'*33} {'─'*33}")
    for i in range(SHOW_ROWS):
        left = f"  {base_sorted[i][0]:20s} {base_sorted[i][1]:.3f}" if i < len(base_sorted) else " " * 34
        t, s = ft_display[i]
        right = f"  {t:20s} {s:.3f}{marker_for(t)}"
        print(f"{left:34s}{right}")
    base_more = len(base_sorted) - SHOW_ROWS
    ft_more = len(ft) - SHOW_ROWS
    if base_more > 0 or ft_more > 0:
        left = f"  · · · {base_more} more" if base_more > 0 else ""
        right = f"  · · · {ft_more} more" if ft_more > 0 else ""
        print(f"  {left:32s}  {right}")

    if new_terms:
        top_new = sorted(new_terms.items(), key=lambda x:-x[1])[:5]
        rest = len(new_terms) - len(top_new)
        terms = ', '.join(f"{t}({v:.2f})" for t,v in top_new)
        print(f"\n  ★ {len(new_terms)} NEW: {terms}{'...' if rest else ''}")
    if sig_boosted:
        top_boost = sorted(sig_boosted.items(), key=lambda x:-x[1][1])[:5]
        rest = len(sig_boosted) - len(top_boost)
        terms = ', '.join(f"{t}({a:.2f}→{b:.2f})" for t,(a,b) in top_boost)
        print(f"  ▲ {len(sig_boosted)} BOOSTED: {terms}{'...' if rest else ''}")

queries = sys.argv[1:] or [
    "My whole family got sick after a birthday party last weekend. Everyone had fever and cough for days.",
    "I got my first two vaccine doses but I'm not sure when I should get my booster shot.",
    "I have had a fever and dry cough for three days now and I'm worried it might be something serious.",
    "I recovered from a respiratory infection two months ago but I still feel tired and exhausted all the time.",
    "How does the virus spread from person to person in indoor settings like offices and schools?",
    "Learning to play guitar takes daily practice and patience, starting with basic chords like G, C, and D.",
]
for q in queries:
    show(q)
print()
