#!/usr/bin/env python3
"""
Three-way side-by-side comparison via OpenSearch Predict API.

v2-mini (base) vs v2-mini-FT vs PubMedBERT-FT.
Markers:
  v2-mini-FT (vs base):   ★ NEW  ▲ BOOSTED  ▼ DROPPED
  PubBERT-FT (vs v2-FT):  ★ NEW  ▲ BOOSTED  ▼ DROPPED  ◆ NOT IN BERT VOCAB

Usage:
    python demo_compare.py
    python demo_compare.py "diabetes and covid risk"
"""
import json, sys, urllib.request

V2_ID   = "xIWxe5wBnofIPe4lihO5"
V2FT_ID = "daO3Cp0BnofIPe4lrHb8"
PUB_ID  = "26MSbZ0BnofIPe4lRHZl"
OS_URL  = "http://localhost:9202"
SHOW_ROWS = 18

STOP = frozenset([
    '.', ',', ':', ';', '!', '?', '-', '(', ')', '[', ']', '/',
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'an', 'and', 'but', 'or', 'nor',
    'not', 'so', 'very', 'just', 'about', 'up', 'down',
    'it', 'its', 'he', 'she', 'they', 'we', 'you', 'my', 'your',
    'his', 'her', 'our', 'their', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
    'also', 'if', 'while', 'because', 'until', 'although',
    'bad', 'good', 'mean', 'much', 'many', 'well', 'back',
    'even', 'still', 'way', 'take', 'come', 'make', 'like',
    'long', 'get', 'got', 'go', 'going', 'know', 'think',
    'see', 'look', 'want', 'give', 'use', 'find', 'tell',
    'thing', 'things', 'something', 'enough', 'child', 'school',
])
THRESHOLD = 0.01
BOOST_RATIO = 1.1

_bert_vocab = None

def get_bert_vocab():
    global _bert_vocab
    if _bert_vocab is None:
        from transformers import AutoTokenizer
        _bert_vocab = set(AutoTokenizer.from_pretrained(
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini"
        ).get_vocab().keys())
    return _bert_vocab


def predict(model_id, text):
    url = f"{OS_URL}/_plugins/_ml/_predict/sparse_encoding/{model_id}"
    data = json.dumps({"text_docs": [text]}).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["inference_results"][0]["output"][0]["dataAsMap"]["response"][0]


def is_visible(t):
    return t not in STOP and not t.startswith('##') and len(t) > 1


def top_n(tokens, n=SHOW_ROWS):
    return [(t, v) for t, v in sorted(tokens.items(), key=lambda x: -x[1])
            if is_visible(t) and v > THRESHOLD][:n]


def marker_vs(t, ref, cur):
    """★ NEW, ▲ BOOSTED, ▼ DROPPED vs reference model."""
    if t not in ref or ref.get(t, 0) < THRESHOLD:
        return " ★"
    ratio = cur[t] / ref[t] if ref[t] > THRESHOLD else 999
    if ratio > BOOST_RATIO:
        return " ▲"
    if ratio < 1 / BOOST_RATIO:
        return " ▼"
    return ""


def marker_pub(t, v2ft, pub, bert_vocab):
    """PubBERT-FT vs v2-mini-FT, ◆ for vocab-new."""
    if t not in bert_vocab:
        return " ◆"
    return marker_vs(t, v2ft, pub)


def show(query):
    base = predict(V2_ID, query)
    v2ft = predict(V2FT_ID, query)
    pub  = predict(PUB_ID, query)
    bert_vocab = get_bert_vocab()

    base_top = top_n(base)
    v2ft_top = top_n(v2ft)
    pub_top  = top_n(pub)

    v2ft_new = {t: v2ft[t] for t in v2ft if v2ft[t] > THRESHOLD and is_visible(t)
                and (t not in base or base.get(t, 0) < THRESHOLD)}
    v2ft_boosted = {t: (base[t], v2ft[t]) for t in v2ft if t in base and base[t] > THRESHOLD
                    and v2ft[t] > base[t] * BOOST_RATIO and v2ft[t] > THRESHOLD and is_visible(t)}
    pub_vocab_new = {t: pub[t] for t in pub if pub[t] > THRESHOLD and t not in bert_vocab and is_visible(t)}

    print(f"\n{'='*108}")
    print(f'  "{query[:100]}"')
    print(f"{'='*108}")
    hdr = (f"  {'v2-mini (' + str(len(base)) + ')':35s}"
           f"{'v2-mini-FT (' + str(len(v2ft)) + ')':35s}"
           f"{'PubBERT-FT (' + str(len(pub)) + ')':35s}")
    print(hdr)
    print(f"  {'─'*34} {'─'*34} {'─'*34}")

    for i in range(SHOW_ROWS):
        left = f"  {base_top[i][0]:20s} {base_top[i][1]:.3f}" if i < len(base_top) else " " * 35
        if i < len(v2ft_top):
            t, s = v2ft_top[i]
            mid = f"  {t:20s} {s:.3f}{marker_vs(t, base, v2ft)}"
        else:
            mid = " " * 35
        if i < len(pub_top):
            t, s = pub_top[i]
            right = f"  {t:20s} {s:.3f}{marker_pub(t, v2ft, pub, bert_vocab)}"
        else:
            right = ""
        print(f"{left:35s}{mid:35s}{right}")

    # Summary lines
    parts_v2, parts_pub = [], []
    if v2ft_new:
        top = sorted(v2ft_new.items(), key=lambda x: -x[1])[:5]
        parts_v2.append(f"★ {len(v2ft_new)} NEW: " + ', '.join(f"{t}({v:.2f})" for t, v in top))
    if v2ft_boosted:
        top = sorted(v2ft_boosted.items(), key=lambda x: -x[1][1])[:5]
        parts_v2.append(f"▲ {len(v2ft_boosted)} BOOSTED: " + ', '.join(f"{t}({a:.2f}→{b:.2f})" for t, (a, b) in top))
    if pub_vocab_new:
        top = sorted(pub_vocab_new.items(), key=lambda x: -x[1])[:5]
        parts_pub.append(f"◆ {len(pub_vocab_new)} VOCAB-NEW: " + ', '.join(f"{t}({v:.2f})" for t, v in top))
    if parts_v2 or parts_pub:
        print()
    if parts_v2:
        print(f"  v2-FT:    {' | '.join(parts_v2)}")
    if parts_pub:
        print(f"  PubBERT:  {' | '.join(parts_pub)}")


queries = sys.argv[1:] or [
    "Steroid treatment for severe pneumonia.",
    "Nausea and diarrhea with respiratory infection.",
    "High blood sugar and infection risk.",
    "Patient needs a breathing machine.",
    "Blood clots and clotting problems after viral infection.",
    "Diabetes and covid risk.",
    "How to configure nginx reverse proxy.",  # OOD
]

for q in queries:
    show(q)
print()
