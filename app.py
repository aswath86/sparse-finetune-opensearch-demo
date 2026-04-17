"""
Sparse Fine-tuning Demo
Two tabs: Predict API (OpenSearch) and Local Probe (HuggingFace).

Usage:
    streamlit run app.py
"""
import json
import urllib.request
import base64
import pathlib
import streamlit as st
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# --- Config ---
OS_URL = "http://localhost:9202"
API_MODELS = {
    "v2-mini":       "xIWxe5wBnofIPe4lihO5",
    "v2-mini-FT":    "daO3Cp0BnofIPe4lrHb8",
    "PubMedBERT-FT": "26MSbZ0BnofIPe4lRHZl",
}
LOCAL_MODELS = {
    "v2-mini":       "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    "v2-mini-FT":    "../sparse-finetune-oscon-demov3/finetuned_model",
    "PubMedBERT-FT": "pubmedbert_finetuned_30ep_v3",
}
FT_PRESETS = [
    "how does the virus spread",
    "how long am i contagious",
    "can you get sick twice",
    "lost my sense of smell after being sick",
    "my whole family got sick after a party",
    "how to configure nginx reverse proxy",
]
PUBMED_PRESETS = [
    "COPD symptoms and treatment",
    "side effects of corticosteroids",
    "gastrointestinal symptoms after infection",
    "antimicrobial resistance in hospitals",
    "why do diabetics get infections easily",
    "my cholesterol is too high",
    "patient on a ventilator in the ICU",
    "how to configure nginx reverse proxy",
]
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
TOP_N = 18

# --- Colors ---
C_NEW     = "#2ecc71"
C_BOOST   = "#3498db"
C_DROP    = "#e74c3c"
C_VOCAB   = "#f39c12"
C_NEUTRAL = "#95a5a6"


# --- Shared helpers ---
@st.cache_resource
def load_bert_vocab():
    tok = AutoTokenizer.from_pretrained(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini"
    )
    return set(tok.get_vocab().keys())


@st.cache_resource
def load_local_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path).eval()
    return tok, model


def filter_tokens(tokens):
    return {t: v for t, v in tokens.items()
            if t not in STOP and (t.startswith('##') or len(t) > 1) and v > THRESHOLD}


def classify_token(t, ref, cur, bert_vocab=None):
    if not ref:
        return "", C_NEUTRAL
    if bert_vocab is not None and t not in bert_vocab:
        return "◆ VOCAB-NEW", C_VOCAB
    if t not in ref or ref.get(t, 0) < THRESHOLD:
        return "★ NEW", C_NEW
    ratio = cur[t] / ref[t] if ref[t] > THRESHOLD else 999
    if ratio > BOOST_RATIO:
        return "▲ BOOSTED", C_BOOST
    if ratio < 1 / BOOST_RATIO:
        return "▼ DROPPED", C_DROP
    return "", C_NEUTRAL


def render_token_column(title, tokens_filtered, total_count, ref_tokens, bert_vocab=None, show_all=False):
    st.markdown(f"**{title}** ({total_count} tokens)")
    limit = len(tokens_filtered) if show_all else TOP_N
    sorted_tokens = sorted(tokens_filtered.items(), key=lambda x: -x[1])[:limit]
    if not sorted_tokens:
        st.caption("No tokens above threshold")
        return

    max_weight = sorted_tokens[0][1]
    for token, weight in sorted_tokens:
        label, color = classify_token(token, ref_tokens, tokens_filtered, bert_vocab)
        bar_width = int((weight / max_weight) * 100)
        badge = f'<span style="color:{color};font-weight:bold">{label}</span>' if label else ""
        st.markdown(
            f'<div style="font-family:monospace;font-size:14px;margin:2px 0">'
            f'<span style="display:inline-block;width:160px">{token}</span>'
            f'<span style="display:inline-block;width:50px;text-align:right">{weight:.3f}</span> '
            f'<span style="display:inline-block;width:{bar_width}px;height:12px;'
            f'background:{color};border-radius:2px;vertical-align:middle"></span> '
            f'{badge}</div>',
            unsafe_allow_html=True,
        )

    # Summary badges
    full_stats = {"★ NEW": 0, "▲ BOOSTED": 0, "◆ VOCAB-NEW": 0}
    for t, v in tokens_filtered.items():
        lbl, _ = classify_token(t, ref_tokens, tokens_filtered, bert_vocab)
        if lbl in full_stats:
            full_stats[lbl] += 1

    badges = []
    if full_stats["◆ VOCAB-NEW"]:
        badges.append(f'<span style="background:{C_VOCAB};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">◆ {full_stats["◆ VOCAB-NEW"]} vocab-new</span>')
    if full_stats["★ NEW"]:
        badges.append(f'<span style="background:{C_NEW};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">★ {full_stats["★ NEW"]} new</span>')
    if full_stats["▲ BOOSTED"]:
        badges.append(f'<span style="background:{C_BOOST};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">▲ {full_stats["▲ BOOSTED"]} boosted</span>')
    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)


# --- API tab helpers ---
def predict_api(model_id, text):
    url = f"{OS_URL}/_plugins/_ml/_predict/sparse_encoding/{model_id}"
    data = json.dumps({"text_docs": [text]}).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["inference_results"][0]["output"][0]["dataAsMap"]["response"][0]


# --- Local probe helpers ---
def sparse_encode_local(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs).logits
    vec = torch.max(
        torch.log1p(torch.relu(out)) * inputs["attention_mask"].unsqueeze(-1), dim=1
    ).values.squeeze()
    ids = (vec > THRESHOLD).nonzero(as_tuple=True)[0]
    tokens = {tokenizer.convert_ids_to_tokens(i.item()): round(vec[i].item(), 4) for i in ids}
    return tokens


# --- App ---
st.set_page_config(page_title="Sparse Fine-tuning Demo", layout="wide")

_logo_b64 = base64.b64encode(pathlib.Path("opensearch_logo.svg").read_bytes()).decode()
st.markdown(
    f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">'
    f'<img src="data:image/svg+xml;base64,{_logo_b64}" width="60">'
    f'<div>'
    f'<h2 style="margin:0;padding:0;line-height:1.2">Fine-tuning Neural Sparse Model</h2>'
    f'<span style="color:gray;font-size:16px">Domain-specific data from an existing OpenSearch index</span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

bert_vocab = load_bert_vocab()

tab_probe, tab_api = st.tabs(["Local Probe", "Predict API"])

# ===== Tab 1: Predict API =====
with tab_api:
    col_query, col_toggle = st.columns([3, 1])
    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)
        show_pub_api = st.toggle("Show PubMedBERT-FT", value=False, key="api_pub")
        show_all_api = st.toggle("Show all tokens", value=False, key="api_all")
    with col_query:
        def _set_api(q): st.session_state.api_query = q
        st.caption("Fine-tuning")
        c1 = st.columns(len(FT_PRESETS))
        for i, pq in enumerate(FT_PRESETS):
            c1[i].button(pq[:30] + "…" if len(pq) > 30 else pq, key=f"api_ft_{i}", on_click=_set_api, args=(pq,))
        st.caption("Vocabulary")
        c2 = st.columns(len(PUBMED_PRESETS))
        for i, pq in enumerate(PUBMED_PRESETS):
            c2[i].button(pq[:30] + "…" if len(pq) > 30 else pq, key=f"api_voc_{i}", on_click=_set_api, args=(pq,))
        query_api = st.text_input("Query", value=FT_PRESETS[0], key="api_query")

    if query_api:
        base_t = predict_api(API_MODELS["v2-mini"], query_api)
        ft_t = predict_api(API_MODELS["v2-mini-FT"], query_api)
        base_f = filter_tokens(base_t)
        ft_f = filter_tokens(ft_t)

        if show_pub_api:
            pub_t = predict_api(API_MODELS["PubMedBERT-FT"], query_api)
            pub_f = filter_tokens(pub_t)
            cols = st.columns(3)
        else:
            cols = st.columns(2)

        with cols[0]:
            render_token_column("v2-mini (base)", base_f, len(base_t), {}, show_all=show_all_api)
        with cols[1]:
            render_token_column("v2-mini-FT", ft_f, len(ft_t), base_t, show_all=show_all_api)
        if show_pub_api:
            with cols[2]:
                render_token_column("PubMedBERT-FT", pub_f, len(pub_t), ft_t, bert_vocab, show_all=show_all_api)

# ===== Tab 2: Local Probe =====
with tab_probe:
    col_query2, col_toggle2 = st.columns([3, 1])
    with col_toggle2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_pub_probe = st.toggle("Show PubMedBERT-FT", value=False, key="probe_pub")
        show_all_probe = st.toggle("Show all tokens", value=False, key="probe_all")
    with col_query2:
        def _set_probe(q): st.session_state.probe_query = q
        st.caption("Fine-tuning")
        c3 = st.columns(len(FT_PRESETS))
        for i, pq in enumerate(FT_PRESETS):
            c3[i].button(pq[:30] + "…" if len(pq) > 30 else pq, key=f"probe_ft_{i}", on_click=_set_probe, args=(pq,))
        st.caption("Vocabulary")
        c4 = st.columns(len(PUBMED_PRESETS))
        for i, pq in enumerate(PUBMED_PRESETS):
            c4[i].button(pq[:30] + "…" if len(pq) > 30 else pq, key=f"probe_voc_{i}", on_click=_set_probe, args=(pq,))
        query_probe = st.text_input("Query", value=FT_PRESETS[0], key="probe_query")

    if query_probe:
        with st.spinner("Running inference..."):
            tok_v2, m_v2 = load_local_model(LOCAL_MODELS["v2-mini"])
            tok_ft, m_ft = load_local_model(LOCAL_MODELS["v2-mini-FT"])
            base_t = sparse_encode_local(m_v2, tok_v2, query_probe)
            ft_t = sparse_encode_local(m_ft, tok_ft, query_probe)
            base_f = filter_tokens(base_t)
            ft_f = filter_tokens(ft_t)

            if show_pub_probe:
                tok_pub, m_pub = load_local_model(LOCAL_MODELS["PubMedBERT-FT"])
                pub_t = sparse_encode_local(m_pub, tok_pub, query_probe)
                pub_f = filter_tokens(pub_t)
                cols = st.columns(3)
            else:
                cols = st.columns(2)

        with cols[0]:
            render_token_column("v2-mini (base)", base_f, len(base_t), {}, show_all=show_all_probe)
        with cols[1]:
            render_token_column("v2-mini-FT", ft_f, len(ft_t), base_t, show_all=show_all_probe)
        if show_pub_probe:
            with cols[2]:
                render_token_column("PubMedBERT-FT", pub_f, len(pub_t), ft_t, bert_vocab, show_all=show_all_probe)
