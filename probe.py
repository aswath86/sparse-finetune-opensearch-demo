#!/usr/bin/env python3
"""
Three-way comparison: v2-mini vs v2-mini-FT vs PubMedBERT-FT.

Same fine-tuning, different base model. Shows that PubMedBERT-FT activates
biomedical tokens that v2-mini-FT cannot, because PubMedBERT was pre-trained
on PubMed and has a domain-specific vocabulary.

Usage:
    python probe.py
    python probe.py --v2ft ../sparse-finetune-oscon-demov3/finetuned_model --pubft pubmedbert_finetuned_30ep_v3
"""
import torch, argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer

V2_MINI = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini"
TOP_N = 8

STOP = frozenset(['.', ',', ':', ';', '!', '?', '-', '(', ')', '[', ']', '/'])

QUERIES = [
    "steroid treatment for severe pneumonia",           # ★ steroids, corticosteroid, immunosuppressive, icu
    "nausea and diarrhea with respiratory infection",    # ★ diarrhea, gastrointestinal, intestinal, digestive
    "high blood sugar and infection risk",               # ★ diabetic, mellitus, cholesterol, carbohydrate
    "patient needs a breathing machine",                 # ★ ventilator, icu
    "blood clots and clotting problems after viral infection",  # ★ anticoagulation, platelet
    "diabetes and covid risk",                           # ★ diabetic, mellitus, bmi, cortisol
    "how to configure nginx reverse proxy",              # OOD control
]

THRESHOLD = 0.01


def sparse_encode(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs).logits
    vec = torch.max(torch.log1p(torch.relu(out)) * inputs["attention_mask"].unsqueeze(-1), dim=1).values.squeeze()
    return vec


def top_tokens(vec, tokenizer, n=TOP_N):
    ids = (vec > THRESHOLD).nonzero(as_tuple=True)[0]
    hits = [(tokenizer.convert_ids_to_tokens(i.item()), round(vec[i].item(), 3)) for i in ids]
    hits = [(t, v) for t, v in hits if t not in STOP and not t.startswith('##') and len(t) > 1]
    hits.sort(key=lambda x: -x[1])
    return hits[:n], hits  # top-N for display, all for analysis


def fmt(hits, bert_vocab=None):
    if not hits:
        return "NONE"
    parts = []
    for t, v in hits:
        star = "★" if bert_vocab is not None and t not in bert_vocab else ""
        parts.append(f"{t}{star}({v:.2f})")
    return ", ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2ft", default="../sparse-finetune-oscon-demov3/finetuned_model")
    p.add_argument("--pubft", default="pubmedbert_finetuned_30ep_v3")
    p.add_argument("--top", type=int, default=TOP_N)
    args = p.parse_args()

    print("Loading models...")
    tok_v2 = AutoTokenizer.from_pretrained(V2_MINI)
    m_v2 = AutoModelForMaskedLM.from_pretrained(V2_MINI).eval()
    bert_vocab = set(tok_v2.get_vocab().keys())

    tok_v2ft = AutoTokenizer.from_pretrained(args.v2ft)
    m_v2ft = AutoModelForMaskedLM.from_pretrained(args.v2ft).eval()

    tok_pub = AutoTokenizer.from_pretrained(args.pubft)
    m_pub = AutoModelForMaskedLM.from_pretrained(args.pubft).eval()

    print(f"(★ = token not in BERT vocab)\n")

    for q in QUERIES:
        v2_vec = sparse_encode(m_v2, tok_v2, q)
        v2ft_vec = sparse_encode(m_v2ft, tok_v2ft, q)
        pub_vec = sparse_encode(m_pub, tok_pub, q)

        v2_hits, _ = top_tokens(v2_vec, tok_v2, args.top)
        v2ft_hits, v2ft_all = top_tokens(v2ft_vec, tok_v2ft, args.top)
        pub_hits, pub_all = top_tokens(pub_vec, tok_pub, args.top)

        v2_n = (v2_vec > 0).sum().item()
        v2ft_n = (v2ft_vec > 0).sum().item()
        pub_n = (pub_vec > 0).sum().item()

        # Scan ALL activations for boost and vocab-new
        v2ft_set = set(t for t, _ in v2ft_all)
        pub_boost = [(t, v) for t, v in pub_all if t not in v2ft_set and t in bert_vocab]
        pub_vocab_new = [(t, v) for t, v in pub_all if t not in bert_vocab]

        print(f"Q: {q}")
        print(f"  v2-mini    ({v2_n:5d} tok): {fmt(v2_hits)}")
        print(f"  v2-mini-FT ({v2ft_n:5d} tok): {fmt(v2ft_hits)}")
        print(f"  PubBERT-FT ({pub_n:5d} tok): {fmt(pub_hits, bert_vocab)}")
        if pub_boost:
            print(f"  🧬 FT boost: {fmt(pub_boost[:TOP_N], bert_vocab)}")
        if pub_vocab_new:
            print(f"  ★  NOT IN BERT VOCAB: {fmt(pub_vocab_new[:TOP_N], bert_vocab)}")
        print()


if __name__ == "__main__":
    main()
