#!/usr/bin/env python3
"""
Step 3: Verify that the fine-tuned model learned domain terminology.

Compares the base (pre-trained) model against the fine-tuned model on
document encoding. Since this is a doc-only encoder, we check how the
model expands document representations with domain-specific terms.

Usage:
    docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \\
        python probe.py --model finetuned_model

What it does:
    - Runs each text through both base and fine-tuned models
    - Extracts activations for target medical terms (vaccine, virus, etc.)
    - Highlights NEW terms the fine-tuned model activates
    - Includes OOD control queries to verify no domain contamination
"""
import torch, argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini"
THRESHOLD = 0.01

TARGET_TERMS = [
    "antibody", "antigen", "booster", "cough", "epidemic", "fatigue",
    "fever", "immunity", "infection", "inflammation", "lung", "mask",
    "mortality", "outbreak", "oxygen", "pneumonia", "protein",
    "respiratory", "spike", "transmission", "vaccine", "viral", "virus",
]

QUERIES = [
    "my whole family got sick after a party",
    "when should i get my booster shot",
    "i have a fever and dry cough",
    "still tired months after being sick",
    "how does the virus spread",
    # OOD control
    "how to configure nginx reverse proxy",
]


def sparse_encode(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs).logits
    return torch.max(torch.log1p(torch.relu(out)) * inputs["attention_mask"].unsqueeze(-1), dim=1).values.squeeze()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="finetuned_model")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).eval()
    ft = AutoModelForMaskedLM.from_pretrained(args.model).eval()

    target_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in TARGET_TERMS}
    target_ids = {t: i for t, i in target_ids.items() if i != tokenizer.unk_token_id}

    for q in QUERIES:
        bv = sparse_encode(base, tokenizer, q)
        fv = sparse_encode(ft, tokenizer, q)
        b_hits = {t: round(bv[i].item(), 3) for t, i in target_ids.items() if bv[i] > THRESHOLD}
        f_hits = {t: round(fv[i].item(), 3) for t, i in target_ids.items() if fv[i] > THRESHOLD}
        b_tok = (bv > 0).sum().item()
        f_tok = (fv > 0).sum().item()
        new = {t: v for t, v in f_hits.items() if t not in b_hits}
        print(f"Q: {q}")
        print(f"  Base ({b_tok} tok): {b_hits or 'NONE'}")
        print(f"  FT   ({f_tok} tok): {f_hits or 'NONE'}")
        if new:
            print(f"  ✅ NEW: {new}")
        print()


if __name__ == "__main__":
    main()
