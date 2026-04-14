#!/usr/bin/env python3
"""
Fine-tune a sparse encoder using the doc-only (inference-free) approach.

Default base model is PubMedBERT, but any BERT-based MLM model works via --model.
Matches the official opensearch-sparse-model-tuning-sample training mechanics:
inf_free queries, InfoNCE + FLOPS, in-batch negatives.

Usage:
    python train.py --data data/train_v2.jsonl --output pubmedbert_finetuned \
        --idf-path idf_pubmedbert_clean.json --in-batch-negatives --batch-size 15 --epochs 30
"""
import argparse, json, random, os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


class IRDataset(Dataset):
    def __init__(self, path):
        self.samples = [json.loads(l) for l in open(path)]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["query"], s["pos"], random.choice(s["negs"])


def sparse_encode(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    pooled = torch.max(logits * attention_mask.unsqueeze(-1), dim=1).values
    return torch.log1p(F.relu(pooled))


def inf_free_encode(input_ids, idf_vector, special_ids):
    batch_size = input_ids.shape[0]
    vocab_size = idf_vector.shape[0]
    out = torch.zeros(batch_size, vocab_size, device=input_ids.device)
    out[torch.arange(batch_size).unsqueeze(-1), input_ids] = 1
    out[:, special_ids] = 0
    return out * torch.relu(idf_vector)


def infonce_loss(q_rep, pos_rep, neg_rep, temperature=0.05, use_in_batch_negatives=False):
    if use_in_batch_negatives:
        all_docs = torch.cat([pos_rep, neg_rep], dim=0)
        scores = torch.matmul(q_rep, all_docs.t()) / temperature
        labels = torch.arange(q_rep.shape[0], device=scores.device)
        return F.cross_entropy(scores, labels)
    else:
        pos_score = (q_rep * pos_rep).sum(dim=-1, keepdim=True)
        neg_score = (q_rep * neg_rep).sum(dim=-1, keepdim=True)
        scores = torch.cat([pos_score, neg_score], dim=1) / temperature
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores, labels)


def flops_loss(rep):
    return torch.sum(torch.mean(torch.abs(rep), dim=0) ** 2)


def collate(batch, tokenizer, max_len):
    queries, positives, negatives = zip(*batch)
    kw = dict(padding=True, truncation=True, max_length=max_len,
              return_tensors="pt", return_token_type_ids=False)
    return tokenizer(list(queries), **kw), tokenizer(list(positives), **kw), tokenizer(list(negatives), **kw)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    print(f"Device: {device}")
    print(f"Base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    # Load IDF weights — tokens not found in this tokenizer's vocab get weight 0
    idf_dict = json.load(open(args.idf_path))
    idf_vector = torch.zeros(tokenizer.vocab_size, device=device)
    matched = 0
    for token, weight in idf_dict.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id:
            idf_vector[tid] = weight
            matched += 1
    print(f"IDF: {matched}/{len(idf_dict)} tokens matched ({100*matched/len(idf_dict):.0f}%)")
    special_ids = torch.tensor(tokenizer.all_special_ids, device=device)

    dataset = IRDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=lambda b: collate(b, tokenizer, args.max_length))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs

    print(f"Training: {len(dataset)} samples, {len(loader)} steps/epoch, {total_steps} total steps")
    print(f"FLOPS: {args.flops_lambda}, batch: {args.batch_size}, epochs: {args.epochs}")
    print(f"In-batch negatives: {args.in_batch_negatives}")

    model.train()
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for q_tok, p_tok, n_tok in pbar:
            q_tok = {k: v.to(device) for k, v in q_tok.items()}
            p_tok = {k: v.to(device) for k, v in p_tok.items()}
            n_tok = {k: v.to(device) for k, v in n_tok.items()}

            q_rep = inf_free_encode(q_tok["input_ids"], idf_vector, special_ids)
            p_rep = sparse_encode(model, p_tok["input_ids"], p_tok["attention_mask"])
            n_rep = sparse_encode(model, n_tok["input_ids"], n_tok["attention_mask"])

            loss_rank = infonce_loss(q_rep, p_rep, n_rep,
                                     use_in_batch_negatives=args.in_batch_negatives)
            flops_w = args.flops_lambda * min(1.0, (step / args.flops_warmup) ** 2) if args.flops_warmup > 0 else args.flops_lambda
            loss_f = flops_loss(p_rep) * flops_w

            loss = loss_rank + loss_f
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", rank=f"{loss_rank.item():.4f}",
                             flops=f"{loss_f.item():.4f}")
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, help="Base model (HuggingFace ID or local path)")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--flops-lambda", type=float, default=0.05)
    p.add_argument("--flops-warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--in-batch-negatives", action="store_true")
    p.add_argument("--idf-path", default="idf_pubmedbert_clean.json")
    train(p.parse_args())
