#!/usr/bin/env python3
"""
Step 2: Fine-tune the doc-only sparse encoder on domain training data.

Takes the training JSONL from prepare_data.py and fine-tunes the
opensearch-neural-sparse-encoding-doc-v2-mini model using the doc-only
(inference-free) approach matching the official OpenSearch architecture.

Usage:
    docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \\
        python train.py --data data/train_v2.jsonl --output finetuned_model \\
        --in-batch-negatives --batch-size 15 --epochs 10

What it does:
    - Queries: encoded with tokenizer + IDF weights only (inf_free, no model)
    - Documents: encoded with full model (log1p(relu(max_pool(logits))))
    - In-batch negatives: all pos+neg docs in the batch become negatives
      for every query (batch_size=15 → 29 negatives per query)
    - InfoNCE contrastive loss + FLOPS regularization on doc representations
    - Saves the fine-tuned model weights and tokenizer

This matches the official opensearch-sparse-model-tuning-sample approach:
    - inf_free=true for queries (tokenizer + IDF)
    - inf_free=false for documents (full model encoding)
    - use_in_batch_negatives=true
    - flops_d_lambda=0.05
"""
import argparse, json, random, os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

# Matches opensearch-sparse-model-tuning-sample: inf_free queries, InfoNCE + FLOPS, in-batch negatives
BASE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini"


class IRDataset(Dataset):
    def __init__(self, path):
        self.samples = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["query"], s["pos"], random.choice(s["negs"])


def sparse_encode(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   position_ids=None, token_type_ids=None).logits
    pooled = torch.max(logits * attention_mask.unsqueeze(-1), dim=1).values
    return torch.log1p(F.relu(pooled))


def inf_free_encode(input_ids, idf_vector, special_ids):
    """Tokenizer-only query encoding: one-hot × IDF weights, no model."""
    batch_size = input_ids.shape[0]
    vocab_size = idf_vector.shape[0]
    out = torch.zeros(batch_size, vocab_size, device=input_ids.device)
    out[torch.arange(batch_size).unsqueeze(-1), input_ids] = 1
    out[:, special_ids] = 0
    return out * torch.relu(idf_vector)


def infonce_loss(q_rep, pos_rep, neg_rep, temperature=0.05, use_in_batch_negatives=False):
    if use_in_batch_negatives:
        # All pos + neg docs become candidates for every query
        all_docs = torch.cat([pos_rep, neg_rep], dim=0)
        scores = torch.matmul(q_rep, all_docs.t()) / temperature
        # Labels: query_i should match pos_i (index i)
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
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(device)

    # Load IDF weights for inf_free query encoding
    idf_dict = json.load(open(args.idf_path))
    idf_vector = torch.zeros(len(tokenizer.vocab), device=device)
    for token, weight in idf_dict.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id:
            idf_vector[tid] = weight
    special_ids = torch.tensor(tokenizer.all_special_ids, device=device)

    dataset = IRDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=lambda b: collate(b, tokenizer, args.max_length))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs

    print(f"Training: {len(dataset)} samples, {len(loader)} steps/epoch, {total_steps} total steps")
    print(f"FLOPS: {args.flops_lambda}, LR: {args.lr}, batch: {args.batch_size}, epochs: {args.epochs}")
    print(f"In-batch negatives: {args.in_batch_negatives}, Query encoding: inf_free (tokenizer + IDF)")

    model.train()
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for q_tok, p_tok, n_tok in pbar:
            q_tok = {k: v.to(device) for k, v in q_tok.items()}
            p_tok = {k: v.to(device) for k, v in p_tok.items()}
            n_tok = {k: v.to(device) for k, v in n_tok.items()}

            # Queries: inf_free (tokenizer + IDF, no model)
            q_rep = inf_free_encode(q_tok["input_ids"], idf_vector, special_ids)
            # Docs: full model encoding
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
                             flops=f"{loss_f.item():.4f}", lr=f"{args.lr:.2e}")
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
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
    p.add_argument("--idf-path", default="idf.json")
    train(p.parse_args())
