# Fine-tuning Neural Sparse Model for Domain-Specific Data

End-to-end pipeline for fine-tuning OpenSearch's sparse encoder model
using data from an existing OpenSearch index. Uses the doc-only encoder
approach (inference-free at query time) matching the official OpenSearch
neural sparse architecture.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 0: index_data.py                                          │
│  Index 49 health articles into OpenSearch                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: prepare_data.py                                        │
│  Fetch docs → LLM generates queries → hard negatives from       │
│  index (must_not) → LLM validates → OOD easy negative           │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: train.py                                               │
│  Doc-only fine-tuning: queries = tokenizer + IDF (no model),    │
│  docs = full model. InfoNCE + FLOPS + in-batch negatives.       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: probe.py                                               │
│  Compare base vs fine-tuned — shows ✅ NEW domain terms         │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: export_torchscript.py                                  │
│  TorchScript trace + tokenizer → model.zip                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Deploy to OpenSearch via ML Commons API                │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: demo_compare.py                                        │
│  Side-by-side via Predict API: ★ NEW / ▲ BOOSTED markers       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers tqdm

# Step 0: Index documents
python index_data.py

# Step 1: Generate training data (requires Ollama with qwen2.5:7b)
python prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

# Step 2: Fine-tune (~2 min on CPU)
python train.py --data data/train_v2.jsonl --output finetuned_model \
    --in-batch-negatives --batch-size 15 --epochs 10

# Step 3: Verify
python probe.py --model finetuned_model

# Step 4: Export for OpenSearch
python export_torchscript.py

# Step 5: Deploy (see below)

# Step 6: Compare (requires deployed models)
python demo_compare.py
```

## Step Details

### Step 0: Index domain documents

```bash
python index_data.py
# → 49 health articles indexed into 'health-articles'

curl -s localhost:9202/health-articles/_count | python3 -m json.tool
```

### Step 1: Generate training data

Requires Ollama (`ollama pull qwen2.5:7b`).

```bash
python prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

# Inspect a sample
head -1 data/train_v2.jsonl | python3 -m json.tool
```

Each sample: `{query, pos, negs: [hard_negative, ood_negative]}`

### Step 2: Fine-tune

```bash
python train.py --data data/train_v2.jsonl --output finetuned_model \
    --in-batch-negatives --batch-size 15 --epochs 10
```

Key parameters:
- `--in-batch-negatives`: batch_size=15 → 29 negatives per query
- `--flops-lambda 0.05`: sparsity regularization
- `--idf-path idf.json`: IDF weights for inf_free query encoding

### Step 3: Probe

```bash
python probe.py --model finetuned_model
```

### Step 4: Export

```bash
python export_torchscript.py finetuned_model model.zip
```

### Step 5: Deploy to OpenSearch

```bash
# Serve model.zip via HTTP
python -m http.server 8082 &

# Register (DevTools)
POST /_plugins/_ml/models/_register
{
  "name": "finetuned-sparse-v3",
  "version": "1.0.0",
  "model_format": "TORCH_SCRIPT",
  "function_name": "SPARSE_ENCODING",
  "url": "http://<HOST>:8082/model.zip",
  "model_content_hash_value": "<sha256sum model.zip>"
}

# Get model_id from task, then deploy
POST /_plugins/_ml/models/<model_id>/_deploy
```

### Step 6: Compare

Update `BASE_ID` and `FT_ID` in `demo_compare.py` with your model IDs, then:

```bash
python demo_compare.py

# Or with a specific query
python demo_compare.py "I got my first two vaccine doses but I'm not sure when I should get my booster shot."
```

## Search Relevancy Workbench

Create two sparse-encoded indices (one per model) and compare with default tokenizer queries (doc-only mode):

**Panel 1 — Base** (index: `health-articles-sparse`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%"}}}}
```

**Panel 2 — Fine-tuned** (index: `health-articles-finetuned`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%"}}}}
```

Demo queries: `can you get sick twice`, `can i exercise after being sick`, `chest feels tight and hard to breathe`

## Requirements

- Python 3.9+ with torch, transformers, tqdm
- OpenSearch on port 9202
- Ollama with qwen2.5:7b on port 11434 (for Step 1 only)
- idf.json (included)
