# Fine-tuning Neural Sparse Model for Domain-Specific Data

End-to-end pipeline for fine-tuning OpenSearch's sparse encoder model
using data from an existing OpenSearch index. Uses the doc-only encoder
approach (inference-free at query time) matching the official OpenSearch
neural sparse architecture.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 0 (Pre-requisite): index_data.py                          │
│  Index 49 health articles into OpenSearch                       │
│  → Creates the "existing domain data" in your cluster           │
│  → Run before the demo; demo starts by showing this index       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: prepare_data.py                                        │
│  Generate training data from the index                          │
│  → Fetch docs from OpenSearch                                   │
│  → LLM generates 5 lay-person queries per doc (Ollama qwen2.5:7b) │
│  → Search index for hard negatives (must_not positive doc)      │
│  → LLM validates negatives (reject if relevant)                 │
│  → LLM generates easy (OOD) negative                            │
│  → Output: data/train_v2.jsonl                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: train.py                                               │
│  Fine-tune the doc-only sparse encoder                          │
│  → Doc-only (inf_free): queries use tokenizer + IDF weights,    │
│    only documents go through the model                          │
│  → In-batch negatives: every doc in the batch is a negative     │
│    for every other query (batch_size=15 → 29 negatives/query)   │
│  → InfoNCE contrastive loss + FLOPS regularization on docs      │
│  → Output: finetuned_model/                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: probe.py                                               │
│  Verify the model learned domain terms                          │
│  → Compare base vs fine-tuned document encoding                 │
│  → Shows ✅ NEW domain terms the fine-tuned model activates     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: export_torchscript.py                                  │
│  Package model for OpenSearch deployment                        │
│  → TorchScript trace + tokenizer → model.zip                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Deploy to OpenSearch                                   │
│  Register and deploy via ML Commons API                         │
│  → POST /_plugins/_ml/models/_register                          │
│  → POST /_plugins/_ml/models/<id>/_deploy                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: demo_compare.py                                        │
│  Side-by-side comparison via OpenSearch Predict API              │
│  → ★ NEW terms, ▲ BOOSTED terms, ▼ DROPPED terms              │
│  → Top-28 token columns for base vs fine-tuned                  │
└─────────────────────────────────────────────────────────────────┘
```

## Full Pipeline (run all steps)

```bash
# Step 0: Index domain documents
python3 index_data.py

# Step 1: Generate training data (requires Ollama with qwen2.5:7b)
python3 prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

# Step 2: Fine-tune (doc-only encoder with in-batch negatives)
docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \
    python train.py --data data/train_v2.jsonl --output finetuned_model \
    --in-batch-negatives --batch-size 15 --epochs 10

# Step 3: Verify
docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \
    python probe.py --model finetuned_model

# Step 4: Export
docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \
    python export_torchscript.py

# Step 5: Deploy (see DevTools section below)

# Step 6: Compare
python3 demo_compare.py
```

## Demo Each Step Independently

### Step 0: Show the index (pre-requisite, already indexed)

```bash
# Verify index exists
curl -s localhost:9202/health-articles/_count | python3 -m json.tool

# Show a few documents
curl -s localhost:9202/health-articles/_search?size=3 | python3 -m json.tool
```

Or re-index from scratch:
```bash
python3 index_data.py
```

### Step 1: Generate training data (requires Ollama)

```bash
python3 prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

# Inspect a sample
head -1 data/train_v2.jsonl | python3 -m json.tool
```

### Step 2: Fine-tune (uses pre-generated data if not regenerating)

```bash
docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \
    python train.py --data data/train_v2.jsonl --output finetuned_model \
    --in-batch-negatives --batch-size 15 --epochs 10
```

### Step 3: Probe (uses pre-trained finetuned_model/ if not retraining)

```bash
docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \
    python probe.py --model finetuned_model
```

### Step 4: Export

```bash
docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \
    python export_torchscript.py
```

### Step 5: Deploy to OpenSearch (DevTools)

```
# Serve model.zip via HTTP first:
#   docker run -d --name sparse-http --network sparse_finetune_default \
#       -v $(pwd):/workspace -w /workspace python:3.11-slim python -m http.server 8082

POST /_plugins/_ml/models/_register
{
  "name": "finetuned-sparse-oscon-demov2",
  "version": "1.0.0",
  "model_format": "TORCH_SCRIPT",
  "function_name": "SPARSE_ENCODING",
  "url": "http://<CONTAINER_IP>:8082/model.zip",
  "model_content_hash_value": "<SHA256>"
}

# Check task, get model_id:
GET /_plugins/_ml/tasks/<task_id>

# Deploy:
POST /_plugins/_ml/models/<model_id>/_deploy
```

### Step 6: Compare (requires models deployed in OpenSearch)

```bash
python3 demo_compare.py

# Or with custom queries
python3 demo_compare.py "my whole family got sick after a party" "do i need a booster shot"
```

### Predict API (DevTools)

```
# Base model (pre-trained)
POST /_plugins/_ml/_predict/sparse_encoding/xIWxe5wBnofIPe4lihO5
{ "text_docs": ["my whole family got sick after a party"] }

# Fine-tuned model
POST /_plugins/_ml/_predict/sparse_encoding/N6OO3ZwBnofIPe4llXYK
{ "text_docs": ["my whole family got sick after a party"] }
```

## Search Relevancy Workbench

Compare base vs fine-tuned on actual search results using two indices:

**Panel 1 — Base** (index: `health-articles-sparse`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%","model_id":"xIWxe5wBnofIPe4lihO5"}}}}
```

**Panel 2 — Fine-tuned** (index: `health-articles-finetuned-v2`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%","model_id":"N6OO3ZwBnofIPe4llXYK"}}}}
```

**Best demo queries:**
1. `kids keep getting sick at school` — School Reopening promoted to #2
2. `feeling hot and shivery` — Fever article surfaces at #2
3. `my whole family got sick after a party` — Superspreader surfaces
4. `how to configure nginx reverse proxy` — OOD control, lower scores

## Requirements

- OpenSearch 2.x on port 9202
- Ollama with qwen2.5:7b on port 11434 (for Step 1)
- Docker with `sparse-ft` image (Python 3.11 + torch + transformers)
- idf.json (IDF weights, included)
