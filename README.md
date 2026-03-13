# Fine-tuning Neural Sparse Model for Domain-Specific Data

End-to-end pipeline for fine-tuning OpenSearch's sparse encoder model
using data from an existing OpenSearch index. Uses the doc-only encoder
approach (inference-free at query time) matching the official OpenSearch
neural sparse architecture.

Companion code for the OpenSearchCon talk:
**"Fine-tuning Neural Sparse Model for Domain Specific Data from an Existing OpenSearch Index"**
— sequel to [Budget Friendly Semantic Search With Neural Sparse Search](https://www.youtube.com/watch?v=kx71KFf-Nv0).

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 0 (Pre-requisite): index_data.py                          │
│  Index 49 health articles into OpenSearch                       │
│  → Creates the "existing domain data" in your cluster           │
│  → Run before the demo; demo starts by showing this index       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: prepare_data.py                                        │
│  Generate training data from the index                          │
│  → Fetch docs from OpenSearch                                   │
│  → LLM generates 5 lay-person queries per doc (Ollama qwen2.5:7b)│
│  → Search index for hard negatives (must_not positive doc)      │
│  → LLM validates negatives (reject if relevant)                 │
│  → LLM generates easy (OOD) negative                            │
│  → Output: data/train_v2.jsonl                                  │
└──────────────────────────┬──────────────────────────────────────┘
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
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: probe.py                                               │
│  Verify the model learned domain terms                          │
│  → Compare base vs fine-tuned document encoding                 │
│  → Shows ✅ NEW domain terms the fine-tuned model activates     │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: export_torchscript.py                                  │
│  Package model for OpenSearch deployment                        │
│  → TorchScript trace + tokenizer → model.zip                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Deploy to OpenSearch                                   │
│  Register and deploy via ML Commons API                         │
│  → POST /_plugins/_ml/models/_register                          │
│  → POST /_plugins/_ml/models/<id>/_deploy                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: demo_compare.py                                        │
│  Side-by-side comparison via OpenSearch Predict API              │
│  → ★ NEW terms, ▲ BOOSTED terms                                │
│  → Compact display prioritizing notable changes                 │
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

Pre-generated training data is included (`data/train_v2.jsonl` — 238 samples),
so you can skip Step 1 and go straight to training.

## Step Details

### Step 0: Index domain documents

```bash
python index_data.py
# → 49 health articles indexed into 'health-articles'

# Verify index exists
curl -s localhost:9202/health-articles/_count | python3 -m json.tool

# Show a few documents
curl -s localhost:9202/health-articles/_search?size=3 | python3 -m json.tool
```

Or re-index from scratch:
```bash
python index_data.py
```

### Step 1: Generate training data

Requires Ollama (`ollama pull qwen2.5:7b`) and the health-articles index
(run `index_data.py` first).

```bash
python prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

# Inspect a sample
head -1 data/train_v2.jsonl | python3 -m json.tool
```

Each sample contains:
- `query`: LLM-generated lay-person question
- `pos`: the original document from the index
- `negs`: [hard_negative_from_index, easy_negative_from_llm]

### Step 2: Fine-tune

```bash
python train.py --data data/train_v2.jsonl --output finetuned_model \
    --in-batch-negatives --batch-size 15 --epochs 10
```

Key parameters:
- `--in-batch-negatives`: all pos+neg docs in the batch become negatives
  for every query (batch_size=15 → 29 negatives per query)
- `--flops-lambda 0.05`: sparsity regularization on doc representations
- `--idf-path idf.json`: IDF weights for inf_free query encoding
- `--seed 37`: reproducible results

### Step 3: Probe

```bash
python probe.py --model finetuned_model
```

Shows domain terms (vaccine, virus, fever, etc.) that the fine-tuned model
activates but the base model doesn't. Includes OOD control queries to
verify no domain contamination.

### Step 4: Export

```bash
python export_torchscript.py finetuned_model model.zip
```

Wraps the model in a TorchScript-compatible module and packages it with
tokenizer files into a zip that OpenSearch ML Commons can load.

### Step 5: Deploy to OpenSearch

```bash
# Serve model.zip via HTTP
python -m http.server 8082 &

# Register (DevTools)
POST /_plugins/_ml/models/_register
{
  "name": "finetuned-sparse",
  "version": "1.0.0",
  "model_format": "TORCH_SCRIPT",
  "function_name": "SPARSE_ENCODING",
  "url": "http://<HOST>:8082/model.zip",
  "model_content_hash_value": "<sha256sum model.zip>"
}

# Check task, get model_id:
GET /_plugins/_ml/tasks/<task_id>

# Deploy:
POST /_plugins/_ml/models/<model_id>/_deploy
```

### Step 6: Compare

Update `BASE_ID` and `FT_ID` in `demo_compare.py` with your deployed model IDs, then:

```bash
python demo_compare.py

# Or with a specific query
python demo_compare.py "I got my first two vaccine doses but I'm not sure when I should get my booster shot."
```

Shows a compact side-by-side comparison with ★ NEW terms (not in base),
▲ BOOSTED terms (>10% increase), and summary counts.

### Predict API (DevTools)

```
# Base model (pre-trained)
POST /_plugins/_ml/_predict/sparse_encoding/<base_model_id>
{ "text_docs": ["I got my first two vaccine doses but I'm not sure when I should get my booster shot."] }

# Fine-tuned model
POST /_plugins/_ml/_predict/sparse_encoding/<finetuned_model_id>
{ "text_docs": ["I got my first two vaccine doses but I'm not sure when I should get my booster shot."] }
```

## Search Relevancy Workbench

Create two sparse-encoded indices (one per model) and compare with default
tokenizer queries (doc-only mode — no model_id needed at query time):

**Panel 1 — Base** (index: `health-articles-sparse`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%"}}}}
```

**Panel 2 — Fine-tuned** (index: `health-articles-finetuned`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%"}}}}
```

**Best demo queries:**
- `can you get sick twice` — Base returns Workplace Outbreaks, FT returns Reinfection Risk
- `can i exercise after being sick` — FT promotes Rehabilitation to #1
- `chest feels tight and hard to breathe` — FT promotes Respiratory Distress to #1
- `my whole family got sick after a party` — FT surfaces Superspreader Events

## Requirements

- Python 3.9+ with torch, transformers, tqdm
- OpenSearch (scripts default to port 9202 — edit `OS_URL` in scripts to change)
- Ollama with qwen2.5:7b (for Step 1 only — skip if using included training data)
- idf.json (included — pre-computed IDF weights from official OpenSearch repo)
