# Fine-tuning Neural Sparse Model with a Domain-Specific Backbone

End-to-end pipeline for fine-tuning OpenSearch's sparse encoder using
**PubMedBERT** as the base model instead of generic BERT. Demonstrates that
a domain-pretrained backbone activates biomedical vocabulary that BERT
physically cannot produce — tokens like `vaccination`, `corticosteroids`,
`gastrointestinal`, and `ventilator` exist as whole words in PubMedBERT's
vocabulary but are broken into meaningless subwords by BERT.

Follow-up to the OpenSearchCon China talk
**"Fine-tuning Neural Sparse Model for Domain Specific Data from an Existing OpenSearch Index"**
([video](https://www.youtube.com/watch?v=9aaXCTq8-NM)).
The China talk showed fine-tuning the sparse head improves retrieval quality.
This demo shows that changing the backbone adds vocabulary richness —
the two improvements are complementary.

Branch `demov4-pubmedbert` of
[sparse-finetune-opensearch-demo](https://github.com/aswath86/sparse-finetune-opensearch-demo).
The `main` branch contains the original (v3) demo.

## The Problem

BERT tokenizes "COPD" → `cop` + `##d`. The model then activates `police`,
`cops`, `policeman` — completely wrong for a medical query.

PubMedBERT has `copd` as a single token. After fine-tuning, it activates
`breath`, `ventilation`, `respiratory`, `lung` — correct domain terms.

No amount of fine-tuning can fix this. The vocabulary is baked into the
backbone. You need a domain-pretrained model.

## What Changed from v3

| | v3 | v4 (this repo) |
|---|---|---|
| Base model | `opensearch-neural-sparse-encoding-doc-v2-mini` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| IDF weights | `idf.json` (BERT vocab, MS MARCO) | `idf_pubmedbert_clean.json` (PubMedBERT vocab, 61K PubMed abstracts) |
| Comparison | 2-way: base vs fine-tuned | 3-way: v2-mini vs v2-mini-FT vs PubMedBERT-FT |
| Markers | ★ NEW, ▲ BOOSTED | + ◆ NOT IN BERT VOCAB |
| Training data | Same `data/train_v2.jsonl` (238 samples) | Same |
| Training | Same hyperparameters | `--model` and `--idf-path` args added |

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 0 (Pre-requisite): index_data.py                          │
│  Index 49 health articles into OpenSearch                       │
│  → Creates the "existing domain data" in your cluster           │ 
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: prepare_data.py                                        │
│  Generate training data from the index                          │
│  → Fetch docs from OpenSearch                                   │
│  → LLM generates 5 layperson queries per doc (Ollama qwen2.5:7b)│
│  → Search index for hard negatives (must_not positive doc)      │
│  → LLM validates negatives (reject if relevant)                 │
│  → LLM generates easy (OOD) negative                            │
│  → Output: data/train_v2.jsonl                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: build_idf.py                                           │
│  Build IDF weights from PubMed abstracts                        │
│  → Downloads pubmed_qa unlabeled split (~61K abstracts)         │
│  → Tokenizes with PubMedBERT (100% vocab match vs 40% w/ BERT)  │
│  → Computes IDF = log(N / df) per token                         │
│  → --zero-stopwords: zeros 286 informal words (my, something)   │
│  → Output: idf_pubmedbert_clean.json                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: train.py                                               │
│  Fine-tune the PubMedBERT sparse encoder                        │
│  → Doc-only (inf_free): queries use tokenizer + IDF weights,    │
│    only documents go through the model                          │
│  → In-batch negatives: every doc in the batch is a negative     │
│    for every other query (batch_size=15 → 29 negatives/query)   │
│  → InfoNCE contrastive loss + FLOPS regularization on docs      │
│  → --model: any BERT-based MLM (default: PubMedBERT)            │
│  → --idf-path: matched IDF file (default: idf_pubmedbert_clean) │
│  → Output: pubmedbert_finetuned_30ep_v3/                        │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: probe.py                                               │
│  Three-way token comparison                                     │
│  → v2-mini (base) vs v2-mini-FT (demov3) vs PubMedBERT-FT       │
│  → Shows ★ NEW tokens, ◆ VOCAB-NEW (not in BERT vocabulary)     │
│  → Includes OOD control queries to verify no contamination      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: export_torchscript.py                                  │
│  Package model for OpenSearch deployment                        │
│  → TorchScript trace + tokenizer → pubmedbert_model.zip         │
│  → --model: path to fine-tuned model directory                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: Deploy to OpenSearch                                   │
│  Register and deploy via ML Commons API                         │
│  → POST /_plugins/_ml/models/_register                          │
│  → POST /_plugins/_ml/models/<id>/_deploy                       │
│  → Use function_name (NOT model_task_type), omit model_config   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 7: demo_compare.py                                        │
│  Three-way comparison via OpenSearch Predict API                │
│  → v2-mini vs v2-mini-FT vs PubMedBERT-FT                       │
│  → ★ NEW, ▲ BOOSTED, ▼ DROPPED, ◆ NOT IN BERT VOCAB             │
│  → Progressive markers: FT vs base, then PubBERT vs FT          │
└─────────────────────────────────────────────────────────────────┘
```

A Streamlit UI (`app.py`) provides an interactive version of probe and
demo_compare with colored token bars and preset queries. See the
[Streamlit App](#streamlit-app) section below.

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 0: Index documents
python index_data.py

# Step 1: Generate training data (requires Ollama with qwen2.5:7b)
python prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

# Step 2: Build IDF from PubMed abstracts (one-time)
python build_idf.py --output idf_pubmedbert_clean.json --zero-stopwords

# Step 3: Fine-tune (~15 min on CPU for 30 epochs)
python train.py --data data/train_v2.jsonl --output pubmedbert_finetuned_30ep_v3 \
    --in-batch-negatives --batch-size 15 --epochs 30

# Step 4: Verify
python probe.py --pubft pubmedbert_finetuned_30ep_v3

# Step 5: Export for OpenSearch
python export_torchscript.py --model pubmedbert_finetuned_30ep_v3 --output pubmedbert_model.zip

# Step 6: Deploy (see below)

# Step 7: Compare (requires deployed models)
python demo_compare.py
```

Pre-generated training data (`data/train_v2.jsonl` — 238 samples) and
pre-built IDF files (`idf_pubmedbert.json`, `idf_pubmedbert_clean.json`)
are included, so you can skip Steps 1-2 and go straight to training.

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

### Step 2: Build IDF weights

The IDF file must match the model's tokenizer. PubMedBERT has a custom
vocabulary trained on PubMed — using BERT's IDF gives only 40% token match
and zero gradient signal for domain-specific tokens like `mellitus`,
`vaccination`, `antibiotic`.

```bash
# Raw IDF from 61K PubMed abstracts
python build_idf.py --output idf_pubmedbert.json

# Cleaned version with stopwords zeroed (recommended for training)
python build_idf.py --output idf_pubmedbert_clean.json --zero-stopwords
```

Why zero stopwords? Words like `my` (IDF=7.33), `something`, `feeling` are
rare in PubMed papers and get artificially high IDF weights. The model then
learns to activate them. Zeroing 286 such entries fixes this at the IDF level
so the model handles stopword suppression naturally.

### Step 3: Fine-tune

```bash
python train.py \
    --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --data data/train_v2.jsonl \
    --output pubmedbert_finetuned_30ep_v3 \
    --idf-path idf_pubmedbert_clean.json \
    --in-batch-negatives --batch-size 15 --epochs 30
```

Key parameters:
- `--model`: any BERT-based MLM model (default: PubMedBERT)
- `--idf-path`: IDF weights matching the model's tokenizer
  (default: `idf_pubmedbert_clean.json`)
- `--in-batch-negatives`: all pos+neg docs in the batch become negatives
  for every query (batch_size=15 → 29 negatives per query)
- `--flops-lambda 0.05`: sparsity regularization on doc representations
- `--seed 37`: reproducible results

### Step 4: Probe

```bash
python probe.py --pubft pubmedbert_finetuned_30ep_v3
```

Three-way comparison: v2-mini (base) vs v2-mini-FT (from demov3) vs
PubMedBERT-FT. Shows top-N tokens per model with stopword filtering.
★ marks tokens not in BERT's vocabulary (PubMedBERT-only).

Includes OOD control queries (e.g., "how to configure nginx reverse proxy")
to verify no domain contamination.

### Step 5: Export

```bash
python export_torchscript.py --model pubmedbert_finetuned_30ep_v3 --output pubmedbert_model.zip
```

Wraps the model in a TorchScript-compatible module and packages it with
tokenizer files into a zip that OpenSearch ML Commons can load.

Note: requires `transformers==5.3.0` — version 5.5.0 breaks TorchScript
tracing (`masking_utils.py` error).

### Step 6: Deploy to OpenSearch

```bash
# Serve model zip via HTTP (OpenSearch in Docker can't see host filesystem)
python -m http.server 8765 &

# Register (DevTools) — use function_name, NOT model_task_type, and omit model_config
POST /_plugins/_ml/models/_register
{
  "name": "pubmedbert-sparse-ft",
  "version": "1.0.0",
  "model_format": "TORCH_SCRIPT",
  "function_name": "SPARSE_ENCODING",
  "url": "http://<HOST_PRIVATE_IP>:8765/pubmedbert_model.zip",
  "model_content_hash_value": "<sha256sum pubmedbert_model.zip>"
}

# Check task, get model_id:
GET /_plugins/_ml/tasks/<task_id>

# Deploy:
POST /_plugins/_ml/models/<model_id>/_deploy
```

Requires cluster settings:
- `allow_registering_model_via_url: true`
- `private_ip_enabled: true`

### Step 7: Compare

Update `V2_ID`, `V2FT_ID`, and `PUB_ID` in `demo_compare.py` with your
deployed model IDs, then:

```bash
python demo_compare.py

# Or with a specific query
python demo_compare.py "COPD symptoms and treatment"
```

Shows a three-way side-by-side comparison with progressive markers:
- v2-mini-FT column (vs base): ★ NEW, ▲ BOOSTED, ▼ DROPPED
- PubMedBERT-FT column (vs v2-FT): ★ NEW, ▲ BOOSTED, ▼ DROPPED, ◆ VOCAB-NEW

### Predict API (DevTools)

```
# Base model (pre-trained v2-mini)
POST /_plugins/_ml/_predict/sparse_encoding/<base_model_id>
{ "text_docs": ["COPD symptoms and treatment"] }

# Fine-tuned v2-mini (demov3)
POST /_plugins/_ml/_predict/sparse_encoding/<v2ft_model_id>
{ "text_docs": ["COPD symptoms and treatment"] }

# PubMedBERT fine-tuned (demov4)
POST /_plugins/_ml/_predict/sparse_encoding/<pubmedbert_model_id>
{ "text_docs": ["COPD symptoms and treatment"] }
```

## Streamlit App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Two tabs:
- **Local Probe** — loads models from disk, runs inference via HuggingFace
- **Predict API** — hits OpenSearch Predict API for deployed models

Features:
- Preset query buttons chosen to maximize subword-vs-whole-token contrast
  (e.g., "COPD symptoms and treatment" where BERT sees `cop`+`police` and
  PubMedBERT sees `copd`+`respiratory`+`ventilation`)
- Typeable text input for custom queries
- PubMedBERT toggle to show/hide the third column
- "Show all tokens" toggle for full activation list
- Colored bars: green (★ NEW), blue (▲ BOOSTED), red (▼ DROPPED),
  gold (◆ VOCAB-NEW)

## Search Relevancy Workbench

Create three sparse-encoded indices (one per model) and compare with default
tokenizer queries (doc-only mode — no model_id needed at query time):

**Panel 1 — Base** (index: `health-articles-sparse`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%"}}}}
```

**Panel 2 — Fine-tuned v2-mini** (index: `health-articles-finetuned`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%"}}}}
```

**Panel 3 — PubMedBERT-FT** (index: `health-articles-pubmedbert`):
```json
{"query":{"neural_sparse":{"content_sparse":{"query_text":"%SearchText%"}}}}
```

**Best demo queries:**
- `COPD symptoms and treatment` — Base: `cop`, `police`. PubMedBERT-FT: `breath`, `ventilation`
- `side effects of corticosteroids` — Base: `##oids`, `##cos`. PubMedBERT-FT: `corticosteroids◆`, `steroids◆`
- `gastrointestinal symptoms after infection` — Base: `##estinal`. PubMedBERT-FT: `gastrointestinal◆`, `intestinal◆`
- `why do diabetics get infections easily` — Base: `##bet`. PubMedBERT-FT: `diabetic◆`, `mellitus◆`
- `patient on a ventilator in the ICU` — Base: `##tor`, `##u`. PubMedBERT-FT: `ventilator◆`, `icu◆`

## Marker Legend

```
v2-mini-FT column (vs v2-mini base):
  ★  NEW        token not activated by base
  ▲  BOOSTED    weight increased >10%
  ▼  DROPPED    weight decreased >10%

PubMedBERT-FT column (vs v2-mini-FT):
  ★  NEW        token not activated by v2-mini-FT
  ▲  BOOSTED    weight increased vs v2-mini-FT
  ▼  DROPPED    weight decreased vs v2-mini-FT
  ◆  VOCAB-NEW  token does not exist in BERT's vocabulary
```

## Key Results

PubMedBERT has 18,213 tokens (60%) that don't exist in BERT's vocabulary.
268 of those are clearly biomedical. After fine-tuning, the model activates
domain terms that BERT physically cannot produce:

| Query | BERT (v2-mini) | PubMedBERT-FT ◆ tokens |
|-------|---------------|------------------------|
| COPD symptoms and treatment | `cop`, `police`, `cops` | `breath`, `ventilation`, `respiratory` |
| side effects of corticosteroids | `##oids`, `##rti`, `##cos` | `corticosteroids◆`, `steroids◆` |
| gastrointestinal symptoms | `##estinal`, `##tro` | `gastrointestinal◆`, `intestinal◆` |
| why do diabetics get infections | `##bet` | `diabetic◆`, `mellitus◆` |
| my cholesterol is too high | `##terol`, `##cho` | `cholesterol◆` |

Retrieval quality (SRW, curated queries): Base nDCG 0.74 → v2-mini-FT **0.88** → PubMedBERT-FT 0.76.
PubMedBERT-FT wins on specific queries where domain vocab matters
("breathing machine" 0.99, "high blood sugar" 0.81) but overall ranking
is noisier because PubMedBERT's sparse head was not pre-trained for sparsity.

## Requirements

- Python 3.9+ with `pip install -r requirements.txt`
- OpenSearch (scripts default to `localhost:9202` — edit `OS_URL` in scripts to change)
- Ollama with `qwen2.5:7b` (for Step 1 only — skip if using included training data)
- `transformers==5.3.0` (5.5.0 breaks TorchScript export)
- IDF files included (`idf_pubmedbert.json`, `idf_pubmedbert_clean.json`) —
  or rebuild with `build_idf.py` (requires `datasets` package)

## Scaling to Production: Vocabulary-Driven Sampling

This demo uses 49 documents — small enough to pull everything. In production with millions of documents, you need a principled way to select which documents become training data. Random sampling misses rare domain terms (where fine-tuning helps most) and over-represents common topics (where the base model is already fine).

### Use the Inverted Index as Your Sampling Guide

OpenSearch's inverted index already knows every term in your corpus and how common each one is. Use that to drive document selection.

**Step 1: Create a temporary fielddata-enabled index**

```json
PUT /my-index-fielddata
{
  "mappings": {
    "properties": {
      "content": { "type": "text", "fielddata": true }
    }
  }
}

POST /_reindex
{
  "source": { "index": "my-index" },
  "dest": { "index": "my-index-fielddata" }
}
```

`fielddata` can't be disabled once enabled, so use a temporary index and delete it when done.

**Step 2: Get terms across three frequency bands**

Rare terms (fine-tuning adds the most new knowledge here):
```json
GET /my-index-fielddata/_search
{
  "size": 0,
  "aggs": {
    "rare": {
      "terms": { "field": "content", "size": 1000, "order": { "_count": "asc" } }
    }
  }
}
```

Medium terms (core domain vocabulary — bulk of real user queries):
```json
GET /my-index-fielddata/_search
{
  "size": 0,
  "aggs": {
    "medium": {
      "terms": { "field": "content", "size": 1000, "min_doc_count": 10, "order": { "_count": "asc" } }
    }
  }
}
```
Adjust `min_doc_count` based on index size to skip the rare band.

Common terms (anchors the model, prevents drift):
```json
GET /my-index-fielddata/_search
{
  "size": 0,
  "aggs": {
    "common": {
      "terms": { "field": "content", "size": 1000, "order": { "_count": "desc" } }
    }
  }
}
```

**Step 3: For each term, find its best representative document**

```json
GET /my-index/_search
{
  "size": 1,
  "query": { "match": { "content": "thrombocytopenia" } }
}
```

The top-scoring document for a term is where that term is most prominent — the best training candidate. Deduplicate as you go (many terms point to the same doc).

**Step 4: Sample proportionally**

- 40% from rare terms
- 40% from medium terms
- 20% from common terms
- Target 1,000–2,000 documents total (up to 5,000 for very diverse domains)

**Step 5: Feed into the existing pipeline**

The sampled documents go straight into `prepare_data.py` — everything downstream is unchanged: LLM generates queries, hard negatives from the index, OOD negatives, same training code, same hyperparameters.

**Step 6: Cleanup**

```json
DELETE /my-index-fielddata
```

### Why Not Just Random Sampling?

Random sampling at scale is biased toward common topics. If 80% of your index is about one topic, 80% of your training data will be too. The base model already handles common terms — it's the rare domain terms where fine-tuning adds value, and random sampling misses them.

### Why Stratified and Not Just Rare Terms?

Sampling only rare terms introduces skew the other direction — you overtrain on niche vocabulary and the model may drift on common queries users actually search for. Stratified sampling across all three bands prevents skew in either direction.
