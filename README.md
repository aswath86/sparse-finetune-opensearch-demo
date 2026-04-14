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

| | v3 (China talk) | v4 (this repo) |
|---|---|---|
| Base model | `opensearch-neural-sparse-encoding-doc-v2-mini` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| IDF weights | `idf.json` (BERT vocab, MS MARCO) | `idf_pubmedbert_clean.json` (PubMedBERT vocab, 61K PubMed abstracts) |
| Comparison | 2-way: base vs fine-tuned | 3-way: v2-mini vs v2-mini-FT vs PubMedBERT-FT |
| Markers | ★ NEW, ▲ BOOSTED | + ◆ NOT IN BERT VOCAB |
| Training data | Same `data/train_v2.jsonl` (238 samples) | Same |
| Training | Same hyperparameters | `--model` and `--idf-path` args added |

## Pipeline Flow

```
index_data.py          Index 49 health articles into OpenSearch
       ↓
prepare_data.py        Generate training data (queries + negatives via Ollama)
       ↓
train.py               Fine-tune PubMedBERT sparse encoder (--model, --idf-path)
       ↓
probe.py               Three-way token comparison (v2-mini vs v2-mini-FT vs PubMedBERT-FT)
       ↓
export_torchscript.py  Package for OpenSearch (--model)
       ↓
Deploy via ML Commons  Register + deploy model.zip
       ↓
demo_compare.py        Three-way via OpenSearch Predict API (★ ▲ ▼ ◆ markers)
```

A Streamlit UI (`app.py`) provides an interactive version of probe and
demo_compare with colored token bars and preset queries.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 0: Index documents
python index_data.py

# Step 1: Generate training data (requires Ollama with qwen2.5:7b)
python prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

# Step 2: Build IDF from PubMed abstracts (one-time)
python -c "
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import json, math

tok = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
ds = load_dataset('pubmed_qa', 'pqa_unlabeled', split='train')
df_count, N = Counter(), 0
for row in ds:
    text = ' '.join(row['context']['contexts'])
    ids = set(tok.encode(text, add_special_tokens=False))
    for i in ids: df_count[i] += 1
    N += 1
idf = {}
for i, c in df_count.items():
    t = tok.decode([i]).strip()
    if t: idf[t] = round(math.log(N / c), 4)
json.dump(idf, open('idf_pubmedbert.json', 'w'))
print(f'{len(idf)} tokens, {N} docs')
"

# Step 2b: Zero stopwords in IDF (prevents model learning to activate 'my', 'something', etc.)
# See idf_pubmedbert_clean.json (included — 286 stopwords zeroed)

# Step 3: Fine-tune (~15 min on CPU for 30 epochs)
python train.py \
    --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --data data/train_v2.jsonl \
    --output pubmedbert_finetuned_30ep_v3 \
    --idf-path idf_pubmedbert_clean.json \
    --in-batch-negatives --batch-size 15 --epochs 30

# Step 4: Verify
python probe.py --pubft pubmedbert_finetuned_30ep_v3

# Step 5: Export for OpenSearch
python export_torchscript.py --model pubmedbert_finetuned_30ep_v3 --output pubmedbert_model.zip

# Step 6: Deploy (see Deployment section below)

# Step 7: Compare via Predict API
python demo_compare.py
```

Pre-generated training data (`data/train_v2.jsonl` — 238 samples) and
pre-built IDF files (`idf_pubmedbert.json`, `idf_pubmedbert_clean.json`)
are included, so you can skip Steps 1-2 and go straight to training.

## IDF Rebuild: Why It Matters

The original `idf.json` was built from MS MARCO using BERT's tokenizer.
Only 40% of PubMedBERT tokens matched — domain-specific tokens like
`mellitus`, `vaccination`, `antibiotic` had IDF weight = 0, so the model
got no gradient signal for them.

`idf_pubmedbert_clean.json` is built from 61K PubMed abstracts using
PubMedBERT's tokenizer (100% match), with 286 informal stopwords zeroed
(words like `my`, `something`, `feeling` that are rare in PubMed papers
and would otherwise get artificially high IDF weights).

## Deployment

```bash
# Serve model zip via HTTP (OpenSearch in Docker can't see host filesystem)
python -m http.server 8765 &

# Register — use function_name, NOT model_task_type, and omit model_config
POST /_plugins/_ml/models/_register
{
  "name": "pubmedbert-sparse-ft",
  "version": "1.0.0",
  "model_format": "TORCH_SCRIPT",
  "function_name": "SPARSE_ENCODING",
  "url": "http://<HOST_PRIVATE_IP>:8765/pubmedbert_model.zip",
  "model_content_hash_value": "<sha256sum pubmedbert_model.zip>"
}

GET /_plugins/_ml/tasks/<task_id>
POST /_plugins/_ml/models/<model_id>/_deploy
```

Requires `allow_registering_model_via_url: true` and `private_ip_enabled: true`
in cluster settings.

## Streamlit App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Two tabs:
- **Local Probe** — loads models from disk, runs inference via HuggingFace
- **Predict API** — hits OpenSearch Predict API for deployed models

Preset queries are chosen to maximize the subword-vs-whole-token contrast
(e.g., "COPD symptoms and treatment" where BERT sees `cop`+`police` and
PubMedBERT sees `copd`+`respiratory`+`ventilation`).

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
- OpenSearch (scripts default to `localhost:9202`)
- Ollama with `qwen2.5:7b` (Step 1 only — skip if using included training data)
- `transformers==5.3.0` (5.5.0 breaks TorchScript export)

## Scaling

See the `main` branch README for vocabulary-driven sampling guidance
(rare/medium/common term stratification via field data terms aggregation)
when working with production indices containing millions of documents.
