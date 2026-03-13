#!/usr/bin/env python3
"""
Step 1: Generate training data from an existing OpenSearch index.

Pulls documents from the 'health-articles' index and creates query-document
pairs for fine-tuning. Uses a local LLM (Ollama) to:
    1. Generate lay-person search queries per document
    2. Find hard negatives by searching the index (must_not positive doc)
    3. Validate hard negatives with LLM (reject if relevant)
    4. Generate an easy (out-of-domain) negative

Usage:
    python prepare_data.py --limit 49 --queries-per-doc 5 --output data/train_v2.jsonl

Requires:
    - OpenSearch on localhost:9202 with health-articles index
    - Ollama on localhost:11434 with qwen2.5:7b
"""
import argparse, json, urllib.request

OS_URL = "http://localhost:9202"
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"

QUERY_PROMPT = """Given this medical article, write {n} different short questions (under 10 words each) that a regular person might type into Google. Each question should approach the topic from a different angle. Use simple everyday English words only, no medical terms. Reply with ONLY the questions, one per line.

Article: {content}"""

VALIDATE_PROMPT = """Query: {query}
Document: {doc}

Is this document relevant to the query? Would a user searching for this query find this document useful? Reply with ONLY "yes" or "no"."""

OOD_PROMPT = """Write one short factual sentence about a topic completely unrelated to medicine, health, or science. Pick a random topic like sports, cooking, travel, music, or architecture. Reply with ONLY the sentence, nothing else."""


def ollama(prompt, temperature=0.7, max_tokens=200):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    })
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate",
        data=body.encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["response"].strip()


def fetch_docs(limit):
    body = json.dumps({"size": limit, "query": {"match_all": {}}, "_source": ["title", "content"]})
    req = urllib.request.Request(f"{OS_URL}/health-articles/_search",
        data=body.encode(), headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return [(h["_id"], h["_source"]["content"]) for h in resp["hits"]["hits"]]


def generate_queries(content, n):
    """Generate n lay-person queries for a document."""
    resp = ollama(QUERY_PROMPT.format(content=content, n=n))
    lines = [l.strip().strip('"').strip("'").lstrip("0123456789.-) ") for l in resp.split("\n") if l.strip()]
    return [l for l in lines if len(l) > 5][:n]


def validate_negative(query, doc):
    """Returns True if doc is a valid negative (NOT relevant to query)."""
    resp = ollama(VALIDATE_PROMPT.format(query=query, doc=doc[:300]), temperature=0, max_tokens=5)
    return resp.lower().startswith("no")


def find_hard_negative(query, exclude_id):
    """Search index with query, exclude positive, return first LLM-validated non-relevant doc."""
    body = json.dumps({
        "size": 5,
        "query": {
            "bool": {
                "must": {"match": {"content": query}},
                "must_not": {"ids": {"values": [exclude_id]}}
            }
        },
        "_source": ["content"]
    })
    req = urllib.request.Request(f"{OS_URL}/health-articles/_search",
        data=body.encode(), headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    for h in resp["hits"]["hits"]:
        doc = h["_source"]["content"]
        if validate_negative(query, doc):
            return doc
    return None


def generate_ood_negative():
    return ollama(OOD_PROMPT, temperature=0.9, max_tokens=60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=49, help="Number of docs to sample")
    p.add_argument("--queries-per-doc", type=int, default=5, help="Number of queries to generate per doc")
    p.add_argument("--output", default="data/train_v2.jsonl")
    args = p.parse_args()

    docs = fetch_docs(args.limit)
    print(f"Fetched {len(docs)} docs from health-articles index")

    samples = []
    for i, (doc_id, content) in enumerate(docs):
        queries = generate_queries(content, args.queries_per_doc)
        if not queries:
            print(f"  [{i+1}/{len(docs)}] SKIP (no queries generated)")
            continue
        for query in queries:
            hard_neg = find_hard_negative(query, doc_id)
            if not hard_neg:
                continue
            easy_neg = generate_ood_negative()
            samples.append({"query": query, "pos": content, "negs": [hard_neg, easy_neg]})
        print(f"  [{i+1}/{len(docs)}] {len(queries)} queries → {query[:60]}")

    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nWrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
