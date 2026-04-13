#!/usr/bin/env python3
"""
Export fine-tuned model to TorchScript zip for OpenSearch ML Commons.

Usage:
    python export_torchscript.py --model biobert_finetuned --output model.zip
"""
import torch, torch.nn.functional as F, os, zipfile, sys, shutil, argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Dict


class SparseEncodingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        pooled = torch.max(logits * inputs["attention_mask"].unsqueeze(-1), dim=1).values
        return {"output": torch.log1p(torch.relu(pooled))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="biobert_finetuned")
    p.add_argument("--output", default="model.zip")
    args = p.parse_args()

    tmp = "torchscript_tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForMaskedLM.from_pretrained(args.model).eval()

    wrapper = SparseEncodingWrapper(base_model).eval()
    dummy = tokenizer("hello world", return_tensors="pt", return_token_type_ids=False, padding="max_length", max_length=32)
    dummy_dict = {"input_ids": dummy["input_ids"], "attention_mask": dummy["attention_mask"]}

    print("Tracing...")
    scripted = torch.jit.trace(wrapper, (dummy_dict,), strict=False)
    scripted.save(os.path.join(tmp, "model.pt"))

    for f in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "config.json"]:
        src = os.path.join(args.model, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp, f))

    print(f"Creating {args.output}...")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(tmp):
            zf.write(os.path.join(tmp, f), f)
            print(f"  {f} ({os.path.getsize(os.path.join(tmp, f)) / 1024 / 1024:.1f}MB)")

    print(f"Done: {os.path.getsize(args.output) / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    main()
