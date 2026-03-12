#!/usr/bin/env python3
"""
Step 4: Export fine-tuned model to TorchScript for OpenSearch deployment.

Wraps the fine-tuned model in a TorchScript-compatible module and packages
it into a zip file that OpenSearch ML Commons can load as a SPARSE_ENCODING
model.

Usage:
    docker run --rm -v $(pwd):/workspace -w /workspace sparse-ft \\
        python export_torchscript.py finetuned_model model.zip

What it does:
    - Loads the fine-tuned model from the specified directory
    - Wraps it in a SparseEncodingWrapper with the expected interface
    - Traces the model with TorchScript
    - Packages model.pt + tokenizer files into a zip
"""
import torch, torch.nn.functional as F, os, zipfile, sys, shutil
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Dict

model_path = sys.argv[1] if len(sys.argv) > 1 else "finetuned_model"
output_zip = sys.argv[2] if len(sys.argv) > 2 else "model.zip"
tmp = "torchscript_tmp"
if os.path.exists(tmp):
    shutil.rmtree(tmp)
os.makedirs(tmp)

print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval()


class SparseEncodingWrapper(torch.nn.Module):
    """Wrapper that matches OpenSearch SPARSE_ENCODING expected interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        pooled = torch.max(logits * attention_mask.unsqueeze(-1), dim=1).values
        sparse = torch.log1p(torch.relu(pooled))
        return {"output": sparse}


wrapper = SparseEncodingWrapper(base_model).eval()

print("Tracing model...")
dummy = tokenizer("hello world test sentence", return_tensors="pt", return_token_type_ids=False, padding="max_length", max_length=32)
dummy_dict = {"input_ids": dummy["input_ids"], "attention_mask": dummy["attention_mask"]}
scripted = torch.jit.trace(wrapper, (dummy_dict,), strict=False)
scripted.save(os.path.join(tmp, "model.pt"))

# Copy tokenizer files
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "config.json"]:
    src = os.path.join(model_path, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(tmp, f))

print(f"Creating {output_zip}...")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(tmp):
        zf.write(os.path.join(tmp, f), f)
        print(f"  {f} ({os.path.getsize(os.path.join(tmp, f)) / 1024 / 1024:.1f}MB)")

print(f"Done: {os.path.getsize(output_zip) / 1024 / 1024:.1f}MB")
