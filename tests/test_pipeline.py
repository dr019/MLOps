import os
import tempfile
import pandas as pd
from src.model import load_model_and_tokenizer
from src.data import TextClassificationDataset
from src.predict import postprocess_logits_to_label
from transformers import DataCollatorWithPadding
import torch
import numpy as np

def test_tokenization_and_dataset():
  model, tokenizer = load_model_and_tokenizer("prajjwal1/bert-tiny", 2)
  texts = ["fire near street", "nice sunny day"]
  labels = [1, 0]
  ds = TextClassificationDataset(texts, labels, tokenizer, max_length=32)
  sample = ds[0]
  assert "input_ids" in sample and "attention_mask" in sample and "labels" in sample
  assert sample["input_ids"].shape[0] == 32

def test_forward_pass_and_postprocess():
  model, tokenizer = load_model_and_tokenizer("prajjwal1/bert-tiny", 2)
  inputs = tokenizer("fire near street", return_tensors="pt", padding="max_length", truncation=True, max_length=32)
  with torch.no_grad():
    out = model(**inputs)
  logits = out.logits.detach().cpu().numpy()[0]
  label = postprocess_logits_to_label(logits, threshold=0.5)
  assert label in (0,1)

def test_collator_shapes():
  _, tokenizer = load_model_and_tokenizer("prajjwal1/bert-tiny", 2)
  collator = DataCollatorWithPadding(tokenizer)
  batch = [
    {"input_ids": torch.tensor([1,2,3]), "attention_mask": torch.tensor([1,1,1]), "labels": 0},
    {"input_ids": torch.tensor([1,2]), "attention_mask": torch.tensor([1,1]), "labels": 1},
  ]
  out = collator(batch)
  assert out["input_ids"].ndim == 2
  assert out["labels"].shape[0] == 2