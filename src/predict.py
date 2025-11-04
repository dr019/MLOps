import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_for_inference(model_dir: str):
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  model = AutoModelForSequenceClassification.from_pretrained(model_dir)
  model.eval()

  return model, tokenizer

def postprocess_logits_to_label(logits: np.ndarray, threshold: float = 0.5) -> int:
  if logits.ndim == 1:
    logits = logits[None, :]
  probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

  return int(probs[0, 1] >= threshold)

def predict_text(model, tokenizer, text: str, max_length: int = 128, threshold: float = 0.5) -> int:
  inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)
  logits = outputs.logits.detach().cpu().numpy()[0]
    
  return postprocess_logits_to_label(logits, threshold)