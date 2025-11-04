from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Tuple

def load_model_and_tokenizer(pretrained_name: str, num_labels: int):
  tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_name,
    num_labels=num_labels,
    id2label={0: "not_disaster", 1: "disaster"},
    label2id={"not_disaster": 0, "disaster": 1},
  )
  return model, tokenizer

def compute_metrics_fn(eval_pred):
  logits, labels = eval_pred
  preds = logits.argmax(axis=-1)

  return {
    "accuracy": accuracy_score(labels, preds),
    "f1": f1_score(labels, preds),
  }