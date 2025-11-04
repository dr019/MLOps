import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import numpy as np
import logging

class TextClassificationDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_length: int):
    self.texts = list(texts)
    self.labels = list(labels) if labels is not None else None
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx]) if self.texts[idx] is not None else ""
    encoded = self.tokenizer(
      text,
      truncation=True,
      padding="max_length",
      max_length=self.max_length,
      return_tensors="pt",
    )
    item = {k: v.squeeze(0) for k, v in encoded.items()}
    if self.labels is not None:
      item["labels"] = int(self.labels[idx])
    return item

def validate_dataframe(df: pd.DataFrame, text_col: str, label_col: str, logger: logging.Logger):
  required_cols = {text_col, label_col}
  missing = required_cols - set(df.columns)
  if missing:
    raise ValueError(f"Missing required columns: {missing}")

  if not pd.api.types.is_string_dtype(df[text_col]):
    logger.warning(f"Column {text_col} is not strictly string dtype; will be cast to str in dataset.")

  if not pd.api.types.is_integer_dtype(df[label_col]):
    try:
      df[label_col] = df[label_col].astype(int)
    except Exception as e:
      raise ValueError("Label column must be convertible to int.") from e

  unique = set(df[label_col].unique())
  if not unique.issubset({0, 1}):
    raise ValueError("Labels must be binary {0,1} for this project.")

  if df[text_col].isna().mean() > 0.05:
    logger.warning("More than 5% texts are NaN; they will be replaced with empty string during tokenization.")
    
def load_and_split(
  csv_path: str,
  text_col: str,
  label_col: str,
  val_size: float,
  random_seed: int,
  logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:

  df = pd.read_csv(csv_path)
  validate_dataframe(df, text_col, label_col, logger)

  train_df, val_df = train_test_split(
    df,
    test_size=val_size,
    random_state=random_seed,
    stratify=df[label_col],
  )
        
  return train_df.reset_index(drop=True), val_df.reset_index(drop=True)