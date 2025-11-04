import os
import pandas as pd
import numpy as np
import tempfile
import logging
from src.data import validate_dataframe, load_and_split

def test_validate_dataframe_ok():
  df = pd.DataFrame({"text": ["a","b"], "target": [0,1]})
  logger = logging.getLogger("test")
  validate_dataframe(df, "text", "target", logger)

def test_validate_dataframe_bad_label():
  df = pd.DataFrame({"text": ["a","b"], "target": [0,2]})
  logger = logging.getLogger("test")
  try:
    validate_dataframe(df, "text", "target", logger)
    assert False, "Should have raised"
  except ValueError:
    assert True

def test_load_and_split_shapes():
  df = pd.DataFrame({"text": [f"t{i}" for i in range(100)], "target": [i%2 for i in range(100)]})
  with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "train.csv")
    df.to_csv(path, index=False)
    train_df, val_df = load_and_split(path, "text", "target", 0.2, 42, logging.getLogger("test"))
    assert len(train_df) == 80
    assert len(val_df) == 20
    assert set(train_df.columns) == {"text","target"}