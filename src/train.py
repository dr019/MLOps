import argparse
import os
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data import TextClassificationDataset, load_and_split
from src.model import load_model_and_tokenizer, compute_metrics_fn

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, help="Path to YAML config")
  parser.add_argument("--verbose", action="store_true")
  return parser.parse_args()

def main():
  args = parse_args()
  cfg = load_config(args.config)
  logger = setup_logger(level=(10 if args.verbose else 20))

  set_seed(int(cfg["train_seed"]))

  model, tokenizer = load_model_and_tokenizer(
    cfg["model_pretrained_name"], int(cfg["model_num_labels"])
  )

  train_df, val_df = load_and_split(
    csv_path=cfg["data_train_csv"],
    text_col=cfg["data_text_col"],
    label_col=cfg["data_label_col"],
    val_size=float(cfg["data_val_size"]),
    random_seed=int(cfg["data_random_seed"]),
    logger=logger,
  )

  train_ds = TextClassificationDataset(texts=train_df[cfg["data_text_col"]], labels=train_df[cfg["data_label_col"]], tokenizer=tokenizer, max_length=int(cfg["model_max_length"]))

  val_ds = TextClassificationDataset(texts=val_df[cfg["data_text_col"]], labels=val_df[cfg["data_label_col"]], tokenizer=tokenizer, max_length=int(cfg["model_max_length"]))

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  training_args = TrainingArguments(
    output_dir=cfg["train_output_dir"],
    num_train_epochs=int(cfg["train_num_train_epochs"]),
    per_device_train_batch_size=int(cfg["train_per_device_train_batch_size"]),
    per_device_eval_batch_size=int(cfg["train_per_device_eval_batch_size"]),
    learning_rate=float(cfg["train_learning_rate"]),
    weight_decay=float(cfg["train_weight_decay"]),
    warmup_ratio=float(cfg["train_warmup_ratio"]),
    logging_steps=int(cfg["train_logging_steps"]),
    evaluation_strategy=cfg["train_evaluation_strategy"],
    eval_steps=int(cfg["train_eval_steps"]),
    save_strategy=cfg["train_save_strategy"],
    save_steps=int(cfg["train_save_steps"]),
    seed=int(cfg["train_seed"]),
    fp16= True if cfg["train_fp16"]=="true" else False,
    report_to=None,
    load_best_model_at_end=True,
    metric_for_best_model=cfg["train_metric_for_best_model"],
    greater_is_better= True if cfg["train_greater_is_better"]=="true" else False,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
  )

  logger.info("Starting training...")
  train_result = trainer.train()
  trainer.save_model(cfg["train_output_dir"])
  tokenizer.save_pretrained(cfg["train_output_dir"])

  logger.info("Evaluating...")
  metrics = trainer.evaluate()
  for k, v in metrics.items():
    logger.info(f"{k}: {v}")

  logger.info(f"Model and tokenizer saved to {cfg['train_output_dir']}")

if __name__ == "__main__":
    main()