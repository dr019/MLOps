import logging
import sys

def setup_logger(name: str = "train", level: int = logging.INFO) -> logging.Logger:
  logger = logging.getLogger(name)
  logger.setLevel(level)

  if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    
  return logger