from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

TRAIN_PATH = Path(os.getenv("TRAIN_PATH", "./data/train_data_sample500.jsonl"))
EVAL_PATH = Path(os.getenv("EVAL_PATH", "./data/eval_data.jsonl"))

def load_data(train_path: Path = TRAIN_PATH, eval_path: Path = EVAL_PATH):
    """Загружает готовые train и eval датасеты из JSONL."""
    train_data = pd.read_json(train_path, lines=True)
    eval_data = pd.read_json(eval_path, lines=True)
    return train_data, eval_data
