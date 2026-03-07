"""
dataset.py — PyTorch Dataset wrapper for tokenized NL→SQL pairs.
"""

from typing import List, Tuple

from torch.utils.data import Dataset
from transformers import T5Tokenizer

from config import MAX_INPUT_LENGTH


class SQLDataset(Dataset):
    """Tokenizes (input, target) string pairs for T5 fine-tuning."""

    def __init__(self, pairs: List[Tuple[str, str]], tokenizer: T5Tokenizer) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        inp, out = self.pairs[idx]
        tokens = self.tokenizer(
            inp,
            text_target=out,
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LENGTH,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["labels"],
        }
