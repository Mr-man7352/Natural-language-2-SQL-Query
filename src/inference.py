"""
inference.py — SQL generation and validation utilities.
"""

import sqlparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import MAX_OUTPUT_LENGTH, MODEL_OUTPUT_DIR, NO_REPEAT_NGRAM_SIZE, NUM_BEAMS


def load_model() -> tuple[T5Tokenizer, T5ForConditionalGeneration, str]:
    """Load the fine-tuned tokenizer and model; return (tokenizer, model, device)."""
    tokenizer = T5Tokenizer.from_pretrained(MODEL_OUTPUT_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")
    return tokenizer, model, device


def generate_sql(
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    schema: str,
    question: str,
    device: str,
) -> str:
    """Generate a SQL query from a natural-language question and table schema."""
    input_text = f"Schema: {schema}, Question: {question}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def is_valid_sql(sql: str) -> bool:
    """Return True if sqlparse can parse the string as valid SQL."""
    return bool(sqlparse.parse(sql.strip()))
