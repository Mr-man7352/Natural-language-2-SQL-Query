"""
main.py — Entry point for the NL→SQL project.

Usage:
  python main.py --train          # Download data and fine-tune the model
  python main.py --infer          # Run the interactive inference loop
  python main.py --train --infer  # Do both in sequence
"""

import argparse
import sys

# Allow importing from src/ without installing the package
sys.path.insert(0, "src")

from dataset import SQLDataset
from download import download_and_extract
from inference import generate_sql, is_valid_sql, load_model
from preprocess import build_schema, create_training_examples
from train import train
from transformers import T5Tokenizer
from config import MODEL_NAME


def run_training() -> None:
    """Download data, build training pairs, tokenize, and fine-tune."""
    download_and_extract()

    print("Building training examples...")
    pairs = create_training_examples()
    print(f"  {len(pairs)} examples ready. Sample:\n  {pairs[0]}\n")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    dataset = SQLDataset(pairs, tokenizer)

    print("Starting training...")
    train(dataset)


def run_inference() -> None:
    """Interactive loop: user provides a schema and question, model returns SQL."""
    tokenizer, model, device = load_model()

    print("\nNL → SQL inference  (type 'exit' to quit)\n")

    while True:
        schema = input("Table schema (e.g. 'name(text), age(real), salary(real)'): ").strip()
        if schema.lower() == "exit":
            break

        question = input("Question: ").strip()
        if question.lower() == "exit":
            break

        sql = generate_sql(tokenizer, model, schema, question, device)

        print(f"\nGenerated SQL:\n  {sql}")
        if not is_valid_sql(sql):
            print("  ⚠ Warning: the generated query may not be valid SQL.")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural Language to SQL — train and/or run inference."
    )
    parser.add_argument("--train", action="store_true", help="Download data and fine-tune the model.")
    parser.add_argument("--infer", action="store_true", help="Run the interactive SQL generation loop.")
    args = parser.parse_args()

    if not args.train and not args.infer:
        parser.print_help()
        sys.exit(0)

    if args.train:
        run_training()

    if args.infer:
        run_inference()


if __name__ == "__main__":
    main()
