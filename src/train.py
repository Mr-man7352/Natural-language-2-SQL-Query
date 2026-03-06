"""Training script for the NL2SQL T5 model.

Usage
-----
.. code-block:: bash

    python -m src.train --config config.yaml

The script:
1. Loads configuration from the given YAML file.
2. Loads the Spider dataset and builds schema-aware prompts.
3. Tokenizes the dataset.
4. Fine-tunes a T5 model using the Hugging Face ``Seq2SeqTrainer``.
5. Saves the best checkpoint to ``training.output_dir``.
"""

from __future__ import annotations

import argparse
import os

from src.utils import ensure_dir, get_logger, load_config, set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune T5 for NL2SQL.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def build_compute_metrics(tokenizer, max_target_length: int):
    """Return a ``compute_metrics`` function compatible with Hugging Face Trainer."""
    from src.evaluate import evaluate_predictions

    def compute_metrics(eval_preds):
        import numpy as np

        predictions, labels = eval_preds

        # Replace -100 (ignored index) with the pad token id before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        results = evaluate_predictions(decoded_preds, decoded_labels)
        # Return only scalar metrics for the Trainer
        return {
            "exact_match": results["exact_match"],
            "exact_set_match": results["exact_set_match"],
            "token_f1": results["token_f1"],
        }

    return compute_metrics


def train(config_path: str) -> None:
    """Main training entry point."""
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        T5ForConditionalGeneration,
        T5Tokenizer,
    )

    from src.data_preprocessing import preprocess_dataset

    cfg = load_config(config_path)
    set_seed(cfg["training"].get("seed", 42))

    model_name = cfg["model"]["name"]
    max_input_length = cfg["model"]["max_input_length"]
    max_target_length = cfg["model"]["max_target_length"]

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    data_cfg = cfg["data"]
    train_ds, dev_ds = preprocess_dataset(
        train_file=data_cfg["train_file"],
        dev_file=data_cfg["dev_file"],
        tables_file=data_cfg["tables_file"],
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        cache_dir=data_cfg.get("cache_dir"),
    )

    output_dir = cfg["training"]["output_dir"]
    ensure_dir(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        logging_dir=cfg["training"]["logging_dir"],
        logging_steps=cfg["training"]["logging_steps"],
        predict_with_generate=True,
        generation_max_length=max_target_length,
        fp16=cfg["training"].get("fp16", False),
        seed=cfg["training"].get("seed", 42),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    compute_metrics = build_compute_metrics(tokenizer, max_target_length)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training …")
    trainer.train()

    logger.info("Saving best model to '%s'.", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
