"""
train.py — Model training logic and custom timing callback.
"""

import time

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOGGING_STEPS,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    NUM_EPOCHS,
    SAVE_STEPS,
)
from dataset import SQLDataset


class TimingCallback(TrainerCallback):
    """Prints human-readable progress, elapsed time, and ETA during training."""

    def __init__(self) -> None:
        self.train_start: float = 0.0
        self.epoch_start: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        self.train_start = time.time()
        print(f"\n{'=' * 55}")
        print(f"  Training started — Total steps: {state.max_steps}")
        print(f"{'=' * 55}\n")

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        self.epoch_start = time.time()
        epoch = int(state.epoch or 0) + 1
        print(f"\n── Epoch {epoch} started ──")

    def on_step_end(self, args, state, control, **kwargs) -> None:
        if state.global_step % 10 == 0 and state.global_step > 0:
            elapsed = time.time() - self.train_start
            steps_done = state.global_step
            steps_left = state.max_steps - steps_done
            eta_sec = (elapsed / steps_done) * steps_left

            print(
                f"  Step {steps_done:>4}/{state.max_steps} "
                f"({100 * steps_done / state.max_steps:5.1f}%) | "
                f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}"
            )

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        epoch_elapsed = time.time() - self.epoch_start
        total_elapsed = time.time() - self.train_start
        print(
            f"\n── Epoch {int(state.epoch or 0)} done in {epoch_elapsed:.1f}s "
            f"(total: {total_elapsed:.1f}s) ──\n"
        )

    def on_train_end(self, args, state, control, **kwargs) -> None:
        total = time.time() - self.train_start
        print(f"\n{'=' * 55}")
        print(f"  Training complete!  Total time: {time.strftime('%H:%M:%S', time.gmtime(total))}")
        print(f"{'=' * 55}\n")


def train(train_dataset: SQLDataset) -> None:
    """Fine-tune T5-small on the provided dataset and save the model."""
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        disable_tqdm=True,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[TimingCallback()],
    )

    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model saved to '{MODEL_OUTPUT_DIR}'.")
