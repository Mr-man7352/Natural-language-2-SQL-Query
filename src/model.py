"""T5-based sequence-to-sequence model wrapper for NL2SQL.

Provides a thin wrapper around a pre-trained (or fine-tuned) T5 model and its
tokenizer so that the rest of the pipeline can interact with a single object.
"""

from __future__ import annotations

from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)


class NL2SQLModel:
    """Wraps a T5 model and tokenizer for NL2SQL inference and fine-tuning.

    Parameters
    ----------
    model_name_or_path:
        A Hugging Face model identifier (e.g. ``"t5-base"``) or a local
        directory containing a saved model.
    max_input_length:
        Maximum number of tokens for the encoder input sequence.
    max_target_length:
        Maximum number of tokens for the decoder output sequence.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).  If ``None`` the
        device is chosen automatically.
    """

    def __init__(
        self,
        model_name_or_path: str = "t5-base",
        max_input_length: int = 512,
        max_target_length: int = 256,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.model_name_or_path = model_name_or_path
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading tokenizer from '%s'.", model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

        logger.info("Loading model from '%s' onto device '%s'.", model_name_or_path, self.device)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str] | str,
        num_beams: int = 4,
        **generate_kwargs: Any,
    ) -> list[str]:
        """Generate SQL queries for one or more natural-language prompts.

        Parameters
        ----------
        prompts:
            A single prompt string or a list of prompt strings.
        num_beams:
            Beam width for beam-search decoding.
        **generate_kwargs:
            Additional keyword arguments forwarded to
            ``model.generate()``.

        Returns
        -------
        list[str]
            Decoded SQL predictions (one per input prompt).
        """
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = self.tokenizer(
            prompts,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_target_length,
                num_beams=num_beams,
                early_stopping=True,
                **generate_kwargs,
            )

        predictions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return predictions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """Save the model and tokenizer to *output_dir*."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved to '%s'.", output_dir)

    @classmethod
    def load(
        cls,
        model_dir: str,
        max_input_length: int = 512,
        max_target_length: int = 256,
        device: str | None = None,
    ) -> "NL2SQLModel":
        """Load a previously saved model from *model_dir*."""
        return cls(
            model_name_or_path=model_dir,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            device=device,
        )
