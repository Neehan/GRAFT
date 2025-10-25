"""Training wrapper that exposes gradient-preserving encode."""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# Underlying encode implementation without @torch.inference_mode()
if hasattr(SentenceTransformer.encode, "__wrapped__"):
    _ENCODE_WITH_GRAD = SentenceTransformer.encode.__wrapped__
else:  # pragma: no cover - fallback for unexpected decorator behavior
    _ENCODE_WITH_GRAD = inspect.unwrap(SentenceTransformer.encode)


class EncoderTrainingWrapper(SentenceTransformer):
    """SentenceTransformer subclass with gradient-friendly encode."""

    def __init__(self, model_name: str, max_length: int, normalize: bool):
        super().__init__(model_name)
        self.max_seq_length = max_length
        self._train_normalize_default = normalize

    def encode_with_grad(
        self,
        sentences: str | Sequence[str] | np.ndarray,
        *,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: str | None = "sentence_embedding",
        precision: str = "float32",
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = True,
        device: str | torch.device | Sequence[str | torch.device] | None = None,
        normalize_embeddings: bool | None = None,
        truncate_dim: int | None = None,
        pool: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> (
        list[torch.Tensor]
        | np.ndarray
        | torch.Tensor
        | dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
    ):
        """Call SentenceTransformer.encode without inference_mode to keep gradients."""
        if normalize_embeddings is None:
            normalize_embeddings = self._train_normalize_default

        return _ENCODE_WITH_GRAD(
            self,
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            precision=precision,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
            **kwargs,
        )

    def get_sentence_transformer(self) -> SentenceTransformer:
        """Return self for evaluation (encode remains inference-mode)."""
        return self
