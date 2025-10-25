"""Training wrapper inheriting SentenceTransformer with gradient-preserving encode."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import trange

logger = logging.getLogger(__name__)


class EncoderTrainingWrapper(SentenceTransformer):
    """SentenceTransformer subclass that mirrors encode() but keeps gradients."""

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
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
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
        """SentenceTransformer.encode analogue without inference_mode or no_grad."""
        if pool is not None or (
            isinstance(device, Sequence) and not isinstance(device, (str, torch.device))
        ):
            raise NotImplementedError("encode_with_grad does not support multi-process pools.")

        if chunk_size is not None:
            raise NotImplementedError("encode_with_grad does not support chunk_size parameter.")

        if precision and precision not in {"float32"}:
            raise ValueError(f"Precision {precision!r} is not supported for gradient computation.")

        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]  # type: ignore[list-item]
            input_was_string = True

        sentences = list(sentences)  # type: ignore[arg-type]

        if not sentences:
            return self._handle_empty_input(convert_to_tensor, convert_to_numpy, device)

        model_kwargs = self.get_model_kwargs()
        unused_kwargs = set(kwargs) - set(model_kwargs) - {"task"}
        if unused_kwargs:
            raise ValueError(
                f"{self.__class__.__name__}.encode_with_grad() received unused kwargs: {sorted(unused_kwargs)}. "
                f"Valid kwargs: {sorted(model_kwargs)}."
            )

        if normalize_embeddings is None:
            normalize_embeddings = self._train_normalize_default

        prompt, extra_features = self._resolve_prompt(prompt, prompt_name, kwargs, sentences)

        if device is None:
            device = self.device

        self.to(device)

        truncate_dim = truncate_dim if truncate_dim is not None else self.truncate_dim

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        all_embeddings: list[Any] = []

        for start_index in trange(
            0,
            len(sentences_sorted),
            batch_size,
            desc="Batches",
            disable=not show_progress_bar,
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch, **kwargs)
            features = util.batch_to_device(features, device)
            features.update(extra_features)

            out_features = self.forward(features, **kwargs)

            if truncate_dim:
                out_features["sentence_embedding"] = util.truncate_embeddings(
                    out_features["sentence_embedding"], truncate_dim
                )

            embeddings = self._gather_embeddings(
                out_features,
                output_value=output_value,
                normalize=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
            )
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        all_embeddings = self._finalize_embeddings(
            all_embeddings,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
        )

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _handle_empty_input(
        self,
        convert_to_tensor: bool,
        convert_to_numpy: bool,
        device: torch.device | str | None,
    ):
        dim = self.get_sentence_embedding_dimension()
        if convert_to_tensor:
            return torch.empty(
                (0, dim),
                device=self.device if device is None else device,
            )
        if convert_to_numpy:
            return np.empty((0, dim))
        return []

    def _resolve_prompt(
        self,
        prompt: str | None,
        prompt_name: str | None,
        kwargs: dict[str, Any],
        sentences: list[str],
    ) -> tuple[str | None, dict[str, Any]]:
        extra_features: dict[str, Any] = {}
        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found. Available: {list(self.prompts.keys())}"
                    ) from None
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name)
        elif prompt_name is not None:
            logger.warning(
                "encode_with_grad received both prompt and prompt_name; ignoring prompt_name."
            )

        if prompt:
            for idx, sentence in enumerate(sentences):
                sentences[idx] = prompt + sentence
            length = self._get_prompt_length(prompt, **kwargs)
            if length is not None:
                extra_features["prompt_length"] = length

        return prompt, extra_features

    def _gather_embeddings(
        self,
        out_features: dict[str, Any],
        *,
        output_value: str | None,
        normalize: bool,
        convert_to_numpy: bool,
    ) -> Iterable[Any]:
        if output_value == "token_embeddings":
            embeddings = []
            for token_emb, attention in zip(
                out_features[output_value],
                out_features["attention_mask"],
            ):
                last_mask_id = len(attention) - 1
                while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                    last_mask_id -= 1
                embeddings.append(token_emb[0 : last_mask_id + 1])
            return embeddings

        if output_value is None:
            embeddings = []
            sentence_embeddings = out_features["sentence_embedding"]
            for idx in range(len(sentence_embeddings)):
                batch_item = {}
                for name, value in out_features.items():
                    try:
                        batch_item[name] = value[idx]
                    except TypeError:
                        batch_item[name] = value
                embeddings.append(batch_item)
            return embeddings

        embeddings = out_features[output_value]
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        if convert_to_numpy:
            embeddings = embeddings.detach().cpu()

        return embeddings

    def _finalize_embeddings(
        self,
        embeddings: list[Any],
        *,
        convert_to_tensor: bool,
        convert_to_numpy: bool,
    ):
        if convert_to_tensor:
            if embeddings:
                if isinstance(embeddings, np.ndarray):
                    return torch.from_numpy(embeddings)
                return torch.stack(embeddings)
            return torch.empty(
                (0, self.get_sentence_embedding_dimension()),
                device=self.device,
            )

        if convert_to_numpy and not isinstance(embeddings, np.ndarray):
            if embeddings and getattr(embeddings[0], "dtype", None) == torch.bfloat16:
                return np.asarray([emb.float().numpy() for emb in embeddings])
            return np.asarray([emb.detach().cpu().numpy() for emb in embeddings])

        if not convert_to_numpy and isinstance(embeddings, np.ndarray):
            embeddings = [torch.from_numpy(embedding) for embedding in embeddings]

        return embeddings

    def get_sentence_transformer(self) -> SentenceTransformer:
        """Return self for evaluation (standard encode still uses inference mode)."""
        return self
