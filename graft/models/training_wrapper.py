"""Training wrapper inheriting SentenceTransformer with gradient-preserving encode."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.util import batch_to_device, truncate_embeddings
from torch import Tensor
from tqdm.autonotebook import trange

logger = logging.getLogger(__name__)


class EncoderTrainingWrapper(SentenceTransformer):
    """SentenceTransformer subclass that mirrors encode() but keeps gradients."""

    def __init__(self, model_name: str, max_length: int, normalize: bool):
        super().__init__(model_name)
        self.max_seq_length = max_length
        self._train_normalize_default = normalize

    def encode_with_grad(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool | None = None,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        """Match SentenceTransformer.encode but keep gradients for single-process flows."""
        if normalize_embeddings is None:
            normalize_embeddings = self._train_normalize_default

        if self.device.type == "hpu" and not getattr(self, "is_hpu_graph_enabled", False):
            import habana_frameworks.torch as ht  # type: ignore

            if hasattr(ht, "hpu") and hasattr(ht.hpu, "wrap_in_hpu_graph"):
                ht.hpu.wrap_in_hpu_graph(self, disable_tensor_cache=True)
                self.is_hpu_graph_enabled = True  # type: ignore[attr-defined]

        was_training = self.training
        self.eval()

        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        model_kwargs = self.get_model_kwargs()
        if unused_kwargs := set(kwargs) - set(model_kwargs) - {"task"}:
            raise ValueError(
                f"{self.__class__.__name__}.encode_with_grad() received unused kwargs: {list(unused_kwargs)}. "
                + (
                    f"Valid kwargs: {model_kwargs}."
                    if model_kwargs
                    else f"As per {self.__class__.__name__}.get_model_kwargs(), this model does not accept additional kwargs."
                )
            )

        if pool is not None or (isinstance(device, list) and len(device) > 0):
            result = self._encode_multi_process(
                sentences,
                show_progress_bar=show_progress_bar,
                input_was_string=input_was_string,
                pool=pool,
                device=device,
                chunk_size=chunk_size,
                prompt_name=prompt_name,
                prompt=prompt,
                batch_size=batch_size,
                output_value=output_value,
                precision=precision,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings,
                truncate_dim=truncate_dim,
                **kwargs,
            )
            if was_training:
                self.train()
            return result

        allowed_precisions = {"float32", "int8", "uint8", "binary", "ubinary"}
        if precision and precision not in allowed_precisions:
            raise ValueError(f"Precision {precision!r} is not supported")

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    ) from exc
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features: dict[str, Any] = {}
        if prompt:
            sentences = [prompt + sentence for sentence in sentences]
            length = self._get_prompt_length(prompt, **kwargs)
            if length is not None:
                extra_features["prompt_length"] = length

        if device is None:
            device = self.device

        self.to(device)

        truncate_dim = truncate_dim if truncate_dim is not None else self.truncate_dim

        all_embeddings: list[Any] = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        try:
            for start_index in trange(
                0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
            ):
                sentences_batch = sentences_sorted[start_index : start_index + batch_size]
                features = self.tokenize(sentences_batch, **kwargs)
                if self.device.type == "hpu":
                    if "input_ids" in features:
                        curr_tokenize_len = features["input_ids"].shape
                        additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]
                        features["input_ids"] = torch.cat(
                            (
                                features["input_ids"],
                                torch.ones((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                            ),
                            -1,
                        )
                        features["attention_mask"] = torch.cat(
                            (
                                features["attention_mask"],
                                torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                            ),
                            -1,
                        )
                        if "token_type_ids" in features:
                            features["token_type_ids"] = torch.cat(
                                (
                                    features["token_type_ids"],
                                    torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                                ),
                                -1,
                            )

                features = batch_to_device(features, device)
                features.update(extra_features)

                out_features = self.forward(features, **kwargs)
                if self.device.type == "hpu":
                    out_features = copy.deepcopy(out_features)

                if truncate_dim:
                    out_features["sentence_embedding"] = truncate_embeddings(
                        out_features["sentence_embedding"], truncate_dim
                    )

                if output_value == "token_embeddings":
                    embeddings: list[Any] = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:
                    embeddings = []
                    for idx in range(len(out_features["sentence_embedding"])):
                        batch_item = {}
                        for name, value in out_features.items():
                            try:
                                batch_item[name] = value[idx]
                            except TypeError:
                                batch_item[name] = value
                        embeddings.append(batch_item)
                else:
                    embeddings = out_features[output_value]
                    if normalize_embeddings:
                        embeddings = F.normalize(embeddings, p=2, dim=1)

                    if convert_to_numpy:
                        embeddings = embeddings.detach().cpu()

                all_embeddings.extend(embeddings)

            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

            if all_embeddings and precision and precision != "float32":
                all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

            if convert_to_tensor:
                if len(all_embeddings):
                    if isinstance(all_embeddings, np.ndarray):
                        all_embeddings = torch.from_numpy(all_embeddings)
                    else:
                        all_embeddings = torch.stack(all_embeddings)
                else:
                    all_embeddings = torch.tensor([], device=self.device)
            elif convert_to_numpy:
                if not isinstance(all_embeddings, np.ndarray):
                    if all_embeddings and isinstance(all_embeddings[0], torch.Tensor) and all_embeddings[0].dtype == torch.bfloat16:
                        all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                    else:
                        all_embeddings = np.asarray(
                            [
                                emb.numpy() if isinstance(emb, torch.Tensor) else emb
                                for emb in all_embeddings
                            ]
                        )
            elif isinstance(all_embeddings, np.ndarray):
                all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

            if input_was_string:
                all_embeddings = all_embeddings[0]

            return all_embeddings
        finally:
            if was_training:
                self.train()

    def get_sentence_transformer(self) -> SentenceTransformer:
        """Return self for evaluation (standard encode still uses inference mode)."""
        return self
