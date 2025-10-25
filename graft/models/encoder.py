"""HuggingFace transformer encoder with mean pooling for dual-encoder retrieval."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class Encoder(nn.Module):
    def __init__(
        self,
        model_name,
        max_len,
        pool,
        freeze_layers,
        normalize,
        padding,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.pool = pool
        self.normalize = normalize
        self.padding = padding

        if freeze_layers > 0:
            for param in list(self.model.parameters())[:freeze_layers]:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.pool == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        elif self.pool == "mean":
            embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool}")

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def encode(self, texts, device):
        encoded = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)

        return embeddings


def load_trained_encoder(checkpoint_path, config, device):
    """Load trained encoder from checkpoint.

    Args:
        checkpoint_path: Path to encoder checkpoint (.pt file)
        config: Config dict with encoder settings
        device: torch device

    Returns:
        Loaded encoder model
    """
    encoder = Encoder(
        model_name=config["encoder"]["model_name"],
        max_len=config["encoder"]["max_len"],
        pool=config["encoder"]["pool"],
        freeze_layers=0,
        normalize=config["encoder"]["normalize"],
        padding=config["encoder"]["padding"],
    )
    encoder.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    encoder.to(device)
    encoder.eval()
    return encoder


def load_zero_shot_encoder(model_name, config, device):
    """Load zero-shot encoder from HuggingFace.

    Args:
        model_name: HuggingFace model name
        config: Config dict with encoder settings
        device: torch device

    Returns:
        Zero-shot encoder model
    """
    encoder = Encoder(
        model_name=model_name,
        max_len=config["encoder"]["max_len"],
        pool=config["encoder"]["pool"],
        freeze_layers=0,
        normalize=config["encoder"]["normalize"],
        padding=config["encoder"]["padding"],
    )
    encoder.to(device)
    encoder.eval()
    return encoder
