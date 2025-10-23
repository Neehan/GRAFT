"""HuggingFace transformer encoder with CLS/mean pooling for dual-encoder retrieval."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Encoder(nn.Module):
    def __init__(self, model_name, max_len, pool, proj_dim, freeze_layers):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.pool = pool
        self.proj_dim = proj_dim

        if freeze_layers > 0:
            for param in list(self.model.parameters())[:freeze_layers]:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.pool == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        elif self.pool == "mean":
            embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool}")

        return embeddings

    def encode(self, texts, device):
        encoded = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)

        return embeddings
