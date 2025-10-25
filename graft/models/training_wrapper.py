"""Training wrapper for encoder that preserves gradients during forward pass."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states, attention_mask):
    """Mean pooling with attention mask (E5 standard)."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EncoderTrainingWrapper(nn.Module):
    """Wrapper around HF transformer for training with gradients.

    Matches SentenceTransformer.encode() behavior but preserves gradients.
    For evaluation, use the SentenceTransformer directly with .encode().
    """

    def __init__(self, model_name, max_length, normalize):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.normalize = normalize

    def tokenize(self, texts):
        """Tokenize texts (no gradients needed)."""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def forward(self, batch_dict):
        """Forward pass with gradient tracking.

        Args:
            batch_dict: Dict with input_ids, attention_mask, etc. from tokenizer

        Returns:
            Embeddings tensor (batch_size, hidden_dim)
        """
        outputs = self.model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        if self.normalize:
            embeddings = nn.functional.normalize(embeddings)

        return embeddings

    def get_sentence_transformer(self):
        """Convert to SentenceTransformer for evaluation/saving.

        Returns:
            SentenceTransformer with the trained weights
        """
        from sentence_transformers import SentenceTransformer

        # Save HF model temporarily
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.save_pretrained(tmp_dir)
            self.tokenizer.save_pretrained(tmp_dir)

            # Load as SentenceTransformer (will auto-detect pooling from config)
            st_model = SentenceTransformer(tmp_dir)
            st_model.max_seq_length = self.max_length

        return st_model
