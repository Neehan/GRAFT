"""Export encoder-only embeddings for entire corpus (no GNN at inference)."""

import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from graft.models.encoder import load_trained_encoder, load_zero_shot_encoder

logger = logging.getLogger(__name__)


def embed_corpus(encoder_path, config, output_path):
    """Embed corpus with encoder.

    Args:
        encoder_path: Path to trained checkpoint (.pt) OR HuggingFace model name
        config: Config dict
        output_path: Path to save embeddings .npy file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_checkpoint = Path(encoder_path).exists()

    if is_checkpoint:
        logger.info(f"Loading trained encoder from {encoder_path}")
        encoder = load_trained_encoder(encoder_path, config, device)
    else:
        logger.info(f"Loading zero-shot model: {encoder_path}")
        encoder = load_zero_shot_encoder(encoder_path, config, device)

    graph = torch.load(config["data"]["graph_path"], weights_only=False)
    corpus_texts = graph.node_text

    embeddings = []
    batch_size = config["encoder"].get("batch_size", 128)

    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding corpus"):
        batch = corpus_texts[i : i + batch_size]
        batch_embeds = encoder.encode(batch, device)
        embeddings.append(batch_embeds.cpu().numpy())

    all_embeddings = np.vstack(embeddings)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_embeddings)
    logger.info(f"Saved embeddings: {all_embeddings.shape}")
