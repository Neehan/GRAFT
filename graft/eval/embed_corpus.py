"""Export encoder-only embeddings for entire corpus (no GNN at inference)."""

import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from graft.models.encoder import Encoder

logger = logging.getLogger(__name__)


def embed_corpus(encoder_path, config, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        model_name=config["encoder"]["model_name"],
        max_len=config["encoder"]["max_len"],
        pool=config["encoder"]["pool"],
        proj_dim=config["encoder"]["proj_dim"],
        freeze_layers=0
    )

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    graph = torch.load(config["data"]["graph_path"])
    corpus_texts = graph.node_text

    embeddings = []
    batch_size = 128

    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding corpus"):
        batch = corpus_texts[i:i + batch_size]
        batch_embeds = encoder.encode(batch, device)
        embeddings.append(batch_embeds.cpu().numpy())

    all_embeddings = np.vstack(embeddings)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_embeddings)
    logger.info(f"Saved embeddings: {all_embeddings.shape}")
