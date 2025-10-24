"""Export encoder-only embeddings for entire corpus."""

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

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        encoder = torch.nn.DataParallel(encoder)

    graph_dir = Path(config["data"]["graph_dir"])
    graph_name = config["data"]["graph_name"]
    semantic_k = config["data"].get("semantic_k")
    knn_only = config["data"].get("knn_only", False)

    if semantic_k is None:
        graph_path = graph_dir / f"{graph_name}.pt"
    else:
        suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
        graph_path = graph_dir / f"{graph_name}{suffix}.pt"

    logger.info(f"Loading graph from {graph_path}")
    graph = torch.load(str(graph_path), weights_only=False)
    corpus_texts = graph.node_text

    embeddings = []

    # Scale batch size by number of GPUs
    base_batch_size = config["data"]["batch_size"]
    batch_size = base_batch_size * max(1, torch.cuda.device_count())
    logger.info(
        f"Batch size: {batch_size} (base={base_batch_size}, GPUs={torch.cuda.device_count()})"
    )

    # Enable mixed precision for faster inference
    use_amp = config["train"]["bf16"]
    if use_amp:
        logger.info("Using mixed precision (bfloat16)")

    with torch.no_grad():
        for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding corpus"):
            batch = corpus_texts[i : i + batch_size]

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if hasattr(encoder, "module"):
                        batch_embeds = encoder.module.encode(batch, device)
                    else:
                        batch_embeds = encoder.encode(batch, device)
            else:
                if hasattr(encoder, "module"):
                    batch_embeds = encoder.module.encode(batch, device)
                else:
                    batch_embeds = encoder.encode(batch, device)

            # Move to CPU immediately to avoid GPU memory accumulation
            embeddings.append(batch_embeds.cpu())

    all_embeddings = torch.cat(embeddings, dim=0).numpy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_embeddings)
    logger.info(f"Saved embeddings: {all_embeddings.shape}")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Embed corpus with encoder")
    parser.add_argument(
        "encoder_path", type=str, help="Path to checkpoint or HF model name"
    )
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("output", type=str, help="Path to save embeddings .npy")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    embed_corpus(args.encoder_path, config, args.output)
