"""Export encoder-only embeddings for entire corpus."""

import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from graft.models.encoder import load_trained_encoder, load_zero_shot_encoder
from graft.eval.build_faiss import build_faiss_index_from_embeddings

logger = logging.getLogger(__name__)


def embed_corpus(encoder_path, config, output_path, index_path=None, cached_embeddings_path=None):
    """Embed corpus with encoder.

    Args:
        encoder_path: Path to trained checkpoint (.pt) OR HuggingFace model name
        config: Config dict
        output_path: Path to save embeddings .npy file
        index_path: Path to save FAISS index directly (optional)
        cached_embeddings_path: Path to cached embeddings to reuse (optional, for zero-shot)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_dir = Path(config["data"]["graph_dir"])
    graph_name = config["data"]["graph_name"]
    semantic_k = config["data"].get("semantic_k")
    knn_only = config["data"].get("knn_only", False)

    if semantic_k is None:
        graph_path = graph_dir / f"{graph_name}.pt"
    else:
        suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
        graph_path = graph_dir / f"{graph_name}{suffix}.pt"

    is_checkpoint = Path(encoder_path).exists()

    if cached_embeddings_path and Path(cached_embeddings_path).exists():
        logger.info(f"Loading cached embeddings from {cached_embeddings_path}")
        all_embeddings = np.load(cached_embeddings_path)
        logger.info(f"Loaded embeddings: {all_embeddings.shape}")
    else:
        if is_checkpoint:
            logger.info(f"Loading trained encoder from {encoder_path}")
            encoder = load_trained_encoder(encoder_path, config, device)
        else:
            logger.info(f"Loading zero-shot model: {encoder_path}")
            encoder = load_zero_shot_encoder(encoder_path, config, device)

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            encoder = torch.nn.DataParallel(encoder)

        logger.info(f"Loading graph from {graph_path}")
        graph = torch.load(str(graph_path), weights_only=False)
        corpus_texts = graph.node_text

        embeddings = []

        batch_size = config["data"]["batch_size"]
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logger.info(
                f"Batch size: {batch_size} total (split across {num_gpus} GPUs = ~{batch_size // num_gpus} per GPU)"
            )
        else:
            logger.info(f"Batch size: {batch_size}")

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

                embeddings.append(batch_embeds.cpu())

        all_embeddings = torch.cat(embeddings, dim=0).numpy()

    if index_path:
        logger.info(f"Building FAISS index directly (skipping embeddings save)")
        build_faiss_index_from_embeddings(all_embeddings, config, index_path)
        logger.info(
            f"Embeddings shape: {all_embeddings.shape}, Index saved to: {index_path}"
        )
    else:
        all_embeddings = all_embeddings.astype(np.float16)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, all_embeddings)
        logger.info(f"Saved embeddings: {all_embeddings.shape} (float16)")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Embed corpus with encoder")
    parser.add_argument(
        "encoder_path", type=str, help="Path to checkpoint or HF model name"
    )
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument(
        "output",
        type=str,
        help="Path to save embeddings .npy (or dummy if --index-path)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        help="Build FAISS index directly (skips saving embeddings)",
    )
    parser.add_argument(
        "--cached-embeddings",
        type=str,
        help="Path to cached embeddings (reuse for zero-shot)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    embed_corpus(
        args.encoder_path,
        config,
        args.output,
        index_path=args.index_path,
        cached_embeddings_path=args.cached_embeddings,
    )
