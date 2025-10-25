"""Unified evaluation script for all retrieval methods."""

import argparse
import logging
import yaml
import torch
import numpy as np
from pathlib import Path

from baselines.retrievers import GRAFTRetriever, ZeroShotRetriever, BM25Retriever
from graft.eval.utils import prepare_queries, evaluate_retrieval
from graft.utils import get_knn_suffix

logger = logging.getLogger(__name__)


def load_embeddings(embeddings_path, sampled_indices):
    """Load embeddings and slice if sampling."""
    if sampled_indices is not None:
        logger.info(f"Loading embeddings and slicing to {len(sampled_indices)} nodes")
        full_embeddings = np.load(embeddings_path, mmap_mode='r')
        return full_embeddings[sampled_indices].copy()
    return np.load(embeddings_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods")
    parser.add_argument(
        "--method",
        type=str,
        default="graft",
        choices=["graft", "zero-shot", "bm25"],
        help="Retrieval method to evaluate (default: graft)",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--encoder-path", type=str, help="Path to encoder checkpoint (overrides config)"
    )
    parser.add_argument(
        "--model-name", type=str, help="HuggingFace model name (for zero-shot)"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to precomputed corpus embeddings (for zero-shot only)",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine split and paths from config
    split = config["eval"]["split"]
    output_dir = Path(config["experiment"]["output_dir"])
    suffix = get_knn_suffix(config)

    # Get encoder path from config if not provided via CLI
    if args.method == "graft" and not args.encoder_path:
        checkpoint = config["eval"]["checkpoint"]
        args.encoder_path = str(output_dir / f"encoder_{checkpoint}{suffix}.pt")
        logger.info(f"Using encoder from config: {args.encoder_path}")

    # Generate output path
    output_path = output_dir / f"results_{split}{suffix}.json"
    logger.info(f"Results will be saved to: {output_path}")

    logger.info(f"Loading graph...")
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

    corpus_size = config["eval"].get("corpus_size")
    corpus_sample_seed = config["eval"].get("corpus_sample_seed", 42)

    if corpus_size is not None:
        num_nodes = len(graph.node_text)
        corpus_size = min(corpus_size, num_nodes)

        indices_cache = graph_dir / f"eval_corpus_indices_{corpus_size}.npy"
        if indices_cache.exists():
            logger.info(f"Loading sampled corpus indices from {indices_cache}")
            sampled_indices = np.load(indices_cache)
        else:
            logger.info(f"Sampling {corpus_size}/{num_nodes} corpus nodes with seed={corpus_sample_seed}")
            rng = np.random.default_rng(corpus_sample_seed)
            sampled_indices = rng.choice(num_nodes, size=corpus_size, replace=False)
            sampled_indices = np.sort(sampled_indices)
            np.save(indices_cache, sampled_indices)
            logger.info(f"Saved sampled indices to {indices_cache}")

        sampled_node_texts = [graph.node_text[i] for i in sampled_indices]
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}
        sampled_doc_id_to_node_ids = {
            doc_id: [old_to_new_idx[old_idx] for old_idx in node_ids if old_idx in old_to_new_idx]
            for doc_id, node_ids in graph.doc_id_to_node_ids.items()
        }
        sampled_doc_id_to_node_ids = {
            doc_id: node_ids for doc_id, node_ids in sampled_doc_id_to_node_ids.items() if node_ids
        }

        logger.info(f"Sampled corpus: {len(sampled_node_texts)} nodes, {len(sampled_doc_id_to_node_ids)} doc_ids")
    else:
        logger.info("Using full corpus for evaluation")
        sampled_indices = None
        sampled_node_texts = graph.node_text
        sampled_doc_id_to_node_ids = graph.doc_id_to_node_ids

    logger.info(f"Preparing queries from {split} split...")
    queries = prepare_queries(split, sampled_doc_id_to_node_ids)
    logger.info(f"Loaded {len(queries)} queries")

    topk = config["index"]["topk"]

    if args.method == "graft":
        logger.info(f"Loading GRAFT model and encoding corpus...")
        embeddings = load_embeddings(
            str(output_dir / f"embeddings_{split}{suffix}.npy"),
            sampled_indices
        )
        retriever = GRAFTRetriever(args.encoder_path, embeddings, config)
        method_name = "GRAFT"

    elif args.method == "zero-shot":
        embeddings = load_embeddings(args.embeddings, sampled_indices)
        retriever = ZeroShotRetriever(args.model_name, embeddings, config)
        method_name = f"ZeroShot-{args.model_name.split('/')[-1]}"

    elif args.method == "bm25":
        retriever = BM25Retriever(sampled_node_texts)
        method_name = "BM25"

    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info(f"Evaluating {method_name}...")

    num_gpus = torch.cuda.device_count()
    eval_batch_size = int(config["encoder"]["eval_batch_size"])
    if eval_batch_size <= 0:
        raise ValueError(
            f"encoder.eval_batch_size must be positive; got {eval_batch_size}"
        )

    # Batch size is total for queries (DataParallel splits encoding across GPUs automatically)
    if args.method == "bm25":
        eval_batch_size = 1
        logger.info(f"BM25 evaluation, batch_size={eval_batch_size}")
    elif num_gpus > 1:
        logger.info(
            f"Multi-GPU evaluation: {num_gpus} GPUs, query batch_size={eval_batch_size} total (~{eval_batch_size // num_gpus} per GPU for encoding)"
        )
    else:
        logger.info(f"Single device evaluation, batch_size={eval_batch_size}")

    evaluate_retrieval(
        queries,
        retriever.search,
        topk,
        method_name,
        str(output_path),
        config["eval"],
        batch_size=eval_batch_size,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
