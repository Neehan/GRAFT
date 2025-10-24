"""Unified evaluation script for all retrieval methods."""

import argparse
import logging
import yaml
import torch
from pathlib import Path
from datasets import load_dataset

from baselines.retrievers import GRAFTRetriever, ZeroShotRetriever, BM25Retriever
from graft.eval.utils import prepare_queries, evaluate_retrieval

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["graft", "zero-shot", "bm25"],
        help="Retrieval method to evaluate",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save results JSON"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (train/validation/test)",
    )
    parser.add_argument(
        "--encoder-path", type=str, help="Path to encoder checkpoint (for GRAFT)"
    )
    parser.add_argument(
        "--model-name", type=str, help="HuggingFace model name (for zero-shot)"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to precomputed corpus embeddings (for GRAFT/zero-shot)",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    logger.info(f"Preparing queries from {args.split} split...")
    queries = prepare_queries(args.split, graph.doc_id_to_id)
    logger.info(f"Loaded {len(queries)} queries")

    topk = config["index"]["topk"]

    if args.method == "graft":
        if not args.encoder_path or not args.embeddings:
            raise ValueError("GRAFT requires --encoder-path and --embeddings")
        retriever = GRAFTRetriever(args.encoder_path, args.embeddings, config)
        method_name = "GRAFT"

    elif args.method == "zero-shot":
        if not args.model_name or not args.embeddings:
            raise ValueError("Zero-shot requires --model-name and --embeddings")
        retriever = ZeroShotRetriever(args.model_name, args.embeddings, config)
        method_name = f"ZeroShot-{args.model_name.split('/')[-1]}"

    elif args.method == "bm25":
        retriever = BM25Retriever(graph.node_text)
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
        args.output,
        batch_size=eval_batch_size,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
