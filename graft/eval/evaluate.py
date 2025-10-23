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
        "--faiss-index", type=str, help="Path to FAISS index (for GRAFT/zero-shot)"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading HotpotQA {args.split} split and graph...")
    dataset = load_dataset("hotpot_qa", "distractor", split=args.split)

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

    logger.info("Preparing queries...")
    queries = prepare_queries(dataset, graph.title_to_id)
    logger.info(f"Loaded {len(queries)} queries")

    topk = config["index"]["topk"]

    if args.method == "graft":
        if not args.encoder_path or not args.faiss_index:
            raise ValueError("GRAFT requires --encoder-path and --faiss-index")
        retriever = GRAFTRetriever(args.encoder_path, args.faiss_index, config)
        method_name = "GRAFT"

    elif args.method == "zero-shot":
        if not args.model_name or not args.faiss_index:
            raise ValueError("Zero-shot requires --model-name and --faiss-index")
        retriever = ZeroShotRetriever(args.model_name, args.faiss_index, config)
        method_name = f"ZeroShot-{args.model_name.split('/')[-1]}"

    elif args.method == "bm25":
        retriever = BM25Retriever(graph.node_text)
        method_name = "BM25"

    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info(f"Evaluating {method_name}...")

    def retrieval_fn(query_text, k):
        return retriever.search(query_text, k)

    evaluate_retrieval(queries, retrieval_fn, topk, method_name, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
