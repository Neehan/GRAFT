"""Prepare HotpotQA data: build graph from HF datasets and optionally augment with kNN."""

import logging
import yaml
import torch
from pathlib import Path
from datasets import load_dataset

from graft.data.build_graph import build_hotpot_graph
from graft.data.augment_graph import augment_with_knn
from graft.data.build_dev_set import build_fixed_dev_set
from graft.data.pair_maker import load_query_pairs

logger = logging.getLogger(__name__)


def prepare_hotpot_data(config, split="train"):
    """Download mteb/hotpotqa, filter corpus, build base graph using train split only."""
    output_dir = Path(config["data"]["graph_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_name = config["data"]["graph_name"]
    base_graph_path = output_dir / f"{graph_name}.pt"

    logger.info("Loading mteb/hotpotqa corpus...")
    corpus = load_dataset("mteb/hotpotqa", "corpus", split="corpus")

    # Load train qrels for building graph edges (co-occurrence)
    qrels_train = load_dataset("mteb/hotpotqa", "default", split="train")

    logger.info(f"Using full corpus: {len(corpus)} documents")

    chunk_size = config["data"]["chunk_size"]
    chunk_overlap = config["data"]["chunk_overlap"]

    logger.info(
        f"Building base graph with chunk_size={chunk_size}, overlap={chunk_overlap}..."
    )
    build_hotpot_graph(
        corpus, qrels_train, str(base_graph_path), chunk_size, chunk_overlap
    )
    logger.info(f"Base graph created: {base_graph_path}")

    # Step 2: Augment with kNN if configured
    semantic_k = config["data"].get("semantic_k")
    if semantic_k is not None:
        knn_only = config["data"].get("knn_only", False)
        suffix = f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
        augmented_path = output_dir / f"{graph_name}{suffix}.pt"

        logger.info(
            f"Augmenting graph with Semantic kNN (k={semantic_k}, knn_only={knn_only})..."
        )
        augment_with_knn(
            str(base_graph_path),
            str(augmented_path),
            config,
            semantic_k,
            knn_only,
            force_recompute=True,
        )
        logger.info(f"Augmented graph created: {augmented_path}")
        final_graph_path = augmented_path
    else:
        final_graph_path = base_graph_path

    # Step 3: Build fixed dev set (DO NOT touch test set during training!)
    logger.info("Building fixed dev set...")
    graph = torch.load(final_graph_path, weights_only=False)

    dev_pairs = load_query_pairs("dev", str(final_graph_path), config, log=True)
    dev_set_path = output_dir / f"{final_graph_path.stem}_dev_set.pt"

    build_fixed_dev_set(graph, dev_pairs, config, str(dev_set_path))

    logger.info(f"Data preparation complete! Final graph: {final_graph_path}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m graft.data.prepare CONFIG_PATH [SPLIT]")
        print("Example: python -m graft.data.prepare configs/hotpot_e5_sage.yml train")
        sys.exit(1)

    config_path = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "train"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    prepare_hotpot_data(config, split)
