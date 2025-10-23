"""Prepare HotpotQA data: build graph from HF datasets."""

import logging
import yaml
from pathlib import Path
from datasets import load_dataset

from graft.data.build_graph import build_hotpot_graph

logger = logging.getLogger(__name__)


def prepare_hotpot_data(config, split="train"):
    """Download HotpotQA and build graph with chunking."""
    output_dir = Path(config["data"]["graph_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_name = config["data"]["graph_name"]
    graph_path = output_dir / f"{graph_name}.pt"

    logger.info(f"Loading HotpotQA {split} split...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    chunk_size = config["data"]["chunk_size"]
    chunk_overlap = config["data"]["chunk_overlap"]

    logger.info(
        f"Building graph with chunk_size={chunk_size}, overlap={chunk_overlap}..."
    )
    build_hotpot_graph(dataset, str(graph_path), chunk_size, chunk_overlap)
    logger.info(f"Data preparation complete! Graph: {graph_path}")


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
