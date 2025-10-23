"""Prepare HotpotQA data: build graph from HF datasets."""

import logging
from pathlib import Path
from datasets import load_dataset

from graft.data.build_graph import build_hotpot_graph

logger = logging.getLogger(__name__)


def prepare_hotpot_data(output_dir, split="train"):
    """Download HotpotQA and build graph."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "page_graph.pt"

    logger.info(f"Loading HotpotQA {split} split...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    logger.info(f"Building graph...")
    build_hotpot_graph(dataset, str(graph_path))
    logger.info(f"Data preparation complete! Graph: {graph_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./datasets/hotpot"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"

    prepare_hotpot_data(output_dir, split)
