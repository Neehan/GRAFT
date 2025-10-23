"""Full eval pipeline: embed corpus -> build FAISS -> evaluate retriever."""

import logging
import yaml
import torch
from pathlib import Path
from datasets import load_dataset

from graft.eval.embed_corpus import embed_corpus
from graft.eval.build_faiss import build_faiss_index
from graft.eval.eval_retriever import evaluate_retriever

logger = logging.getLogger(__name__)


def run_full_eval(encoder_path, config_path, output_dir, split="validation"):
    """Run complete eval pipeline: embed, index, evaluate."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    index_path = output_dir / "index.faiss"
    results_path = output_dir / "results.json"

    logger.info("Step 1/3: Embedding corpus...")
    embed_corpus(encoder_path, config, str(embeddings_path))

    logger.info("Step 2/3: Building FAISS index...")
    build_faiss_index(str(embeddings_path), config, str(index_path))

    logger.info(f"Loading HotpotQA {split} split and graph...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    graph = torch.load(config["data"]["graph_path"])

    logger.info("Step 3/3: Evaluating retriever...")
    evaluate_retriever(encoder_path, str(index_path), config, str(results_path), dataset, graph)

    logger.info(f"Evaluation complete! Results: {results_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    encoder_path = sys.argv[1]
    config_path = sys.argv[2]
    output_dir = sys.argv[3]
    split = sys.argv[4] if len(sys.argv) > 4 else "validation"

    run_full_eval(encoder_path, config_path, output_dir, split)
