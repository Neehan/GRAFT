"""Build FAISS flat (exact) index on GPU for retrieval."""

import logging
import faiss
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def build_faiss_index_from_embeddings(embeddings, config, output_path):
    """Build FAISS index from embeddings array."""
    if embeddings.dtype == np.float16:
        logger.info("Converting float16 embeddings to float32 for FAISS")
        embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]
    n = embeddings.shape[0]

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    use_fp16 = config["index"]["use_fp16"]

    if not hasattr(faiss, "get_num_gpus") or not hasattr(faiss, "index_cpu_to_all_gpus"):
        raise RuntimeError(
            "FAISS GPU support is required for IndexFlatIP but is not available in this build"
        )

    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        raise RuntimeError("IndexFlatIP requires CUDA GPUs but none are visible to FAISS")

    base_index = faiss.IndexFlatIP(d)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_fp16
    co.shard = True

    logger.info(f"Moving flat index to all {num_gpus} visible GPUs (fp16={use_fp16})")
    gpu_index = faiss.index_cpu_to_all_gpus(base_index, co)

    logger.info(f"Adding {n:,} vectors on GPU")
    gpu_index.add(embeddings)
    index = faiss.index_gpu_to_cpu(gpu_index)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, output_path)
    logger.info(f"FAISS index saved: {output_path}")


def build_faiss_index(embeddings_path, config, output_path):
    """Load embeddings from file and build FAISS index (for kNN graph augmentation)."""
    embeddings = np.load(embeddings_path)
    build_faiss_index_from_embeddings(embeddings, config, output_path)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("embeddings", type=str, help="Path to embeddings .npy file")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("output", type=str, help="Path to save FAISS index")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    build_faiss_index(args.embeddings, config, args.output)
