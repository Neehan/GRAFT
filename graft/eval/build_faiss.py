"""Build FAISS IVF-HNSW index for fast ANN retrieval at inference."""

import logging
import faiss
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings_path, config, output_path):
    embeddings = np.load(embeddings_path)
    d = embeddings.shape[1]

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index_type = config["index"]["type"]

    if index_type == "ivf":
        nlist = config["index"]["ivf_nlist"]
        nprobe = config["index"]["ivf_nprobe"]
        m = config["index"]["hnsw_m"]

        logger.info(
            f"Building IVF index with HNSW quantizer: nlist={nlist}, nprobe={nprobe}"
        )
        quantizer = faiss.IndexHNSWFlat(d, m)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe
    elif index_type == "hnsw":
        m = config["index"]["hnsw_m"]
        ef_construction = config["index"]["hnsw_ef_construction"]
        ef_search = config["index"]["hnsw_ef_search"]
        logger.info(
            f"Building HNSW index: M={m}, ef_construction={ef_construction}, ef_search={ef_search}"
        )
        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(embeddings)
        index.hnsw.efSearch = ef_search
    elif index_type == "flat":
        logger.info("Building Flat index")
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, output_path)
    logger.info(f"FAISS index saved: {output_path}")


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
