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
        logger.info(f"Building HNSW index: M={m}")
        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)
    elif index_type == "flat":
        logger.info("Building Flat index")
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, output_path)
    logger.info(f"FAISS index saved: {output_path}")
