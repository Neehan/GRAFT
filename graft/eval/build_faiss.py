"""Build FAISS IVF-HNSW index for fast ANN retrieval at inference."""

import logging
import faiss
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings_path, config, output_path):
    embeddings = np.load(embeddings_path)
    d = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index_type = config["index"]["faiss_type"]

    if index_type == "IVF,HNSW":
        nlist = config["index"]["nlist"]
        quantizer = faiss.IndexHNSWFlat(d, 32)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexFlatIP(d)

    index.add(embeddings)

    if hasattr(index, "nprobe"):
        index.nprobe = config["index"]["nprobe"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, output_path)
    logger.info(f"FAISS index saved: {output_path}")
