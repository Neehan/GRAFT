"""Build FAISS IVF-HNSW index for fast ANN retrieval at inference."""

import logging
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_faiss_index_from_embeddings(embeddings, config, output_path):
    """Build FAISS index from embeddings array."""
    if embeddings.dtype == np.float16:
        logger.info("Converting float16 embeddings to float32 for FAISS")
        embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]
    n = embeddings.shape[0]

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    omp_threads = config["index"]["omp_threads"]
    faiss.omp_set_num_threads(omp_threads)
    logger.info(f"FAISS using {omp_threads} OpenMP threads")

    index_type = config["index"]["type"]

    if index_type == "ivf":
        nlist = config["index"]["ivf_nlist"]
        nprobe = config["index"]["ivf_nprobe"]
        m = config["index"]["hnsw_m"]
        train_sample_size = config["index"]["ivf_train_sample_size"]
        add_batch_size = config["index"]["add_batch_size"]

        logger.info(
            f"Building IVF index with HNSW quantizer: nlist={nlist}, nprobe={nprobe}, M={m}"
        )
        quantizer = faiss.IndexHNSWFlat(d, m)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        if train_sample_size is not None and train_sample_size < n:
            logger.info(
                f"Training IVF on {train_sample_size:,} sampled vectors (out of {n:,})"
            )
            train_indices = np.random.choice(n, train_sample_size, replace=False)
            index.train(embeddings[train_indices])
        else:
            logger.info(f"Training IVF on all {n:,} vectors")
            index.train(embeddings)

        logger.info(f"Adding {n:,} vectors in batches of {add_batch_size:,}")
        for i in tqdm(range(0, n, add_batch_size), desc="Adding to IVF index"):
            batch_end = min(i + add_batch_size, n)
            index.add(embeddings[i:batch_end])

        index.nprobe = nprobe
    elif index_type == "hnsw":
        m = config["index"]["hnsw_m"]
        ef_construction = config["index"]["hnsw_ef_construction"]
        ef_search = config["index"]["hnsw_ef_search"]
        quantize = config["index"]["quantize"]
        add_batch_size = config["index"]["add_batch_size"]

        if quantize:
            logger.info(
                f"Building HNSW+SQ8 index: M={m}, ef_construction={ef_construction}, ef_search={ef_search}"
            )
            index = faiss.index_factory(d, f"HNSW{m},SQ8", faiss.METRIC_INNER_PRODUCT)
        else:
            logger.info(
                f"Building HNSW index: M={m}, ef_construction={ef_construction}, ef_search={ef_search}"
            )
            index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)

        index.hnsw.efConstruction = ef_construction

        if quantize and not index.is_trained:
            train_sample_size = config["index"]["sq8_train_sample_size"]
            if train_sample_size is not None and train_sample_size < n:
                logger.info(
                    f"Training SQ8 quantizer on {train_sample_size:,} sampled vectors (out of {n:,})"
                )
                train_indices = np.random.choice(n, train_sample_size, replace=False)
                index.train(embeddings[train_indices])
            else:
                logger.info(f"Training SQ8 quantizer on all {n:,} vectors")
                index.train(embeddings)

        logger.info(f"Adding {n:,} vectors in batches of {add_batch_size:,}")
        for i in tqdm(range(0, n, add_batch_size), desc="Adding to HNSW index"):
            batch_end = min(i + add_batch_size, n)
            index.add(embeddings[i:batch_end])

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
