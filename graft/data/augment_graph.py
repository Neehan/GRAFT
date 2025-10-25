"""Augment graph with kNN edges based on embedding similarity."""

import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from baselines.retrievers import ZeroShotRetriever

logger = logging.getLogger(__name__)


def embed_graph_nodes(graph, config, device=None):
    """Embed all graph nodes using pretrained encoder.

    Args:
        graph: PyG graph with node_text attribute
        config: Config dict with encoder settings
        device: torch device

    Returns:
        numpy array of embeddings (num_nodes, hidden_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config["encoder"]["model_name"]
    logger.info(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name, device=str(device))
    encoder.max_seq_length = config["encoder"]["max_len"]
    encoder.eval()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs with DataParallel for encoder")
        encoder = torch.nn.DataParallel(encoder)

    node_texts = graph.node_text
    batch_size = config["encoder"]["eval_batch_size"]

    logger.info(f"Embedding {len(node_texts)} nodes with batch_size={batch_size}...")
    base_encoder = encoder.module if num_gpus > 1 else encoder
    embeddings = base_encoder.encode(
        node_texts,
        convert_to_tensor=False,
        normalize_embeddings=config["encoder"]["normalize_embeddings"],
        batch_size=batch_size,
        show_progress_bar=True,
        device=str(device)
    )

    return embeddings


def build_knn_edges(embeddings, k, config):
    """Build undirected kNN edges using FAISS.

    Args:
        embeddings: numpy array (num_nodes, hidden_dim)
        k: Number of nearest neighbors
        config: Config dict with batch_size

    Returns:
        List of (src, dst) tuples
    """
    num_nodes, _ = embeddings.shape

    logger.info("Normalizing embeddings for FAISS")
    embeddings = ZeroShotRetriever.normalize_embeddings(embeddings)

    logger.info(f"Building Flat index: {num_nodes} nodes (exact search)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = ZeroShotRetriever.build_index(embeddings, config["index"], device=device)

    logger.info("Flat index built")

    batch_size = int(config["index"]["gpu_search_batch_size"])
    if batch_size <= 0:
        raise ValueError("index.gpu_search_batch_size must be a positive integer")

    logger.info(f"Starting kNN search: k={k}, batch_size={batch_size}...")
    all_indices = []

    for i in tqdm(range(0, num_nodes, batch_size), desc="kNN search"):
        batch_end = min(i + batch_size, num_nodes)
        batch_embeddings = embeddings[i:batch_end]
        _, batch_indices = index.search(batch_embeddings, k + 1)
        all_indices.append(batch_indices)

    indices = np.vstack(all_indices)

    logger.info("Building edge list from kNN results...")
    src = np.repeat(np.arange(num_nodes), k)
    dst = indices[:, 1:].flatten()

    edges_fwd = np.stack([src, dst], axis=1)
    edges_bwd = np.stack([dst, src], axis=1)
    edges = np.unique(np.vstack([edges_fwd, edges_bwd]), axis=0)

    return [(int(s), int(d)) for s, d in edges]


def augment_with_knn(
    input_graph_path,
    output_graph_path,
    config,
    k,
    knn_only=False,
    force_recompute=False,
):
    """Augment graph with kNN edges.

    Args:
        input_graph_path: Path to input graph .pt file
        output_graph_path: Path to save augmented graph
        config: Config dict with encoder settings
        k: Number of nearest neighbors
        knn_only: If True, use only kNN edges (ablation). If False, add to existing edges (hybrid)
        force_recompute: If True, ignore cached embeddings

    Returns:
        Path to output graph
    """
    logger.info(f"Loading graph from {input_graph_path}")
    graph = torch.load(input_graph_path, weights_only=False)

    model_name = config["encoder"]["model_name"]
    embeddings_cache = str(input_graph_path).replace(
        ".pt", f"_embeddings_{model_name.replace('/', '_')}.npy"
    )

    if Path(embeddings_cache).exists() and not force_recompute:
        logger.info(f"Loading cached embeddings from {embeddings_cache}")
        embeddings = np.load(embeddings_cache)
    else:
        embeddings = embed_graph_nodes(graph, config)
        logger.info(f"Caching embeddings to {embeddings_cache}")
        np.save(embeddings_cache, embeddings)

    old_num_edges = graph.edge_index.size(1)
    if old_num_edges > 0:
        old_degrees = torch.bincount(graph.edge_index[0], minlength=len(graph.node_text))
    else:
        old_degrees = torch.zeros(len(graph.node_text), dtype=torch.long)
    old_mean_degree = old_degrees.float().mean().item()

    knn_edges = build_knn_edges(embeddings, k, config)

    if knn_only:
        logger.info(f"Mode: kNN-only (ablation) - using only semantic edges")
        new_edge_index = torch.tensor(knn_edges, dtype=torch.long).T
    else:
        logger.info(
            f"Mode: Hybrid - keeping structural edges + adding semantic kNN edges"
        )
        existing_edges = [tuple(e) for e in graph.edge_index.T.tolist()]
        combined_edges = list(set(existing_edges + knn_edges))
        new_edge_index = torch.tensor(combined_edges, dtype=torch.long).T

    graph.edge_index = new_edge_index

    new_num_edges = new_edge_index.size(1)
    if new_num_edges > 0:
        new_degrees = torch.bincount(new_edge_index[0], minlength=len(graph.node_text))
    else:
        new_degrees = torch.zeros(len(graph.node_text), dtype=torch.long)
    new_mean_degree = new_degrees.float().mean().item()

    logger.info(f"Graph augmentation complete:")
    logger.info(f"  Before: {old_num_edges} edges, mean degree={old_mean_degree:.2f}")
    logger.info(f"  After:  {new_num_edges} edges, mean degree={new_mean_degree:.2f}")
    if not knn_only:
        logger.info(f"  Added: {new_num_edges - old_num_edges} kNN edges")

    Path(output_graph_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_graph_path)
    logger.info(f"Augmented graph saved to {output_graph_path}")

    return output_graph_path


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Augment graph with kNN edges based on semantic similarity"
    )
    parser.add_argument("input_graph", type=str, help="Path to input graph .pt file")
    parser.add_argument("output_graph", type=str, help="Path to save augmented graph")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of nearest neighbors (default: 5)"
    )
    parser.add_argument(
        "--knn-only",
        action="store_true",
        help="Use only kNN edges (ablation), discard structural edges",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recompute embeddings even if cached",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    augment_with_knn(
        args.input_graph,
        args.output_graph,
        config,
        args.k,
        args.knn_only,
        args.force_recompute,
    )
