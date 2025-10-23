"""Augment graph with kNN edges based on embedding similarity."""

import logging
import torch
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

from graft.models.encoder import Encoder

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
    encoder = Encoder(
        model_name=model_name,
        max_len=config["encoder"]["max_len"],
        pool=config["encoder"]["pool"],
        freeze_layers=0,
    )

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        encoder = torch.nn.DataParallel(encoder)

    encoder.to(device)
    encoder.eval()

    node_texts = graph.node_text
    embeddings = []

    # Scale batch size by number of GPUs
    base_batch_size = config["data"]["batch_size"]
    batch_size = base_batch_size * max(1, torch.cuda.device_count())
    logger.info(
        f"Batch size: {batch_size} (base={base_batch_size}, GPUs={torch.cuda.device_count()})"
    )

    # Enable mixed precision for faster inference
    use_amp = config["train"]["bf16"]
    if use_amp:
        logger.info("Using mixed precision (bfloat16)")

    logger.info(f"Embedding {len(node_texts)} nodes...")

    num_nodes = len(node_texts)
    with torch.no_grad():
        for i in tqdm(range(0, num_nodes, batch_size), desc="Embedding"):
            batch = node_texts[i : i + batch_size]

            # Use autocast for mixed precision
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_embeds = _encode_batch(encoder, batch, config, device)
            else:
                batch_embeds = _encode_batch(encoder, batch, config, device)

            # Move to CPU immediately to avoid GPU memory accumulation
            embeddings.append(batch_embeds.cpu())

    return torch.cat(embeddings, dim=0).numpy()


def _encode_batch(encoder, texts, config, device):
    """Helper to encode a batch of texts.

    Handles both DataParallel and single-GPU cases.
    """
    # Access tokenizer from encoder or encoder.module
    if hasattr(encoder, "module"):
        tokenizer = encoder.module.tokenizer
        max_len = encoder.module.max_len
        pool = encoder.module.pool
    else:
        tokenizer = encoder.tokenizer
        max_len = encoder.max_len
        pool = encoder.pool

    encoded = tokenizer(
        texts,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Forward pass
    if hasattr(encoder, "module"):
        outputs = encoder.module.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    else:
        outputs = encoder.model(input_ids=input_ids, attention_mask=attention_mask)

    # Pooling
    if pool == "cls":
        embeddings = outputs.last_hidden_state[:, 0]
    elif pool == "mean":
        embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1, keepdim=True)
    else:
        raise ValueError(f"Unknown pooling method: {pool}")

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
    num_nodes, dim = embeddings.shape

    logger.info(f"Converting embeddings to float32...")
    embeddings = embeddings.astype(np.float32)
    embeddings = np.ascontiguousarray(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    logger.info(f"Building FAISS index for {num_nodes} nodes...")

    index = faiss.IndexFlatIP(dim)
    logger.info(f"Adding {num_nodes} vectors to index...")
    index.add(embeddings)
    logger.info("Index built on CPU")

    if torch.cuda.is_available():
        logger.info("Moving index to GPU...")
        ngpus = torch.cuda.device_count()
        if ngpus > 1:
            index = faiss.index_cpu_to_all_gpus(index)
            logger.info(f"FAISS index replicated across {ngpus} GPUs")
        else:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("FAISS index moved to GPU 0")
    else:
        logger.info("CUDA not available, using CPU for FAISS (will be slow)")

    batch_size = config["data"]["batch_size"]
    logger.info(f"Starting kNN search: k={k}, batch_size={batch_size}...")
    all_indices = []

    for i in tqdm(range(0, num_nodes, batch_size), desc="kNN search"):
        batch_end = min(i + batch_size, num_nodes)
        batch_embeddings = embeddings[i:batch_end]
        _, batch_indices = index.search(batch_embeddings, k + 1)
        all_indices.append(batch_indices)

    indices = np.vstack(all_indices)

    logger.info("Building edge list from kNN results...")
    edges = []
    for src_node in tqdm(range(num_nodes), desc="Building edges"):
        neighbors = indices[src_node][1:]
        for dst_node in neighbors:
            edges.append((src_node, int(dst_node)))
            edges.append((int(dst_node), src_node))

    return list(set(edges))


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
    old_degrees = torch.zeros(len(graph.node_text), dtype=torch.long)
    if old_num_edges > 0:
        for node_id in graph.edge_index[0]:
            old_degrees[node_id] += 1
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
    new_degrees = torch.zeros(len(graph.node_text), dtype=torch.long)
    if new_num_edges > 0:
        for node_id in new_edge_index[0]:
            new_degrees[node_id] += 1
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
