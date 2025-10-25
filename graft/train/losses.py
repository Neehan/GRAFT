"""InfoNCE, neighbor-contrast, and link-prediction losses for graph-conditioned retrieval."""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def info_nce_loss(query_embeds, doc_embeds, labels, tau, labels_mask=None):
    """Soft-OR InfoNCE with in-batch negatives.

    Standard in-batch negative sampling (Karpukhin et al., 2020):
    For each query i:
      - Numerator: logsumexp over query i's positives only
      - Denominator: logsumexp over ALL subgraph nodes
      - Other queries' positives are treated as negatives
      - Hard negatives come from neg_seed_ratio in sampler (already in subgraph)

    Args:
        query_embeds: (B, D)
        doc_embeds: (N, D) all nodes in subgraph (shared across batch)
        labels: (B, K) positive node indices into doc_embeds, per query
        tau: temperature
        labels_mask: (B, K) boolean mask for valid labels (handles variable positives)

    Returns:
        Scalar loss (averaged across B queries)
    """
    batch_size = query_embeds.size(0)
    num_subgraph_nodes = doc_embeds.size(0)

    # Normalize embeddings to prevent scale issues with mixed query/subgraph nodes
    query_embeds = F.normalize(query_embeds, dim=-1)
    doc_embeds = F.normalize(doc_embeds, dim=-1)

    # Compute scores in float32 for numerical stability
    scores = torch.matmul(query_embeds.float(), doc_embeds.float().T) / tau

    # Build positive mask: mark each query's own positives (B, N)
    pos_mask = torch.zeros(
        (batch_size, num_subgraph_nodes), dtype=torch.bool, device=query_embeds.device
    )

    # Create batch indices for scatter: [0,0,0,...,1,1,1,...,B-1,B-1,...]
    batch_indices = torch.arange(batch_size, device=query_embeds.device).unsqueeze(1).expand_as(labels)

    if labels_mask is not None:
        # Only scatter valid labels
        valid_batch_idx = batch_indices[labels_mask]
        valid_labels = labels[labels_mask]
        pos_mask[valid_batch_idx, valid_labels] = True
    else:
        # Scatter all labels
        pos_mask[batch_indices.flatten(), labels.flatten()] = True

    # Compute per-query InfoNCE loss
    # Numerator: logsumexp over positives only (use dtype-safe mask value)
    mask_val = torch.finfo(scores.dtype).min
    pos_scores = scores.masked_fill(~pos_mask, mask_val)
    log_numerator = torch.logsumexp(pos_scores, dim=1)  # (B,)

    # Denominator: logsumexp over all subgraph nodes (in-batch negatives)
    log_denominator = torch.logsumexp(scores, dim=1)  # (B,)

    # Compute per-query loss and normalize by number of positives
    # This prevents batches with many positives from dominating the gradient
    per_query_loss = -(log_numerator - log_denominator)  # (B,)

    # Count number of positives per query
    num_positives = pos_mask.sum(dim=1).float()  # (B,)

    # Normalize each query's loss by its number of positives
    normalized_loss = per_query_loss / num_positives  # (B,)

    # Average across batch
    loss = normalized_loss.mean()

    return loss


def neighbor_contrast_loss(node_embeds, edge_index, tau_graph):
    """Graph-based neighbor contrastive loss for smoothness.

    Args:
        node_embeds: (N, D)
        edge_index: (2, E) COO format
        tau_graph: temperature

    Returns:
        Scalar loss
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=node_embeds.device)

    src = edge_index[0]
    dst = edge_index[1]

    pos_scores = (node_embeds[src] * node_embeds[dst]).sum(dim=1) / tau_graph

    all_scores = torch.matmul(node_embeds[src], node_embeds.T) / tau_graph

    log_denominator = torch.logsumexp(all_scores, dim=1)

    loss = -(pos_scores - log_denominator).mean()
    return loss


def link_prediction_loss(node_embeds, pos_edges, neg_edges):
    """Link prediction loss via binary classification on edges.

    Args:
        node_embeds: (N, D)
        pos_edges: (2, E_pos)
        neg_edges: (2, E_neg)

    Returns:
        Scalar loss
    """
    pos_scores = (node_embeds[pos_edges[0]] * node_embeds[pos_edges[1]]).sum(dim=1)
    neg_scores = (node_embeds[neg_edges[0]] * node_embeds[neg_edges[1]]).sum(dim=1)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )

    return (pos_loss + neg_loss) / 2


def compute_total_loss(
    query_embeds,
    doc_embeds,
    labels,
    node_embeds,
    edge_index,
    pos_edges,
    neg_edges,
    lambda_q2d,
    tau,
    tau_graph,
    alpha_link,
    labels_mask=None,
):
    """Combine all losses: InfoNCE (q2d) + neighbor contrast + optional link prediction.

    Supports variable-length positives via labels_mask.
    """
    loss_q2d = info_nce_loss(query_embeds, doc_embeds, labels, tau, labels_mask)
    loss_nbr = neighbor_contrast_loss(node_embeds, edge_index, tau_graph)
    loss = lambda_q2d * loss_q2d + (1 - lambda_q2d) * loss_nbr

    if alpha_link > 0 and pos_edges is not None and neg_edges is not None:
        loss_link = link_prediction_loss(node_embeds, pos_edges, neg_edges)
        loss = loss + alpha_link * loss_link

    return loss, loss_q2d, loss_nbr
