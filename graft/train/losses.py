"""InfoNCE, neighbor-contrast, and link-prediction losses for graph-conditioned retrieval."""

import torch
import torch.nn.functional as F


def info_nce_loss(query_embeds, doc_embeds, labels, tau, labels_mask=None, hard_negs=None):
    """Soft-OR InfoNCE with optional hard negatives and label masking.

    Args:
        query_embeds: (B, D)
        doc_embeds: (N, D) all nodes in subgraph
        labels: (B, K) positive node indices (padded with zeros)
        tau: temperature
        labels_mask: (B, K) boolean mask for valid labels (handles variable positives)
        hard_negs: (B, M) hard negative node indices (mined from subgraph)

    Returns:
        Scalar loss
    """
    if hard_negs is not None:
        hard_neg_embeds = doc_embeds[hard_negs]
        scores_in_batch = torch.matmul(query_embeds, doc_embeds.T) / tau
        scores_hard = torch.matmul(query_embeds.unsqueeze(1), hard_neg_embeds.transpose(1, 2)).squeeze(1) / tau
        scores = torch.cat([scores_in_batch, scores_hard], dim=1)
    else:
        scores = torch.matmul(query_embeds, doc_embeds.T) / tau

    if labels.dim() == 1:
        return F.cross_entropy(scores, labels)

    batch_size, num_candidates = scores.size()
    pos_mask = torch.zeros_like(scores, dtype=torch.bool)

    if labels_mask is not None:
        # Use mask to filter valid labels
        for i in range(batch_size):
            valid_idx = labels[i][labels_mask[i]]
            if len(valid_idx) > 0:
                pos_mask[i, valid_idx] = True
    else:
        # All labels are valid
        batch_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1)
        pos_mask[batch_indices, labels] = True

    # Mask out non-positives with large negative value
    pos_scores = scores.masked_fill(~pos_mask, -1e10)
    log_numerator = torch.logsumexp(pos_scores, dim=1)
    log_denominator = torch.logsumexp(scores, dim=1)
    loss = -(log_numerator - log_denominator).mean()

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
    hard_negs=None,
):
    """Combine all losses: InfoNCE (q2d) + neighbor contrast + optional link prediction.

    Supports hard negatives and variable-length positives via labels_mask.
    """
    loss_q2d = info_nce_loss(query_embeds, doc_embeds, labels, tau, labels_mask, hard_negs)
    loss_nbr = neighbor_contrast_loss(node_embeds, edge_index, tau_graph)
    loss = lambda_q2d * loss_q2d + (1 - lambda_q2d) * loss_nbr

    if alpha_link > 0 and pos_edges is not None and neg_edges is not None:
        loss_link = link_prediction_loss(node_embeds, pos_edges, neg_edges)
        loss = loss + alpha_link * loss_link

    return loss, loss_q2d, loss_nbr
