"""InfoNCE, neighbor-contrast, and link-prediction losses for graph-conditioned retrieval."""

import torch
import torch.nn.functional as F


def info_nce_loss(query_embeds, doc_embeds, labels, tau):
    """
    Soft-OR InfoNCE: sum exp(sim) over all positives in numerator.

    query_embeds: (B, D)
    doc_embeds: (N, D) where N includes all nodes in subgraph
    labels: (B, K) where K is number of positives per query, or (B,) for single positive
    tau: temperature

    For multiple positives, uses soft-OR: any positive being similar makes numerator large.
    As tau->0, behaves like max over positives (not average).
    """
    scores = torch.matmul(query_embeds, doc_embeds.T) / tau

    if labels.dim() == 1:
        # Single positive per query: standard InfoNCE
        return F.cross_entropy(scores, labels)

    # Soft-OR InfoNCE for multiple positives (vectorized, numerically stable)
    # Use log-sum-exp trick to avoid overflow/underflow
    # log(sum(exp(x))) = logsumexp(x)

    # Create mask for positives: (B, N)
    batch_size, num_candidates = scores.size()
    pos_mask = torch.zeros_like(scores, dtype=torch.bool)

    # Set mask[i, labels[i, :]] = True for each query
    batch_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1)
    pos_mask[batch_indices, labels] = True

    # Numerator: log(sum(exp(scores))) for positives only
    # Set non-positive scores to -inf so they don't contribute
    pos_scores = torch.where(
        pos_mask, scores, torch.tensor(-1e10, device=scores.device)
    )
    log_numerator = torch.logsumexp(pos_scores, dim=1)  # (B,)

    # Denominator: log(sum(exp(scores))) for all candidates
    log_denominator = torch.logsumexp(scores, dim=1)  # (B,)

    # -log(numerator / denominator) = -(log_numerator - log_denominator)
    loss = -(log_numerator - log_denominator).mean()

    return loss


def neighbor_contrast_loss(node_embeds, edge_index, tau_graph):
    """
    node_embeds: (N, D)
    edge_index: (2, E) COO format
    tau_graph: temperature
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
    """
    node_embeds: (N, D)
    pos_edges: (2, E_pos)
    neg_edges: (2, E_neg)
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
):
    """
    Combine all losses
    """
    loss_q2d = info_nce_loss(query_embeds, doc_embeds, labels, tau)

    loss_nbr = neighbor_contrast_loss(node_embeds, edge_index, tau_graph)

    loss = lambda_q2d * loss_q2d + (1 - lambda_q2d) * loss_nbr

    if alpha_link > 0 and pos_edges is not None and neg_edges is not None:
        loss_link = link_prediction_loss(node_embeds, pos_edges, neg_edges)
        loss = loss + alpha_link * loss_link

    return loss, loss_q2d, loss_nbr
