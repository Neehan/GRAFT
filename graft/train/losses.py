"""InfoNCE, neighbor-contrast, and link-prediction losses for graph-conditioned retrieval."""

import torch
import torch.nn.functional as F


def info_nce_loss(query_embeds, doc_embeds, labels, tau, hard_neg_embeds=None):
    """
    query_embeds: (B, D)
    doc_embeds: (N, D) where N = B * (1 + num_negatives)
    labels: (B,) indices of positives in doc_embeds
    tau: temperature
    hard_neg_embeds: (B, K, D) optional hard negatives for each query
    """
    scores = torch.matmul(query_embeds, doc_embeds.T) / tau

    if hard_neg_embeds is not None:
        B, K, D = hard_neg_embeds.shape
        hard_scores = torch.matmul(query_embeds.unsqueeze(1), hard_neg_embeds.transpose(1, 2)).squeeze(1) / tau
        scores = torch.cat([scores, hard_scores], dim=1)

    return F.cross_entropy(scores, labels)


def neighbor_contrast_loss(node_embeds, edge_index, tau_graph):
    """
    node_embeds: (N, D)
    edge_index: (2, E) COO format
    tau_graph: temperature
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=node_embeds.device)

    src, dst = edge_index
    num_edges = edge_index.size(1)

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
    pos_src, pos_dst = pos_edges
    neg_src, neg_dst = neg_edges

    pos_scores = (node_embeds[pos_src] * node_embeds[pos_dst]).sum(dim=1)
    neg_scores = (node_embeds[neg_src] * node_embeds[neg_dst]).sum(dim=1)

    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

    return (pos_loss + neg_loss) / 2


def compute_total_loss(query_embeds, doc_embeds, labels, node_embeds, edge_index, pos_edges, neg_edges, lambda_q2d, tau, tau_graph, alpha_link, hard_neg_embeds=None):
    """
    Combine all losses
    """
    loss_q2d = info_nce_loss(query_embeds, doc_embeds, labels, tau, hard_neg_embeds)

    loss_nbr = neighbor_contrast_loss(node_embeds, edge_index, tau_graph)

    loss = lambda_q2d * loss_q2d + (1 - lambda_q2d) * loss_nbr

    if alpha_link > 0:
        loss_link = link_prediction_loss(node_embeds, pos_edges, neg_edges)
        loss = loss + alpha_link * loss_link

    return loss, loss_q2d, loss_nbr
