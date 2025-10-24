"""Fast hard negative mining via batched similarity search."""

import torch


class HardNegativeMiner:
    """Mine hard negatives from subgraph via similarity ranking."""

    def __init__(self, config):
        self.num_hard_negs = config["train"]["hardneg_per_query"]

    def mine_hard_negatives(self, query_embeds, subgraph_embeds, positive_indices):
        """Mine hard negatives from subgraph.

        Args:
            query_embeds: (B, D)
            subgraph_embeds: (N, D)
            positive_indices: List[List[int]] positive indices per query

        Returns:
            (B, M) hard negative indices
        """
        scores = torch.matmul(query_embeds, subgraph_embeds.T)  # (B, N)
        batch_size, num_nodes = scores.size()

        # Build positive mask: (B, N)
        pos_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, pos_idx in enumerate(positive_indices):
            pos_mask[i, pos_idx] = True

        # Mask out positives and get top-k
        neg_scores = scores.masked_fill(pos_mask, -1e10)
        _, hard_negs = torch.topk(neg_scores, self.num_hard_negs, dim=1)

        return hard_negs
