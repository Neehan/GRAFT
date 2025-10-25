"""Fast hard negative mining via batched similarity search."""

import logging
import torch

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """Mine hard negatives from subgraph via similarity ranking.

    Uses proportional mining: number of hard negatives scales with subgraph size.
    """

    def __init__(self, config):
        self.hardneg_ratio = config["graph"]["hardneg_ratio"]
        self.hardneg_min = config["graph"]["hardneg_min"]
        self.hardneg_max = config["graph"]["hardneg_max"]

    def mine_hard_negatives(self, query_embeds, subgraph_embeds, positive_indices):
        """Mine hard negatives from subgraph with proportional sampling.

        Number of hard negatives per query = min(max(ratio * num_available, min), max)

        Args:
            query_embeds: (B, D)
            subgraph_embeds: (N, D)
            positive_indices: List[List[int]] positive indices per query

        Returns:
            (B, M) hard negative indices (M can vary per batch based on subgraph size)
        """
        scores = torch.matmul(query_embeds, subgraph_embeds.T)  # (B, N)
        batch_size, num_nodes = scores.size()

        # Build positive mask: (B, N)
        pos_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, pos_idx in enumerate(positive_indices):
            pos_mask[i, pos_idx] = True

        # Compute proportional num_hard_negs based on first query's negatives
        # (assumes similar pos/neg ratio across queries in batch)
        neg_candidates_sample = (~pos_mask[0]).sum().item()
        num_hard_negs = int(neg_candidates_sample * self.hardneg_ratio)
        num_hard_negs = max(self.hardneg_min, min(num_hard_negs, self.hardneg_max))

        hard_negs = []
        all_nodes = torch.arange(num_nodes, device=query_embeds.device)
        for i in range(batch_size):
            neg_candidates = all_nodes[~pos_mask[i]]
            if neg_candidates.numel() == 0:
                logger.warning("HardNegativeMiner: no negatives available for query %d", i)
                hard_negs.append(
                    torch.full(
                        (num_hard_negs,),
                        fill_value=0,
                        dtype=torch.long,
                        device=query_embeds.device,
                    )
                )
                continue

            k = min(num_hard_negs, neg_candidates.numel())
            candidate_scores = scores[i, neg_candidates]
            _, top_idx = torch.topk(candidate_scores, k=k, dim=0)
            selected = neg_candidates[top_idx]

            if k < num_hard_negs:
                # Fill remainder with random negatives
                remaining = num_hard_negs - k
                perm = torch.randperm(neg_candidates.numel(), device=query_embeds.device)[:remaining]
                selected = torch.cat([selected, neg_candidates[perm]], dim=0)

            hard_negs.append(selected[:num_hard_negs])

        return torch.stack(hard_negs, dim=0)
