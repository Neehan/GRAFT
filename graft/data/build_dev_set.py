"""Build fixed dev set with hard negatives from graph."""

import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


def build_fixed_dev_set(graph, dev_pairs, config, output_path):
    """
    Build fixed dev set with hard negatives from graph neighbors.

    Args:
        graph: Graph object with edge_index and node_text
        dev_pairs: List of dev query pairs from pair_maker
        config: Config dict with eval settings
        output_path: Path to save the eval set

    Returns:
        eval_set: List of dicts with query, candidates, num_positives
    """
    num_nodes = len(graph.node_text)
    num_samples = min(len(dev_pairs), config["eval"]["num_samples"])
    base_negatives = config["eval"]["num_negatives"]

    # Find max positives to determine total candidates
    max_positives = max(len(pair["pos_nodes"]) for pair in dev_pairs[:num_samples])
    total_candidates = max_positives + base_negatives

    edge_index = graph.edge_index

    eval_set = []
    for i in range(num_samples):
        pair = dev_pairs[i]
        pos_nodes = pair["pos_nodes"]
        pos_set = set(pos_nodes)

        # Sample hard negatives: neighbors of positive nodes
        hard_negs = set()
        for pos_node in pos_nodes:
            # Get neighbors of this positive node
            neighbors = edge_index[1][edge_index[0] == pos_node].tolist()
            hard_negs.update(neighbors)

        # Remove positives from hard negatives
        hard_negs = list(hard_negs - pos_set)

        # Determine how many negatives we need
        num_negatives_needed = total_candidates - len(pos_nodes)

        # Use hard negatives first, then fill with random
        if len(hard_negs) >= num_negatives_needed:
            # More than enough hard negatives, sample subset
            hard_neg_indices = torch.randperm(len(hard_negs))[
                :num_negatives_needed
            ].tolist()
            neg_nodes = [hard_negs[idx] for idx in hard_neg_indices]
        else:
            # Not enough hard negatives, use all and fill with random
            neg_nodes = hard_negs.copy()
            num_random_needed = num_negatives_needed - len(hard_negs)

            # Sample random negatives (excluding positives and hard negatives)
            excluded = pos_set | set(hard_negs)
            random_negs = []
            while len(random_negs) < num_random_needed:
                rand_idx = torch.randint(0, num_nodes, (1,)).item()
                if rand_idx not in excluded:
                    random_negs.append(rand_idx)
                    excluded.add(rand_idx)

            neg_nodes.extend(random_negs)

        # Combine positives and negatives
        candidate_indices = pos_nodes + neg_nodes

        eval_set.append(
            {
                "query": pair["query"],
                "qid": pair["qid"],
                "candidate_indices": candidate_indices,
                "num_positives": len(pos_nodes),
            }
        )

    # Save to disk
    output_path = Path(output_path)
    torch.save(eval_set, output_path)

    logger.info(
        f"Built fixed dev set: {len(eval_set)} queries, {total_candidates} total candidates each"
    )
    logger.info(f"Saved dev set to: {output_path}")

    return eval_set
