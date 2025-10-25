"""GraphSAGE neighbor sampler for batched query-doc pairs with k-hop subgraphs."""

import logging
import torch
from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

logger = logging.getLogger(__name__)


class GraphBatchSampler:
    def __init__(
        self, graph, train_pairs, query_batch_size, fanouts, neg_seed_ratio, target_subgraph_size, rank=0, world_size=1
    ):
        self.graph = graph
        self.train_pairs = train_pairs
        self.batch_size = query_batch_size
        self.num_hops = len(fanouts)
        self.fanouts = fanouts
        self.neg_seed_ratio = neg_seed_ratio
        self.target_subgraph_size = target_subgraph_size
        self.rank = rank
        self.world_size = world_size

        # Create persistent sampler (CPU-based, no workers)
        self.data = Data(
            edge_index=graph.edge_index,
            num_nodes=len(graph.node_text),
        )
        self.sampler = NeighborSampler(
            data=self.data,
            num_neighbors=fanouts,
            replace=False,
            disjoint=False,
        )

    def __len__(self):
        total_batches = len(self.train_pairs) // self.batch_size
        return total_batches // self.world_size

    def _sample_negative_seed(self, used_nodes):
        """Pick a random node outside the positive set to enforce contrastive support."""
        num_nodes = len(self.graph.node_text)
        if len(used_nodes) >= num_nodes:
            return None

        used = set(used_nodes)

        # Try a few random draws before falling back to a scan.
        for _ in range(8):
            candidate = torch.randint(0, num_nodes, (1,)).item()
            if candidate not in used:
                return candidate

        for candidate in range(num_nodes):
            if candidate not in used:
                return candidate

        return None

    def _sample_neighbors(self, seed_nodes):
        """Fast neighbor sampling using persistent sampler.

        Returns:
            subset: All sampled node IDs
            edge_index: Edges in the subgraph (relabeled)
        """
        seed_tensor = torch.tensor(seed_nodes, dtype=torch.long).unique()

        # Use persistent sampler (CPU-only, no loader overhead)
        sampler_input = NodeSamplerInput(
            input_id=None,
            node=seed_tensor,
        )
        result = self.sampler.sample_from_nodes(sampler_input)

        edge_index = torch.stack([result.row, result.col], dim=0)  # type: ignore

        return result.node, edge_index

    def _sample_negative_edges(self, edge_index, num_nodes, num_neg_samples):
        """Sample negative edges (non-existing edges) from subgraph."""
        if num_neg_samples <= 0:
            return torch.empty((2, 0), dtype=torch.long)

        edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        neg_edges = []
        while len(neg_edges) < num_neg_samples:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            if src != dst and (src, dst) not in edge_set:
                neg_edges.append((src, dst))

        if not neg_edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(neg_edges, dtype=torch.long).t()

    def _stabilize_subgraph_size(self, subset, edge_index, all_pos_nodes):
        """Trim or pad subgraph to reach target size for stable denominator."""
        num_nodes = len(subset)

        if num_nodes > self.target_subgraph_size:
            # Trim: keep all positive nodes, sample from others
            all_pos_set = set(all_pos_nodes)
            pos_indices = [i for i, node_id in enumerate(subset.tolist()) if node_id in all_pos_set]
            non_pos_indices = [i for i, node_id in enumerate(subset.tolist()) if node_id not in all_pos_set]

            keep_non_pos = self.target_subgraph_size - len(pos_indices)
            if keep_non_pos > 0 and non_pos_indices:
                perm = torch.randperm(len(non_pos_indices))[:keep_non_pos]
                sampled_non_pos = [non_pos_indices[i] for i in perm.tolist()]
                keep_indices = pos_indices + sampled_non_pos
            else:
                keep_indices = pos_indices[:self.target_subgraph_size]

            keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long)
            subset = subset[keep_indices_tensor]

            # Relabel edges
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
            edge_mask = torch.tensor(
                [(e0.item() in node_mapping and e1.item() in node_mapping)
                 for e0, e1 in zip(edge_index[0], edge_index[1])],
                dtype=torch.bool
            )
            edge_index = edge_index[:, edge_mask]
            edge_index = torch.tensor(
                [[node_mapping[e0.item()], node_mapping[e1.item()]]
                 for e0, e1 in zip(edge_index[0], edge_index[1])],
                dtype=torch.long
            ).t()

        elif num_nodes < self.target_subgraph_size:
            # Pad: add random negatives
            subset_set = set(subset.tolist())
            num_to_add = self.target_subgraph_size - num_nodes
            added_nodes = []
            for _ in range(num_to_add):
                neg_node = self._sample_negative_seed(list(subset_set) + added_nodes)
                if neg_node is not None:
                    added_nodes.append(neg_node)

            if added_nodes:
                subset = torch.cat([subset, torch.tensor(added_nodes, dtype=torch.long)])

        return subset, edge_index

    def __iter__(self):
        for batch_idx in range(0, len(self.train_pairs) // self.batch_size):
            if batch_idx % self.world_size != self.rank:
                continue

            i = batch_idx * self.batch_size
            batch_pairs = self.train_pairs[i : i + self.batch_size]

            queries = [pair["query"] for pair in batch_pairs]
            pos_nodes_list = [pair["pos_nodes"] for pair in batch_pairs]

            all_pos_nodes = []
            for nodes in pos_nodes_list:
                all_pos_nodes.extend(nodes)
            all_pos_nodes = list(dict.fromkeys(all_pos_nodes))

            num_neg_seeds = int(len(all_pos_nodes) * self.neg_seed_ratio)
            neg_seeds = []
            for _ in range(num_neg_seeds):
                neg_seed = self._sample_negative_seed(all_pos_nodes + neg_seeds)
                if neg_seed is not None:
                    neg_seeds.append(neg_seed)

            seed_nodes = all_pos_nodes + neg_seeds

            # Use proper neighbor sampling with fanouts
            subset, edge_index = self._sample_neighbors(seed_nodes)

            # Stabilize subgraph size for constant denominator in InfoNCE
            subset, edge_index = self._stabilize_subgraph_size(subset, edge_index, all_pos_nodes)

            num_nodes = len(subset)
            num_pos_edges = edge_index.size(1)

            pos_edges = edge_index
            neg_edges = self._sample_negative_edges(
                edge_index, num_nodes, num_pos_edges
            )

            subgraph = type(
                "Subgraph",
                (),
                {
                    "edge_index": edge_index,
                    "n_id": subset,
                    "n_id_cpu": subset.cpu(),  # type: ignore
                },
            )()

            yield {
                "queries": queries,
                "pos_nodes": pos_nodes_list,
                "subgraph": subgraph,
                "pos_edges": pos_edges,
                "neg_edges": neg_edges,
            }
