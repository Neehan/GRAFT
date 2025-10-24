"""GraphSAGE neighbor sampler for batched query-doc pairs with k-hop subgraphs."""

import logging
import torch
from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

logger = logging.getLogger(__name__)


class GraphBatchSampler:
    def __init__(
        self, graph, train_pairs, query_batch_size, fanouts, rank=0, world_size=1
    ):
        self.graph = graph
        self.train_pairs = train_pairs
        self.batch_size = query_batch_size
        self.num_hops = len(fanouts)
        self.fanouts = fanouts
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

        edge_index = torch.stack([result.row, result.col], dim=0)

        return result.node, edge_index

    def _sample_negative_edges(self, edge_index, num_nodes, num_neg_samples):
        """Sample negative edges (non-existing edges) from subgraph."""
        edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        neg_edges = []
        while len(neg_edges) < num_neg_samples:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            if src != dst and (src, dst) not in edge_set:
                neg_edges.append((src, dst))

        return torch.tensor(neg_edges, dtype=torch.long).t()

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

            extra_seed = self._sample_negative_seed(all_pos_nodes)
            seed_nodes = all_pos_nodes + ([extra_seed] if extra_seed is not None else [])

            # Use proper neighbor sampling with fanouts
            subset, edge_index = self._sample_neighbors(seed_nodes)

            num_nodes = len(subset)
            num_pos_edges = edge_index.size(1)

            pos_edges = edge_index if num_pos_edges > 0 else None
            neg_edges = (
                self._sample_negative_edges(edge_index, num_nodes, num_pos_edges)
                if num_pos_edges > 0
                else None
            )

            subgraph = type(
                "Subgraph",
                (),
                {
                    "edge_index": edge_index,
                    "n_id": subset,
                    "n_id_cpu": subset.cpu(),
                },
            )()

            yield {
                "queries": queries,
                "pos_nodes": pos_nodes_list,
                "subgraph": subgraph,
                "pos_edges": pos_edges,
                "neg_edges": neg_edges,
            }
