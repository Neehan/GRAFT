"""GraphSAGE neighbor sampler for batched query-doc pairs with k-hop subgraphs."""

import torch
from torch_geometric.utils import k_hop_subgraph


class GraphBatchSampler:
    def __init__(
        self, graph, train_pairs, query_batch_size, fanouts, rank=0, world_size=1
    ):
        self.graph = graph
        self.train_pairs = train_pairs
        self.batch_size = query_batch_size
        self.num_hops = len(fanouts)
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        total_batches = len(self.train_pairs) // self.batch_size
        return total_batches // self.world_size

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
            pos_nodes = torch.tensor([pair["pos_node"] for pair in batch_pairs])

            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                pos_nodes.clone(),
                self.num_hops,
                self.graph.edge_index,
                relabel_nodes=True,
            )

            num_nodes = len(subset)
            num_pos_edges = edge_index.size(1)

            pos_edges = edge_index.clone() if num_pos_edges > 0 else None
            neg_edges = (
                self._sample_negative_edges(edge_index, num_nodes, num_pos_edges)
                if num_pos_edges > 0
                else None
            )

            subgraph = type(
                "Subgraph",
                (),
                {
                    "edge_index": edge_index.clone(),
                    "n_id": subset.clone(),
                    "n_id_cpu": subset.cpu(),
                },
            )()

            yield {
                "queries": queries,
                "pos_nodes": pos_nodes,
                "subgraph": subgraph,
                "pos_edges": pos_edges,
                "neg_edges": neg_edges,
            }
