"""GraphSAGE neighbor sampler for batched query-doc pairs with k-hop subgraphs."""

import torch
from torch_geometric.loader import NeighborSampler


class GraphBatchSampler:
    def __init__(self, graph, train_pairs, batch_size_queries, fanouts):
        self.graph = graph
        self.train_pairs = train_pairs
        self.batch_size = batch_size_queries
        self.fanouts = fanouts
        self.neighbor_sampler = NeighborSampler(
            graph.edge_index,
            sizes=fanouts,
            batch_size=batch_size_queries,
            shuffle=True
        )

    def __len__(self):
        return len(self.train_pairs) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.train_pairs), self.batch_size):
            batch_pairs = self.train_pairs[i:i + self.batch_size]

            queries = [pair["query"] for pair in batch_pairs]
            pos_nodes = torch.tensor([pair["pos_node"] for pair in batch_pairs])

            batch_size, n_id, adjs = next(iter(self.neighbor_sampler([pos_nodes])))

            subgraph = type('Subgraph', (), {
                'edge_index': adjs[0].edge_index,
                'n_id': n_id
            })()

            yield {
                "queries": queries,
                "pos_nodes": pos_nodes,
                "neg_nodes": torch.tensor([]),
                "subgraph": subgraph
            }
