"""GraphSAGE neighbor sampler for batched query-doc pairs with k-hop subgraphs."""

import torch
from torch_geometric.utils import k_hop_subgraph


class GraphBatchSampler:
    def __init__(self, graph, train_pairs, batch_size_queries, fanouts):
        self.graph = graph
        self.train_pairs = train_pairs
        self.batch_size = batch_size_queries
        self.num_hops = len(fanouts)

    def __len__(self):
        return len(self.train_pairs) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.train_pairs), self.batch_size):
            batch_pairs = self.train_pairs[i:i + self.batch_size]

            queries = [pair["query"] for pair in batch_pairs]
            pos_nodes = torch.tensor([pair["pos_node"] for pair in batch_pairs])

            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                pos_nodes,
                self.num_hops,
                self.graph.edge_index,
                relabel_nodes=True
            )

            subgraph = type('Subgraph', (), {
                'edge_index': edge_index,
                'n_id': subset
            })()

            yield {
                "queries": queries,
                "pos_nodes": pos_nodes,
                "subgraph": subgraph
            }
