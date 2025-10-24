"""GraphSAGE neighbor sampler for batched query-doc pairs with k-hop subgraphs."""

import torch
from torch_sparse import SparseTensor


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

        # Build SparseTensor adjacency once (GPU-optimized)
        self.adj_t = SparseTensor.from_edge_index(
            graph.edge_index, sparse_sizes=(len(graph.node_text), len(graph.node_text))
        ).t()

        # Move to GPU for fastest sampling
        if torch.cuda.is_available():
            self.adj_t = self.adj_t.to("cuda")

    def __len__(self):
        total_batches = len(self.train_pairs) // self.batch_size
        return total_batches // self.world_size

    def _sample_neighbors(self, seed_nodes):
        """Ultra-fast GPU neighbor sampling using torch_sparse C++ kernels.

        Returns:
            subset: All sampled node IDs
            edge_index: Edges in the subgraph (relabeled)
        """
        seed_tensor = torch.tensor(seed_nodes, dtype=torch.long).unique()

        # Move to GPU if adjacency is on GPU
        if self.adj_t.device() != torch.device("cpu"):
            seed_tensor = seed_tensor.to(self.adj_t.device())

        # Call the optimized sampler (multi-hop, fanouts per layer)
        subset, edge_index, _, _ = torch.ops.torch_sparse.neighbor_sample(
            self.adj_t.storage.rowptr(),
            self.adj_t.storage.col(),
            seed_tensor,
            self.fanouts,
            False,  # replace
            True,  # directed
        )

        return subset, edge_index

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

            # Use proper neighbor sampling with fanouts
            subset, edge_index = self._sample_neighbors(all_pos_nodes)

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
