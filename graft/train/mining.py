"""FAISS-based hard negative mining for in-batch contrastive retrieval training."""

import torch
import faiss
import numpy as np


class HardNegativeMiner:
    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device
        self.index = None
        self.corpus_embeddings = None

    def build_index(self, corpus_texts):
        self.encoder.eval()
        embeddings = []

        for i in range(0, len(corpus_texts), 128):
            batch = corpus_texts[i : i + 128]
            batch_embeds = self.encoder.encode(batch, self.device)
            embeddings.append(batch_embeds.cpu().numpy())

        self.corpus_embeddings = np.vstack(embeddings)

        dim = self.corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.corpus_embeddings)
        self.index.add(self.corpus_embeddings)

    def mine_hard_negatives(self, queries, pos_nodes, k):
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        self.encoder.eval()
        query_embeds = self.encoder.encode(queries, self.device).cpu().numpy()
        faiss.normalize_L2(query_embeds)

        distances, indices = self.index.search(query_embeds, k + 10)

        neg_nodes = []
        for i, query_indices in enumerate(indices):
            pos_node = pos_nodes[i].item()
            negs = [idx for idx in query_indices if idx != pos_node][:k]
            neg_nodes.extend(negs)

        return torch.tensor(neg_nodes, device=self.device)
