"""GraphSAGE message passing for relation-aware node embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        for i in range(layers):
            in_channels = in_dim if i == 0 else hidden_dim
            self.layers.append(SAGEConv(in_channels, hidden_dim))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
