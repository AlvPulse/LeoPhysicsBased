import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

class LinearHarmonicModel(nn.Module):
    """
    A simple linear model that learns weights for harmonic features.
    Input: [SNR_1, Power_1, SNR_2, Power_2, ..., SNR_N, Power_N]
    Output: Probability of event
    """
    def __init__(self, num_harmonics=10, features_per_harmonic=2):
        super(LinearHarmonicModel, self).__init__()
        input_dim = num_harmonics * features_per_harmonic
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        out = self.linear(x)
        return torch.sigmoid(out)

class GNNEventDetector(torch.nn.Module):
    """
    GNN for event detection based on peak graph.
    Uses Graph Attention Networks (GAT) to weigh edges.
    """
    def __init__(self, node_features=3, hidden_channels=32, heads=2):
        super(GNNEventDetector, self).__init__()
        # Input: [freq, power, snr]
        self.conv1 = GATConv(node_features, hidden_channels, heads=heads, dropout=0.2)
        # Output dim = hidden_channels * heads

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.2)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # 2. Readout layer (aggregation)
        # Use Max Pooling + Mean Pooling
        x1 = global_max_pool(x, batch)
        x2 = global_mean_pool(x, batch)
        x = x1 + x2

        # 3. Apply a final classifier
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)

        return torch.sigmoid(x)
