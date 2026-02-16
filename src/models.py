import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

class LinearHarmonicModel(nn.Module):
    """
    Learns a weighted combination of harmonic features.
    Input: [SNR_1, Power_1, ..., SNR_N, Power_N] (default N=10)
    Output: Probability of event
    """
    def __init__(self, num_harmonics=10, features_per_harmonic=2):
        super(LinearHarmonicModel, self).__init__()
        input_dim = num_harmonics * features_per_harmonic

        # Simple Linear Classifier (Logistic Regression equivalent)
        # Weights represent importance of each harmonic's SNR/Power
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        logits = self.linear(x)
        return logits

class GNNEventDetector(torch.nn.Module):
    """
    GNN for event detection based on peak graph.
    Uses Graph Attention Networks (GAT) to learn relationships between peaks.
    """
    def __init__(self, node_features=3, hidden_channels=32, heads=2):
        super(GNNEventDetector, self).__init__()
        # Input: [freq_norm, power_norm, snr_norm]

        # GAT Layer 1
        self.conv1 = GATConv(node_features, hidden_channels, heads=heads, dropout=0.2)
        # Output dim = hidden_channels * heads

        # GAT Layer 2
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.2)

        # Classifier
        # Global Max + Mean Pooling -> 2 * hidden_channels
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
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
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = torch.cat([x_max, x_mean], dim=1)

        # 3. Apply a final classifier
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)

        return x
