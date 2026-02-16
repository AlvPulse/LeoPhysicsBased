import torch
import torch.nn as nn
import torch.nn.functional as F
# Removed GNN imports
# from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

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
        return self.linear(x)

# GNNEventDetector has been removed as per user request to replace it with Gradient Boosting.
