from src.models import LinearHarmonicModel
import torch.nn as nn
import torch

class CombinedModel(nn.Module):
    """
    Combines Linear and GNN outputs for final classification.
    """
    def __init__(self, gnn_model, linear_model):
        super(CombinedModel, self).__init__()
        self.gnn = gnn_model
        self.linear = linear_model
        # Ensemble weights
        self.w_gnn = nn.Parameter(torch.tensor(0.5))
        self.w_lin = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, batch, linear_features):
        out_gnn = self.gnn(x, edge_index, batch)
        out_lin = self.linear(linear_features)

        # Weighted average of logits or probabilities?
        # Let's average the probabilities
        return self.w_gnn * out_gnn + self.w_lin * out_lin
