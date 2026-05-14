"""PyTorch model for subcultural identity and mental health prediction."""

import torch
import torch.nn as nn


class SubCultureNN(nn.Module):
    """Neural network for subcultural identity to mental health prediction."""

    def __init__(self, input_dims=5, output_dims=13):
        super(SubCultureNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dims)
        )

    def forward(self, x):
        return self.net(x)


def create_model(input_dims=5, output_dims=13):
    """Create and return a new model."""
    return SubCultureNN(input_dims, output_dims)


def save_model(model, path):
    """Save model weights."""
    torch.save(model.state_dict(), path)


def load_model(path, input_dims=5, output_dims=13):
    """Load model weights."""
    model = SubCultureNN(input_dims, output_dims)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
