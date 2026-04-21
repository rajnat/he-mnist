"""
Plaintext neural network with square activations.

Why square activations (x²) instead of ReLU?
  - ReLU is not a polynomial, so it can't be evaluated on ciphertext.
  - Polynomials can be. x² is degree-2, so each layer costs exactly one
    multiplicative depth level in the CKKS noise budget.
  - Keeping the network to 2 layers means depth=2, which is feasible without
    bootstrapping.
"""

import torch
import torch.nn as nn


class SquareActivation(nn.Module):
    def forward(self, x):
        return x * x


class HEFriendlyNet(nn.Module):
    """
    Architecture: 784 -> 128 -> 64 -> 10
    Two hidden layers, each followed by x² activation.
    Multiplicative depth = 2 (one per activation layer).
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = SquareActivation()
        self.fc2 = nn.Linear(128, 64)
        self.act2 = SquareActivation()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

    def get_weights(self):
        """Return all weight/bias tensors as plain lists for HE inference."""
        return {
            "w1": self.fc1.weight.detach().numpy(),
            "b1": self.fc1.bias.detach().numpy(),
            "w2": self.fc2.weight.detach().numpy(),
            "b2": self.fc2.bias.detach().numpy(),
            "w3": self.fc3.weight.detach().numpy(),
            "b3": self.fc3.bias.detach().numpy(),
        }
