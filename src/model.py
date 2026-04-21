"""
Plaintext neural networks with square activations.

HEFriendlyNet  — 2 hidden layers, no bootstrapping needed (TenSEAL)
DeepHENet      — 3 hidden layers, requires bootstrapping mid-forward-pass (OpenFHE)

Why square activations (x²) instead of ReLU?
  - ReLU is not a polynomial, so it can't be evaluated on ciphertext.
  - x² is degree-2, costs exactly one multiplicative depth level per call.
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


class DeepHENet(nn.Module):
    """
    Architecture: 784 -> 16 -> 8 -> 10
    Two hidden layers with x² activations, requires bootstrapping mid-inference.

    The network itself has the same depth as HEFriendlyNet, but the list-of-
    ciphertexts HE representation means each neuron is a separate ciphertext.
    Bootstrapping refreshes all hidden-layer ciphertexts after sq1, allowing
    the second half of the network to run with a fresh level budget.

    HE inference layout:
      [mm1 + sq1]  →  BOOTSTRAP (16 ciphertexts)  →  [mm2 + sq2 + mm3]

    Architecture is intentionally small (16, 8 hidden units) so that bootstrapping
    16 ciphertexts per sample is feasible in a demo setting.
    Expected plaintext accuracy: ~88-92% (small capacity, simple dataset).
    """
    H1 = 16
    H2 = 8

    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, self.H1)
        self.act1 = SquareActivation()
        self.fc2  = nn.Linear(self.H1, self.H2)
        self.act2 = SquareActivation()
        self.fc3  = nn.Linear(self.H2, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

    def get_weights(self):
        return {
            "w1": self.fc1.weight.detach().numpy(),
            "b1": self.fc1.bias.detach().numpy(),
            "w2": self.fc2.weight.detach().numpy(),
            "b2": self.fc2.bias.detach().numpy(),
            "w3": self.fc3.weight.detach().numpy(),
            "b3": self.fc3.bias.detach().numpy(),
        }
