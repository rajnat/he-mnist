"""Plaintext training loop."""

import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import HEFriendlyNet
from src.data import get_dataloaders


def train(epochs: int = 10, lr: float = 1e-3, save_path: str = "model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HEFriendlyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"  loss={total_loss/len(train_loader):.4f}  test_acc={acc:.2%}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
