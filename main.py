"""
HE-MNIST: end-to-end pipeline.

Steps:
  1. Train HEFriendlyNet on plaintext MNIST (or load cached weights).
  2. Build a TenSEAL CKKS context and encrypt N test images.
  3. Run forward pass on ciphertext — model never sees raw pixels.
  4. Decrypt only the output logits and report accuracy.
"""

import argparse
import os
import time

import torch

from src.data import get_flat_test_samples
from src.model import HEFriendlyNet
from src.train import train, evaluate, get_dataloaders
from src.encrypt import build_context, encrypt_batch
from src.he_inference import run_he_inference


MODEL_PATH = "model.pt"


def load_or_train(epochs: int = 10) -> HEFriendlyNet:
    model = HEFriendlyNet()
    if os.path.exists(MODEL_PATH):
        print(f"Loading cached model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        print("No cached model found — training from scratch.")
        model = train(epochs=epochs, save_path=MODEL_PATH)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--he-samples", type=int, default=10,
                        help="Number of test images to encrypt and evaluate")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model exists")
    args = parser.parse_args()

    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    # --- Step 1: Plaintext training ---
    print("=" * 60)
    print("STEP 1: Plaintext Training")
    print("=" * 60)
    model = load_or_train(epochs=args.epochs)
    model.eval()

    _, test_loader = get_dataloaders()
    device = torch.device("cpu")
    plain_acc = evaluate(model, test_loader, device)
    print(f"Plaintext test accuracy: {plain_acc:.2%}")

    # --- Step 2: Build HE context and encrypt images ---
    print("\n" + "=" * 60)
    print("STEP 2: Building CKKS Context & Encrypting Images")
    print("=" * 60)
    print("Parameters: poly_modulus_degree=16384, scale=2^40, depth=6 (mm+sq+mm+sq+mm = 5 ops, +1 buffer)")

    context = build_context()
    images, labels = get_flat_test_samples(n=args.he_samples)

    t0 = time.time()
    enc_images = encrypt_batch(context, images)
    enc_time = time.time() - t0
    print(f"Encrypted {args.he_samples} images in {enc_time:.2f}s "
          f"({enc_time/args.he_samples:.3f}s per image)")

    # --- Step 3: HE Inference ---
    print("\n" + "=" * 60)
    print("STEP 3: Inference on Encrypted Images")
    print("=" * 60)
    print("Model weights are plaintext. Input images stay encrypted throughout.\n")

    weights = model.get_weights()

    t0 = time.time()
    results = run_he_inference(enc_images, weights, labels)
    he_time = time.time() - t0

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    he_correct = sum(r["correct"] for r in results)
    print(f"Plaintext accuracy (full test set): {plain_acc:.2%}")
    print(f"HE inference accuracy ({args.he_samples} samples):  "
          f"{he_correct}/{args.he_samples} = {he_correct/args.he_samples:.2%}")
    print(f"HE inference time: {he_time:.2f}s ({he_time/args.he_samples:.2f}s per sample)")
    print("\nNote: accuracy difference is expected — CKKS uses approximate arithmetic.")


if __name__ == "__main__":
    main()
