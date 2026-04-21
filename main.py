"""
HE-MNIST: end-to-end pipeline.

Default mode  (TenSEAL, no bootstrapping):
  1. Train HEFriendlyNet (784->128->64->10) on plaintext MNIST.
  2. Encrypt test images with CKKS (TenSEAL).
  3. Run inference on ciphertext — model never sees raw pixels.

Deep mode  --deep  (OpenFHE, with bootstrapping):
  1. Train DeepHENet (784->256->128->64->10) on plaintext MNIST.
  2. Encrypt test images with CKKS (OpenFHE).
  3. Run inference with a mid-network bootstrap to restore the noise budget.
"""

import os
import argparse
import time

import torch

from src.data import get_flat_test_samples
from src.model import HEFriendlyNet, DeepHENet
from src.train import train, evaluate, get_dataloaders
from src.encrypt import build_context, encrypt_batch
from src.he_inference import run_he_inference


MODEL_PATH      = "model.pt"
DEEP_MODEL_PATH = "model_deep.pt"


def load_or_train_shallow(epochs):
    model = HEFriendlyNet()
    if os.path.exists(MODEL_PATH):
        print(f"Loading cached model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        print("Training HEFriendlyNet from scratch.")
        model = train(epochs=epochs, save_path=MODEL_PATH)
    return model


def load_or_train_deep(epochs):
    from src.train import train_model
    model = DeepHENet()
    if os.path.exists(DEEP_MODEL_PATH):
        print(f"Loading cached deep model from {DEEP_MODEL_PATH}")
        model.load_state_dict(torch.load(DEEP_MODEL_PATH, weights_only=True))
    else:
        print("Training DeepHENet from scratch.")
        model = train_model(model, epochs=epochs, save_path=DEEP_MODEL_PATH)
    return model


def run_shallow(args):
    print("=" * 60)
    print("STEP 1: Plaintext Training  (HEFriendlyNet, TenSEAL)")
    print("=" * 60)
    model = load_or_train_shallow(args.epochs)
    model.eval()

    _, test_loader = get_dataloaders()
    plain_acc = evaluate(model, test_loader, torch.device("cpu"))
    print(f"Plaintext test accuracy: {plain_acc:.2%}")

    print("\n" + "=" * 60)
    print("STEP 2: Building CKKS Context & Encrypting Images")
    print("=" * 60)
    context = build_context()
    images, labels = get_flat_test_samples(n=args.he_samples)
    t0 = time.time()
    enc_images = encrypt_batch(context, images)
    enc_time = time.time() - t0
    print(f"Encrypted {args.he_samples} images in {enc_time:.2f}s")

    print("\n" + "=" * 60)
    print("STEP 3: Inference on Encrypted Images  (no bootstrapping)")
    print("=" * 60)
    weights = model.get_weights()
    t0 = time.time()
    results = run_he_inference(enc_images, weights, labels)
    he_time = time.time() - t0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    he_correct = sum(r["correct"] for r in results)
    print(f"Plaintext accuracy (full test set) : {plain_acc:.2%}")
    print(f"HE inference accuracy ({args.he_samples} samples)    : "
          f"{he_correct}/{args.he_samples} = {he_correct/args.he_samples:.2%}")
    print(f"HE inference time                  : {he_time:.2f}s  "
          f"({he_time/args.he_samples:.2f}s/sample)")


def run_deep(args):
    """
    Train DeepHENet and export weights as numpy arrays.
    HE inference runs in a separate process (run_deep_he.py) to avoid
    the OpenMP conflict between torch and openfhe when loaded together.
    """
    import numpy as np

    print("=" * 60)
    print("STEP 1: Plaintext Training  (DeepHENet)")
    print("=" * 60)
    model = load_or_train_deep(args.epochs)
    model.eval()

    _, test_loader = get_dataloaders()
    plain_acc = evaluate(model, test_loader, torch.device("cpu"))
    print(f"Plaintext test accuracy: {plain_acc:.2%}")

    # Export weights as numpy arrays so run_deep_he.py can load them
    # without importing torch (avoiding the libomp conflict).
    weights = model.get_weights()
    np.savez("deep_weights.npz", **weights)
    print("Weights saved to deep_weights.npz")

    # Export test image labels so run_deep_he.py knows what to expect
    from src.data import get_flat_test_samples
    images, labels = get_flat_test_samples(n=args.he_samples)
    np.savez("deep_test_data.npz",
             images=np.array(images),
             labels=np.array(labels))
    print(f"Test data ({args.he_samples} samples) saved to deep_test_data.npz")

    print(f"\nTo run HE inference with bootstrapping:")
    print(f"  uv run python run_deep_he.py --he-samples {args.he_samples}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--he-samples", type=int, default=5,
                        help="Images to encrypt+evaluate (deep mode is slow — keep small)")
    parser.add_argument("--retrain",    action="store_true")
    parser.add_argument("--deep",       action="store_true",
                        help="Use DeepHENet + OpenFHE bootstrapping instead of TenSEAL")
    args = parser.parse_args()

    if args.retrain:
        for p in [MODEL_PATH, DEEP_MODEL_PATH]:
            if os.path.exists(p):
                os.remove(p)

    if args.deep:
        run_deep(args)
    else:
        run_shallow(args)


if __name__ == "__main__":
    main()
