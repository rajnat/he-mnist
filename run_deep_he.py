"""
Deep HE inference with bootstrapping — runs WITHOUT importing torch.

torch and openfhe both bundle their own copy of libomp. Loading both in the
same process causes an OpenMP conflict and a segfault on macOS. The solution
is to keep them in separate processes:

  main.py         — trains the model, exports weights to deep_weights.npz
  run_deep_he.py  — loads weights from npz, runs OpenFHE bootstrapping inference

Usage:
  uv run python main.py --deep          # train + export weights
  uv run python run_deep_he.py          # HE inference with bootstrapping
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import numpy as np
import openfhe as fhe

from src.bootstrap import build_bootstrap_context
from src.he_inference_deep import run_deep_he_inference, NUM_SLOTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-samples",   type=int, default=3,
                        help="Number of encrypted images to evaluate")
    parser.add_argument("--weights-file", default="deep_weights.npz")
    parser.add_argument("--data-file",    default="deep_test_data.npz")
    args = parser.parse_args()

    # Load weights exported by main.py --deep
    if not os.path.exists(args.weights_file):
        print(f"ERROR: {args.weights_file} not found.")
        print("Run first:  uv run python main.py --deep")
        return

    w = np.load(args.weights_file)
    weights = {k: w[k] for k in w.files}
    print(f"Loaded weights: { {k: v.shape for k,v in weights.items()} }")

    d = np.load(args.data_file)
    images = d["images"][:args.he_samples].tolist()
    labels = d["labels"][:args.he_samples].tolist()
    print(f"Loaded {len(images)} test images, labels: {labels}")

    # Build OpenFHE bootstrap context
    print("\n" + "=" * 60)
    print("Building Bootstrap CKKS Context")
    print("=" * 60)
    cc, keypair = build_bootstrap_context(num_slots=NUM_SLOTS)

    # Encrypt images
    print("\nEncrypting images ...")
    t0 = time.time()
    enc_images = []
    for img in images:
        padded = img + [0.0] * (NUM_SLOTS - len(img))
        pt = cc.MakeCKKSPackedPlaintext(padded)
        enc_images.append(cc.Encrypt(keypair.publicKey, pt))
    print(f"Encrypted {len(images)} images in {time.time()-t0:.2f}s")

    # Run deep HE inference with bootstrapping
    print("\n" + "=" * 60)
    print("Deep HE Inference  (with bootstrapping)")
    print("=" * 60)
    print("Bootstrap is inserted after sq1 to refresh the level budget.")
    print("Watch the level counter go up, reset at BOOTSTRAP, then continue.\n")

    t0 = time.time()
    results = run_deep_he_inference(cc, keypair, enc_images, weights, labels)
    total = time.time() - t0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    he_correct = sum(r["correct"] for r in results)
    print(f"HE accuracy : {he_correct}/{len(results)} = {he_correct/len(results):.2%}")
    print(f"Total time  : {total:.1f}s  ({total/len(results):.1f}s/sample)")
    print("\nNote: most of the per-sample time is the bootstrap call.")


if __name__ == "__main__":
    main()
