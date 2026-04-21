"""
Inference on encrypted images.

The forward pass mirrors HEFriendlyNet but every operation is performed
on CKKSVectors (ciphertext). The model weights remain plaintext — only
the image (input) is encrypted. This is the standard "private inference"
or "secure inference" setup.

Noise budget note:
  - fc1 matmul + bias add: no depth consumed (linear)
  - act1 (square): consumes 1 multiplication level
  - fc2 matmul + bias add: no depth consumed
  - act2 (square): consumes 1 multiplication level
  - fc3 matmul + bias add: no depth consumed
  Total depth used = 2, matching [60, 40, 40, 60] coeff chain.
"""

import numpy as np
import tenseal as ts


def he_linear(enc_vec: ts.CKKSVector, weight: np.ndarray, bias: np.ndarray) -> ts.CKKSVector:
    """
    Compute weight @ enc_vec + bias entirely in ciphertext.
    TenSEAL's mm_row_vector evaluates the dot products for each output neuron.
    """
    return enc_vec.mm(weight.T.tolist()) + bias.tolist()


def he_square(enc_vec: ts.CKKSVector) -> ts.CKKSVector:
    """Square activation: x² — polynomial, so it works on ciphertext."""
    return enc_vec.square()


def he_forward(enc_image: ts.CKKSVector, weights: dict) -> list:
    """
    Run the full HEFriendlyNet forward pass on an encrypted image.
    Returns decrypted logits (10-element list).
    The image is never decrypted inside this function.
    """
    x = he_linear(enc_image, weights["w1"], weights["b1"])
    x = he_square(x)
    x = he_linear(x, weights["w2"], weights["b2"])
    x = he_square(x)
    x = he_linear(x, weights["w3"], weights["b3"])
    return x.decrypt()


def run_he_inference(enc_images: list, weights: dict, true_labels: list):
    """Evaluate HE inference on a list of encrypted images."""
    correct = 0
    results = []

    for i, (enc_img, label) in enumerate(zip(enc_images, true_labels)):
        logits = he_forward(enc_img, weights)
        pred = int(np.argmax(logits))
        correct += int(pred == label)
        results.append({
            "index": i,
            "true_label": label,
            "predicted": pred,
            "correct": pred == label,
            "logits": logits,
        })
        status = "✓" if pred == label else "✗"
        print(f"  [{status}] sample {i}: true={label} pred={pred}")

    acc = correct / len(enc_images)
    print(f"\nHE Inference accuracy: {correct}/{len(enc_images)} = {acc:.2%}")
    return results
