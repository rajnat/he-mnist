"""
HE inference for DeepHENet using OpenFHE bootstrapping.

Representation: one ciphertext per neuron (list-of-ciphertexts).
  - Input image: single packed ciphertext (784 slots used)
  - After first linear: list of H1 ciphertexts, each holding one neuron value
    (all slots contain the same value due to EvalSum)
  - After activation: same list, each ciphertext squared
  - After bootstrap: same list, levels refreshed
  - After second linear: list of H2 ciphertexts
  - After output linear: list of 10 ciphertexts (one per class logit)

Why list-of-ciphertexts?
  Packing all neuron values into a single ciphertext would be more efficient
  (one bootstrap call instead of H1 bootstrap calls) but requires implementing
  the diagonal matrix encoding method. For this educational demo we use the
  simpler list representation so the code is transparent.

Architecture: 784 -> H1 -> H2 -> 10  (H1=16, H2=8 to keep bootstrap cost manageable)
Bootstrap inserted after first activation block (mm1 + sq1).
"""

import numpy as np
import openfhe as fhe
import time


NUM_SLOTS = 1024   # must match build_bootstrap_context


def _linear_packed_input(cc, ct_in: fhe.Ciphertext, weight: np.ndarray,
                         bias: np.ndarray, in_dim: int) -> list:
    """
    Linear layer: packed ciphertext input → list of scalar ciphertexts.

    For each output neuron i:
      1. Element-wise multiply ct_in by weight row i (plaintext)
      2. EvalSum over in_dim slots → all slots hold the dot product
      3. Add scalar bias[i]
    Returns list of len(bias) ciphertexts.
    """
    out_dim = weight.shape[0]
    outputs = []
    for i in range(out_dim):
        row = weight[i].tolist() + [0.0] * (NUM_SLOTS - in_dim)
        row_pt   = cc.MakeCKKSPackedPlaintext(row)
        ct_prod  = cc.EvalMult(ct_in, row_pt)
        ct_sum   = cc.EvalSum(ct_prod, in_dim)
        bias_pt  = cc.MakeCKKSPackedPlaintext([bias[i]])
        outputs.append(cc.EvalAdd(ct_sum, bias_pt))
    return outputs


def _linear_list_input(cc, cts_in: list, weight: np.ndarray,
                       bias: np.ndarray) -> list:
    """
    Linear layer: list of scalar ciphertexts → list of scalar ciphertexts.

    Each input ciphertext holds one neuron value in all slots (from EvalSum).
    For output neuron j: sum_i(w[j,i] * ct_in[i]) + bias[j]
    """
    out_dim, in_dim = weight.shape
    assert len(cts_in) == in_dim
    outputs = []
    for j in range(out_dim):
        result = None
        for i, ct in enumerate(cts_in):
            w_pt = cc.MakeCKKSPackedPlaintext([float(weight[j, i])])
            term = cc.EvalMult(ct, w_pt)
            result = term if result is None else cc.EvalAdd(result, term)
        bias_pt = cc.MakeCKKSPackedPlaintext([float(bias[j])])
        outputs.append(cc.EvalAdd(result, bias_pt))
    return outputs


def _square(cc, cts: list) -> list:
    """x² activation applied to each ciphertext in the list."""
    return [cc.EvalMult(ct, ct) for ct in cts]


def _bootstrap_list(cc, cts: list) -> list:
    """Bootstrap every ciphertext in the list, restoring its level."""
    refreshed = []
    for i, ct in enumerate(cts):
        t0 = time.time()
        ct_fresh = cc.EvalBootstrap(ct)
        refreshed.append(ct_fresh)
        print(f"    bootstrapped neuron {i:2d}  "
              f"level {ct.GetLevel()} → {ct_fresh.GetLevel()}  "
              f"({time.time()-t0:.1f}s)")
    return refreshed


def _decrypt_list(cc, keypair, cts: list) -> list:
    """Decrypt a list of scalar ciphertexts → list of floats."""
    logits = []
    for ct in cts:
        result = cc.Decrypt(keypair.secretKey, ct)
        result.SetLength(1)
        logits.append(result.GetRealPackedValue()[0])
    return logits


def he_forward_deep(cc, keypair, enc_image: fhe.Ciphertext,
                    weights: dict, in_dim: int = 784) -> list:
    """
    Forward pass for DeepHENet on one encrypted image.

    Level budget usage (OpenFHE counts up from 0):
      mm1 : level +1 (EvalMult in packed linear)
      sq1 : level +1
      --- BOOTSTRAP ---  (level reset to ~LEVELS_AFTER)
      mm2 : level +1
      sq2 : level +1
      mm3 : level +1
    """
    print(f"  [mm1]  level {enc_image.GetLevel()}", end=" → ")
    cts1 = _linear_packed_input(cc, enc_image, weights["w1"], weights["b1"], in_dim)
    print(f"{cts1[0].GetLevel()}")

    print(f"  [sq1]  level {cts1[0].GetLevel()}", end=" → ")
    cts1 = _square(cc, cts1)
    print(f"{cts1[0].GetLevel()}")

    print(f"  [bootstrap]  refreshing {len(cts1)} ciphertexts ...")
    cts1 = _bootstrap_list(cc, cts1)

    print(f"  [mm2]  level {cts1[0].GetLevel()}", end=" → ")
    cts2 = _linear_list_input(cc, cts1, weights["w2"], weights["b2"])
    print(f"{cts2[0].GetLevel()}")

    print(f"  [sq2]  level {cts2[0].GetLevel()}", end=" → ")
    cts2 = _square(cc, cts2)
    print(f"{cts2[0].GetLevel()}")

    print(f"  [mm3]  level {cts2[0].GetLevel()}", end=" → ")
    cts_out = _linear_list_input(cc, cts2, weights["w3"], weights["b3"])
    print(f"{cts_out[0].GetLevel()}")

    print("  [decrypt]")
    return _decrypt_list(cc, keypair, cts_out)


def run_deep_he_inference(cc, keypair, enc_images: list,
                          weights: dict, true_labels: list):
    correct = 0
    results = []
    for i, (enc_img, label) in enumerate(zip(enc_images, true_labels)):
        print(f"\nSample {i}  (true label: {label})")
        t0 = time.time()
        logits = he_forward_deep(cc, keypair, enc_img, weights)
        pred   = int(np.argmax(logits))
        correct += int(pred == label)
        status = "✓" if pred == label else "✗"
        print(f"  [{status}] predicted={pred}  time={time.time()-t0:.1f}s")
        results.append({"index": i, "true_label": label,
                        "predicted": pred, "correct": pred == label})

    acc = correct / len(enc_images)
    print(f"\nDeep HE accuracy: {correct}/{len(enc_images)} = {acc:.2%}")
    return results
