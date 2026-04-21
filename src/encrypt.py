"""
TenSEAL context setup and image encryption using the CKKS scheme.

Key CKKS parameters explained:
  poly_modulus_degree (n):
    - Must be a power of 2. Controls the "ring" size.
    - Larger n → more noise budget + can pack more values, but slower.
    - 16384 supports up to 438 total modulus bits; our chain uses 360.

  coeff_mod_bit_sizes:
    - A list of bit-sizes for the coefficient modulus chain.
    - First and last entries are special primes (setup/teardown).
    - Each middle entry = one rescaling slot, consumed by one operation.
    - IMPORTANT: in TenSEAL, BOTH mm and square consume one slot each,
      because mm multiplies ciphertext by plaintext (doubling the scale)
      and then rescales to restore it — that rescale spends a chain slot.
    - Our forward pass has 5 operations: mm, sq, mm, sq, mm → needs depth 5.
    - [60, 40×6, 60] gives depth 6 (one slot of headroom). Total = 360 bits.

  scale (2^40):
    - Fixed-point encoding resolution. Higher = more precision, less noise budget.
    - Must match middle coeff_mod_bit_sizes entries.

  global_scale:
    - TenSEAL convenience: auto-applies scale to every encoding.
"""

import tenseal as ts
import numpy as np


def build_context() -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()  # needed for vector rotations (matmul)
    return context


def encrypt_image(context: ts.Context, flat_image: list) -> ts.CKKSVector:
    """Encrypt a flat 784-element pixel vector as a single CKKS ciphertext."""
    return ts.ckks_vector(context, flat_image)


def encrypt_batch(context: ts.Context, images: list) -> list:
    """Encrypt a list of flat images. Returns list of CKKSVectors."""
    return [encrypt_image(context, img) for img in images]
