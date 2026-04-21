"""
OpenFHE CKKS context with bootstrapping enabled.

What bootstrapping does
-----------------------
After enough multiplications, a ciphertext's modulus chain is exhausted —
no more rescaling slots remain. The ciphertext is "stuck" at level 0.

Bootstrapping homomorphically evaluates the CKKS decryption circuit on the
ciphertext itself, producing a fresh ciphertext at a high level that encodes
the same approximate value. The server never learns the plaintext: the
secret key is never used directly; instead it is encoded into the scheme's
public evaluation keys during setup.

The four internal steps of CKKS bootstrapping
----------------------------------------------
1. ModRaise    : Lift the ciphertext from small modulus q₀ back to full Q.
                 This introduces approximate multiples of q₀ as error.
2. CoeffToSlot : Apply a homomorphic DFT to move polynomial coefficients
                 into the encoding slots. Costs levelBudget[0] levels.
3. EvalMod     : Homomorphically compute x mod q₀ using a polynomial
                 approximation of (q₀/2π)·sin(2πx/q₀). This is the most
                 expensive step — it removes the error introduced by ModRaise.
                 Depth cost ≈ log₂(polynomial degree).
4. SlotToCoeff : Inverse DFT back to coefficient form. Costs levelBudget[1].

Total levels consumed per bootstrap ≈ levelBudget[0] + evalMod_depth + levelBudget[1]
For levelBudget=[4,4] OpenFHE reports this as 22 levels (EvalMod alone costs 14).

Parameter choices
-----------------
LEVELS_BEFORE : levels available to consume before bootstrapping.
                Our DeepHENet uses mm+sq = 2 ops before the bootstrap point.
LEVELS_AFTER  : levels available after bootstrapping.
                The rest of DeepHENet needs mm+sq+mm+sq+mm = 5 ops.
ringDim       : must be ≥ 2^16 for bootstrapping (larger DFT support needed).
"""

import openfhe as fhe

LEVEL_BUDGET   = [4, 4]
LEVELS_BEFORE  = 2   # mm1 + sq1
LEVELS_AFTER   = 6   # mm2 + sq2 + mm3 + sq3 + mm4 + 1 buffer


def get_bootstrap_depth() -> int:
    """Levels consumed by one bootstrap call with levelBudget=[4,4]."""
    return fhe.FHECKKSRNS.GetBootstrapDepth(LEVEL_BUDGET, fhe.UNIFORM_TERNARY)


def build_bootstrap_context(num_slots: int = 1024):
    """
    Build an OpenFHE CKKS context with bootstrapping enabled.

    num_slots : values packed per ciphertext (must be power of 2, ≤ ringDim/2).
                784 MNIST pixels fit in 1024 slots.
    """
    boot_depth  = get_bootstrap_depth()
    total_depth = LEVELS_BEFORE + boot_depth + LEVELS_AFTER

    params = fhe.CCParamsCKKSRNS()
    params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    params.SetSecurityLevel(fhe.HEStd_NotSet)   # relaxed for demo speed
    params.SetRingDim(1 << 16)                   # 65536 — minimum for bootstrapping
    params.SetScalingModSize(59)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(num_slots)
    params.SetMultiplicativeDepth(total_depth)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKE)
    cc.Enable(fhe.KEYSWITCH)
    cc.Enable(fhe.LEVELEDSHE)
    cc.Enable(fhe.ADVANCEDSHE)
    cc.Enable(fhe.FHE)           # FHE flag unlocks EvalBootstrap

    keypair = cc.KeyGen()
    cc.EvalMultKeyGen(keypair.secretKey)

    import math
    rot_indices = [2**i for i in range(int(math.log2(num_slots)) + 1)]
    cc.EvalRotateKeyGen(keypair.secretKey, rot_indices)

    cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], num_slots)
    cc.EvalBootstrapKeyGen(keypair.secretKey, num_slots)

    print("Bootstrap context ready:")
    print(f"  Ring dimension    : {1 << 16}")
    print(f"  Slots per ct      : {num_slots}")
    print(f"  Levels before     : {LEVELS_BEFORE}  (mm1 + sq1)")
    print(f"  Bootstrap depth   : {boot_depth}  (CoeffToSlot + EvalMod + SlotToCoeff)")
    print(f"  Levels after      : {LEVELS_AFTER}  (mm2 + sq2 + mm3 + sq3 + mm4 + buffer)")
    print(f"  Total chain depth : {total_depth}")

    return cc, keypair
