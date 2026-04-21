# HE-MNIST: Homomorphic Encryption on MNIST

A hands-on exploration of Fully Homomorphic Encryption (FHE) applied to image classification.

## What This Project Demonstrates

**Homomorphic Encryption** lets you perform arithmetic (add, multiply) directly on ciphertext — the server computing on your data never sees the plaintext.

This project follows the **private inference** pattern:
1. Train a lightweight neural network on plaintext MNIST images
2. Encrypt test images using the CKKS scheme (supports approximate real-valued arithmetic)
3. Run inference on the encrypted images — the model never sees the raw pixels
4. Decrypt only the final output (class logits)

## HE Concepts Used

| Concept | What it means here |
|---|---|
| **CKKS scheme** | Approximate arithmetic over real numbers (needed for floats/activations) |
| **Polynomial modulus degree** | Controls precision vs. performance tradeoff |
| **Scale** | Fixed-point encoding resolution for real numbers |
| **Noise budget** | Each multiplication consumes "budget"; too many → decryption fails |
| **Bootstrapping** | Refreshes noise budget (we avoid it by keeping the network shallow) |

## Why Shallow Networks?

Each multiply-depth level in a neural network costs one level of noise budget. Activation functions like ReLU are not polynomial, so we use **square activations** (x²) which are degree-2 polynomials — exact and HE-friendly.

## Project Structure

```
he-mnist/
├── src/
│   ├── data.py          # MNIST loading and preprocessing
│   ├── model.py         # Plaintext NN (square activations)
│   ├── train.py         # Plaintext training
│   ├── encrypt.py       # TenSEAL context setup + image encryption
│   └── he_inference.py  # Inference on encrypted images
├── notebooks/
│   └── walkthrough.ipynb  # Step-by-step explanation
├── requirements.txt
└── main.py
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```
