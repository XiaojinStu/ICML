#!/usr/bin/env python3
"""
Quick test script for NAE-TTA v2.1
Runs a minimal experiment to verify implementation correctness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from nae_core import angular_entropy_stable, safe_normalize, stable_arccos
from numerical_utils import get_numerical_tokens, apply_numerical_mask

def test_nae_computation():
    """Test NAE computation with synthetic data"""
    print("=" * 60)
    print("Test 1: NAE Computation")
    print("=" * 60)

    # Create synthetic logits and embeddings
    vocab_size = 100
    embed_dim = 64
    num_idx = list(range(10))  # First 10 tokens are numerical

    # Synthetic embedding layer
    class MockEmbedLayer:
        def __init__(self):
            torch.manual_seed(42)
            self.weight = torch.randn(vocab_size, embed_dim)

    embed_layer = MockEmbedLayer()

    # Test case 1: Uniform distribution
    logits = torch.zeros(vocab_size)
    nae_uniform = angular_entropy_stable(logits, num_idx, embed_layer, use_float32=True)
    print(f"  Uniform logits -> NAE: {nae_uniform.item():.6f}")

    # Test case 2: Concentrated distribution
    logits_concentrated = torch.zeros(vocab_size)
    logits_concentrated[0] = 10.0  # High probability on first token
    nae_concentrated = angular_entropy_stable(logits_concentrated, num_idx, embed_layer, use_float32=True)
    print(f"  Concentrated logits -> NAE: {nae_concentrated.item():.6f}")

    # Verify: concentrated should have lower NAE than uniform
    assert nae_concentrated < nae_uniform, "Concentrated distribution should have lower NAE"
    print("  PASSED: Concentrated < Uniform")

    # Test case 3: Check gradient flow
    logits_grad = torch.zeros(vocab_size, requires_grad=True)
    nae_grad = angular_entropy_stable(logits_grad, num_idx, embed_layer, use_float32=True)
    nae_grad.backward()
    assert logits_grad.grad is not None, "Gradients should flow through NAE"
    assert not torch.isnan(logits_grad.grad).any(), "Gradients should not contain NaN"
    print("  PASSED: Gradient flow verified")

    # Test case 4: Check NaN handling
    logits_extreme = torch.zeros(vocab_size)
    logits_extreme[0] = 1000.0  # Very extreme
    nae_extreme = angular_entropy_stable(logits_extreme, num_idx, embed_layer, use_float32=True)
    assert not torch.isnan(nae_extreme), "NAE should not be NaN for extreme inputs"
    print(f"  Extreme logits -> NAE: {nae_extreme.item():.6f}")
    print("  PASSED: No NaN for extreme inputs")

    print("\nAll NAE computation tests PASSED!\n")


def test_numerical_mask():
    """Test numerical masking"""
    print("=" * 60)
    print("Test 2: Numerical Masking")
    print("=" * 60)

    vocab_size = 100
    num_idx = [0, 1, 2, 3, 4]  # Only these are numerical

    logits = torch.randn(vocab_size)
    masked = apply_numerical_mask(logits, num_idx)

    # Check that non-numerical positions are -inf
    non_num_mask = torch.ones(vocab_size, dtype=torch.bool)
    non_num_mask[num_idx] = False

    assert (masked[non_num_mask] == float('-inf')).all(), "Non-numerical should be -inf"
    print("  PASSED: Non-numerical tokens masked to -inf")

    # Check that numerical positions are unchanged
    num_idx_tensor = torch.tensor(num_idx)
    assert torch.allclose(masked[num_idx_tensor], logits[num_idx_tensor]), "Numerical should be unchanged"
    print("  PASSED: Numerical tokens preserved")

    # Check that argmax gives numerical output
    pred = torch.argmax(masked).item()
    assert pred in num_idx, "Prediction should be numerical token"
    print(f"  PASSED: Prediction {pred} is in numerical set")

    print("\nAll numerical masking tests PASSED!\n")


def test_safe_normalize():
    """Test safe normalization"""
    print("=" * 60)
    print("Test 3: Safe Normalization")
    print("=" * 60)

    # Normal case
    x = torch.randn(10, 64)
    x_norm = safe_normalize(x, eps=1e-8, dim=-1)
    norms = torch.norm(x_norm, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Norms should be ~1"
    print("  PASSED: Normal vectors normalized correctly")

    # Near-zero case
    x_small = torch.zeros(64) + 1e-10
    x_small_norm = safe_normalize(x_small.unsqueeze(0), eps=1e-8, dim=-1)
    assert not torch.isnan(x_small_norm).any(), "Should not produce NaN for near-zero input"
    print("  PASSED: Near-zero input handled without NaN")

    # Zero case
    x_zero = torch.zeros(64)
    x_zero_norm = safe_normalize(x_zero.unsqueeze(0), eps=1e-8, dim=-1)
    assert not torch.isnan(x_zero_norm).any(), "Should not produce NaN for zero input"
    print("  PASSED: Zero input handled without NaN")

    print("\nAll safe normalize tests PASSED!\n")


def test_stable_arccos():
    """Test stable arccos"""
    print("=" * 60)
    print("Test 4: Stable Arccos")
    print("=" * 60)

    # Normal range
    x = torch.tensor([-0.9, -0.5, 0.0, 0.5, 0.9])
    angles = stable_arccos(x, eps=1e-4)
    assert not torch.isnan(angles).any(), "Normal range should not produce NaN"
    print("  PASSED: Normal range computed correctly")

    # Boundary cases
    x_boundary = torch.tensor([-1.0, 1.0])
    angles_boundary = stable_arccos(x_boundary, eps=1e-4)
    assert not torch.isnan(angles_boundary).any(), "Boundary values should be handled"
    print(f"  Boundary angles: {angles_boundary.tolist()}")
    print("  PASSED: Boundary cases handled")

    # Out of range (should be clamped)
    x_out = torch.tensor([-1.5, 1.5])
    angles_out = stable_arccos(x_out, eps=1e-4)
    assert not torch.isnan(angles_out).any(), "Out of range should be clamped"
    print("  PASSED: Out of range values clamped")

    print("\nAll stable arccos tests PASSED!\n")


def main():
    print("\n" + "=" * 60)
    print("NAE-TTA v2.1 Quick Test Suite")
    print("=" * 60 + "\n")

    test_safe_normalize()
    test_stable_arccos()
    test_numerical_mask()
    test_nae_computation()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe implementation is ready for experiments.")
    print("Run: python nae_tta.py --exp_name test --num_samples 5")


if __name__ == "__main__":
    main()
