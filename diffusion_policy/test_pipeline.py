#!/usr/bin/env python3
"""
Quick test script to verify the diffusion policy pipeline works.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from diffusion_policy.models.diffusion_unet import DiffusionUNet
from diffusion_policy.utils.dataset import G1TrajectoryDataset
from diffusion_policy.utils.normalizer import LinearNormalizer


def test_dataset():
    """Test dataset loading."""
    print("Testing dataset loading...")
    dataset = G1TrajectoryDataset(
        data_dir="../data",
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
    )

    print(f"✓ Loaded {len(dataset.trajectories)} trajectories")
    print(f"✓ Total samples: {len(dataset)}")

    sample = dataset[0]
    print(f"✓ Sample obs shape: {sample['obs'].shape}")
    print(f"✓ Sample actions shape: {sample['actions'].shape}")

    return dataset


def test_model(dataset):
    """Test model creation and forward pass."""
    print("\nTesting model...")

    sample = dataset[0]
    obs_dim = sample['obs'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    model = DiffusionUNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        num_diffusion_iters=10,  # Small for testing
        down_dims=(64, 128, 256),  # Small for testing
        obs_encoder_layers=(128, 128),
    )

    print(f"✓ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    obs = sample['obs'].unsqueeze(0)  # Add batch dimension
    actions = sample['actions'].unsqueeze(0)

    # Compute loss
    loss_dict = model.compute_loss(obs, actions)
    print(f"✓ Forward pass successful, loss: {loss_dict['loss'].item():.4f}")

    # Test sampling
    sampled_actions = model.conditional_sample(obs, use_ddim=True, ddim_steps=5)
    print(f"✓ Sampling successful, shape: {sampled_actions.shape}")

    return model


def test_normalizer(dataset):
    """Test data normalization."""
    print("\nTesting normalizer...")

    stats = dataset.get_stats()
    normalizer = LinearNormalizer(mode='limits')

    normalizer.params['obs'] = {
        'min': stats['obs']['min'],
        'max': stats['obs']['max'],
        'scale': stats['obs']['max'] - stats['obs']['min'],
    }
    normalizer.params['obs']['scale'][normalizer.params['obs']['scale'] < 1e-8] = 1.0

    sample = dataset[0]
    obs_norm = normalizer.normalize(sample['obs'], 'obs')
    obs_denorm = normalizer.denormalize(obs_norm, 'obs')

    error = torch.abs(sample['obs'] - obs_denorm).max().item()
    print(f"✓ Normalization round-trip error: {error:.6f}")

    return normalizer


def main():
    """Run all tests."""
    print("="*60)
    print("Diffusion Policy Pipeline Test")
    print("="*60)

    try:
        # Test dataset
        dataset = test_dataset()

        # Test model
        model = test_model(dataset)

        # Test normalizer
        normalizer = test_normalizer(dataset)

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        print("\nYou can now run:")
        print("  python scripts/train.py --data_dir ../data --output_dir outputs")
        print("\nNote: Training on CPU will be slow. Use --device cuda:0 if available.")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
