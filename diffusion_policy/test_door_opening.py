#!/usr/bin/env python3
"""
Test script for door opening dataset and model.

Verifies that the task-specific observations are correctly computed.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from diffusion_policy.models.diffusion_unet import DiffusionUNet
from diffusion_policy.utils.dataset_door_opening import G1DoorOpeningDataset
from diffusion_policy.utils.normalizer import LinearNormalizer


def test_dataset():
    """Test dataset loading with door opening observations."""
    print("Testing door opening dataset...")
    dataset = G1DoorOpeningDataset(
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

    # Verify observation dimension
    assert sample['obs'].shape[-1] == 10, f"Expected obs_dim=10, got {sample['obs'].shape[-1]}"
    print(f"✓ Observation dimension correct: 10")

    # Print sample observation to verify content
    print(f"\nSample observation (first timestep):")
    obs_first = sample['obs'][0].numpy()
    print(f"  Left hand (torso): [{obs_first[0]:.3f}, {obs_first[1]:.3f}, {obs_first[2]:.3f}]")
    print(f"  Right hand (torso): [{obs_first[3]:.3f}, {obs_first[4]:.3f}, {obs_first[5]:.3f}]")
    print(f"  Base vel cmd: [{obs_first[6]:.3f}, {obs_first[7]:.3f}, {obs_first[8]:.3f}]")
    print(f"  Grasp state: {obs_first[9]:.3f}")

    return dataset


def test_model(dataset):
    """Test model creation and forward pass."""
    print("\nTesting model with 10-dim observations...")

    sample = dataset[0]
    obs_dim = sample['obs'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    assert obs_dim == 10, f"Expected obs_dim=10, got {obs_dim}"

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
    print(f"  Input: {obs_dim}-dim observations (task-specific)")
    print(f"  Output: {action_dim}-dim actions (joint positions)")

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

    # Print normalization stats
    print(f"\nObservation normalization ranges:")
    print(f"  Left hand x: [{stats['obs']['min'][0]:.3f}, {stats['obs']['max'][0]:.3f}]")
    print(f"  Left hand y: [{stats['obs']['min'][1]:.3f}, {stats['obs']['max'][1]:.3f}]")
    print(f"  Left hand z: [{stats['obs']['min'][2]:.3f}, {stats['obs']['max'][2]:.3f}]")
    print(f"  Base vel x: [{stats['obs']['min'][6]:.3f}, {stats['obs']['max'][6]:.3f}]")
    print(f"  Base vel y: [{stats['obs']['min'][7]:.3f}, {stats['obs']['max'][7]:.3f}]")
    print(f"  Yaw rate: [{stats['obs']['min'][8]:.3f}, {stats['obs']['max'][8]:.3f}]")
    print(f"  Grasp: [{stats['obs']['min'][9]:.3f}, {stats['obs']['max'][9]:.3f}]")

    return normalizer


def main():
    """Run all tests."""
    print("="*60)
    print("Door Opening Dataset & Model Test")
    print("="*60)
    print("\nObservation Space (10-dim):")
    print("  1-3: Left hand position in torso frame (x, y, z)")
    print("  4-6: Right hand position in torso frame (x, y, z)")
    print("  7-9: Base velocity command (vx, vy, yaw_rate)")
    print("  10: Left hand grasp state")
    print("")

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
        print("\nYou can now train with:")
        print("  python scripts/train_door_opening.py --data_dir ../data --output_dir outputs")
        print("\nObservation breakdown:")
        print("  - Hand positions: Relative to torso (better for manipulation)")
        print("  - Velocity commands: Direct control signals")
        print("  - Grasp state: Binary/continuous grasping indicator")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
