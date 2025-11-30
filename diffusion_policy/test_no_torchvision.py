#!/usr/bin/env python3
"""
Test script that completely avoids torchvision.
"""

import sys
from pathlib import Path

print("Testing without torchvision...")

print("\nStep 1: Importing torch...")
import torch
print(f"✓ torch imported (version: {torch.__version__})")

print("\nStep 2: Importing PIL...")
from PIL import Image
print("✓ PIL imported")

print("\nStep 3: Testing dataset import (should not import torchvision)...")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from diffusion_policy.utils.dataset_image_door_opening import G1ImageDoorOpeningDataset
    print("✓ Dataset class imported (no torchvision)")
except Exception as e:
    print(f"✗ Dataset import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Testing dataset initialization...")
try:
    dataset = G1ImageDoorOpeningDataset(
        data_dir="../data",
        image_dir="../data/images",
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        include_velocities=False,
    )
    print(f"✓ Dataset initialized: {len(dataset)} samples")
except Exception as e:
    print(f"✗ Dataset initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Testing dataset indexing...")
try:
    sample = dataset[0]
    print(f"✓ Sample retrieved:")
    print(f"  - Images shape: {sample['images'].shape}")
    print(f"  - DOF state shape: {sample['dof_state'].shape}")
    print(f"  - Actions shape: {sample['actions'].shape}")
except Exception as e:
    print(f"✗ Dataset indexing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Testing simple image encoder...")
try:
    from diffusion_policy.models.image_encoder_simple import SimpleImageEncoder
    encoder = SimpleImageEncoder(image_size=(224, 224), feature_dim=256)
    print("✓ Simple image encoder created")
    
    # Test forward pass
    test_img = torch.randn(1, 3, 224, 224)
    features = encoder(test_img)
    print(f"✓ Encoder forward pass: {features.shape}")
except Exception as e:
    print(f"✗ Simple encoder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 7: Testing model import...")
try:
    from diffusion_policy.models.diffusion_unet_image import DiffusionUNetImage
    print("✓ Model class imported")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 8: Testing model initialization with simple encoder...")
try:
    model = DiffusionUNetImage(
        dof_dim=sample['dof_state'].shape[-1],
        action_dim=sample['actions'].shape[-1],
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        image_backbone='simple',
        use_simple_encoder=True,
    )
    print(f"✓ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed without torchvision!")
print("You can now run training with:")
print("  python scripts/train_image_door_opening.py --data_dir ../data --image_dir ../data/images --image_backbone simple --use_simple_encoder")

