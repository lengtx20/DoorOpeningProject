#!/usr/bin/env python3
"""
Minimal test script to isolate segfault issue.
"""

import sys
from pathlib import Path

print("Step 1: Importing basic libraries...")
try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")
    sys.exit(1)

print("\nStep 2: Importing torch...")
try:
    import torch
    print(f"✓ torch imported (version: {torch.__version__})")
except Exception as e:
    print(f"✗ torch failed: {e}")
    sys.exit(1)

print("\nStep 3: Importing PIL/Pillow...")
try:
    from PIL import Image
    print("✓ PIL imported")
except Exception as e:
    print(f"✗ PIL failed: {e}")
    sys.exit(1)

print("\nStep 4: Importing torchvision...")
try:
    from torchvision import transforms
    print("✓ torchvision imported")
except Exception as e:
    print(f"✗ torchvision failed: {e}")
    sys.exit(1)

print("\nStep 5: Testing ResNet model loading...")
try:
    import torchvision.models as models
    print("Loading ResNet18...")
    resnet = models.resnet18(pretrained=False)  # Don't download weights
    print("✓ ResNet18 loaded (pretrained=False)")
except Exception as e:
    print(f"✗ ResNet failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Testing dataset import...")
try:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from diffusion_policy.utils.dataset_image_door_opening import G1ImageDoorOpeningDataset
    print("✓ Dataset class imported")
except Exception as e:
    print(f"✗ Dataset import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 7: Testing dataset initialization...")
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

print("\nStep 8: Testing dataset indexing...")
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

print("\nStep 9: Testing model import...")
try:
    from diffusion_policy.models.diffusion_unet_image import DiffusionUNetImage
    print("✓ Model class imported")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 10: Testing model initialization...")
try:
    model = DiffusionUNetImage(
        dof_dim=sample['dof_state'].shape[-1],
        action_dim=sample['actions'].shape[-1],
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        image_backbone='resnet18',
    )
    print(f"✓ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! The issue might be in the training script itself.")
print("Try running the training script again.")

