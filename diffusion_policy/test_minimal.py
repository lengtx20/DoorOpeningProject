#!/usr/bin/env python3
"""
Minimal test - just import the training script to see where it crashes.
"""

import sys
print("Starting minimal test...")

print("\n1. Importing sys...")
import sys
print("✓")

print("\n2. Importing pathlib...")
from pathlib import Path
print("✓")

print("\n3. Importing torch...")
import torch
print(f"✓ torch {torch.__version__}")

print("\n4. Importing PIL...")
from PIL import Image
print("✓")

print("\n5. Setting up paths...")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print("✓")

print("\n6. Importing simple image encoder...")
try:
    from diffusion_policy.models.image_encoder_simple import SimpleImageEncoder
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n7. Importing dataset (no torchvision)...")
try:
    from diffusion_policy.utils.dataset_image_door_opening import G1ImageDoorOpeningDataset
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n8. Importing model (no torchvision)...")
try:
    from diffusion_policy.models.diffusion_unet_image import DiffusionUNetImage
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports successful!")
print("The issue might be in the training script itself.")
print("Try running: python test_no_torchvision.py")

