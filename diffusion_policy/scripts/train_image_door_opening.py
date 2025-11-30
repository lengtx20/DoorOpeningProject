#!/usr/bin/env python3
"""
Training script for diffusion policy on G1 door opening task with images + 29 DOF.

Input:
- Images (visual observations)
- 29 DOF state (joint positions + optionally velocities)

Output:
- 10-dim actions (left hand pos, right hand pos, base vel cmd, grasp state)
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffusion_policy.models.diffusion_unet_image import DiffusionUNetImage
from diffusion_policy.utils.dataset_image_door_opening import G1ImageDoorOpeningDataset
from diffusion_policy.utils.normalizer import LinearNormalizer


def train_epoch(model, dataloader, optimizer, device, normalizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['images'].to(device)  # (B, obs_horizon, C, H, W)
        dof_state = batch['dof_state'].to(device)  # (B, obs_horizon, dof_dim)
        actions = batch['actions'].to(device)  # (B, pred_horizon, action_dim=10)

        # Normalize DOF state and actions
        B, T_obs, D_dof = dof_state.shape
        dof_flat = dof_state.reshape(B * T_obs, D_dof)
        dof_norm = normalizer.normalize(dof_flat, 'dof_state').reshape(B, T_obs, D_dof)

        _, T_act, D_act = actions.shape
        actions_flat = actions.reshape(B * T_act, D_act)
        actions_norm = normalizer.normalize(actions_flat, 'actions').reshape(B, T_act, D_act)

        # Images are already normalized by ImageNet stats in dataset

        # Compute loss
        loss_dict = model.compute_loss(images, dof_norm, actions_norm)
        loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, normalizer):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch['images'].to(device)
        dof_state = batch['dof_state'].to(device)
        actions = batch['actions'].to(device)

        # Normalize DOF state and actions
        B, T_obs, D_dof = dof_state.shape
        dof_flat = dof_state.reshape(B * T_obs, D_dof)
        dof_norm = normalizer.normalize(dof_flat, 'dof_state').reshape(B, T_obs, D_dof)

        _, T_act, D_act = actions.shape
        actions_flat = actions.reshape(B * T_act, D_act)
        actions_norm = normalizer.normalize(actions_flat, 'actions').reshape(B, T_act, D_act)

        # Compute loss
        loss_dict = model.compute_loss(images, dof_norm, actions_norm)
        loss = loss_dict['loss']

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    """Main training function."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Fix data directory path (handle relative paths)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        # If relative, try relative to project root first, then current dir
        project_root = Path(__file__).parent.parent.parent
        if (project_root / data_dir).exists():
            data_dir = project_root / data_dir
        elif not data_dir.exists():
            # Try relative to current working directory
            data_dir = Path.cwd() / data_dir
    
    args.data_dir = str(data_dir.resolve())
    
    # Fix image directory path
    if args.image_dir is None:
        image_dir = data_dir / "images"
    else:
        image_dir = Path(args.image_dir)
        if not image_dir.is_absolute():
            image_dir = data_dir.parent / image_dir if (data_dir.parent / image_dir).exists() else Path.cwd() / image_dir
    
    args.image_dir = str(image_dir.resolve())
    
    print(f"Data directory: {args.data_dir}")
    print(f"Image directory: {args.image_dir}")
    
    if not Path(args.data_dir).exists():
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"image_door_opening_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    try:
        dataset = G1ImageDoorOpeningDataset(
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            action_horizon=args.action_horizon,
            include_velocities=args.include_velocities,
            image_size=tuple(args.image_size),
            augment=args.augment,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    # On macOS, num_workers > 0 can cause segfaults, so use 0 if on macOS
    import platform
    if platform.system() == 'Darwin':  # macOS
        num_workers = 0
        pin_memory = False
    else:
        num_workers = args.num_workers
        pin_memory = args.device.startswith('cuda')  # Only pin memory for GPU
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Compute normalization statistics
    print("\nComputing normalization statistics...")
    stats = dataset.get_stats()
    normalizer = LinearNormalizer(mode='limits')

    # Set normalizer params for DOF state
    normalizer.params['dof_state'] = {
        'min': stats['dof_state']['min'],
        'max': stats['dof_state']['max'],
        'scale': stats['dof_state']['max'] - stats['dof_state']['min'],
    }
    normalizer.params['dof_state']['scale'][normalizer.params['dof_state']['scale'] < 1e-8] = 1.0

    # Set normalizer params for actions
    normalizer.params['actions'] = {
        'min': stats['actions']['min'],
        'max': stats['actions']['max'],
        'scale': stats['actions']['max'] - stats['actions']['min'],
    }
    normalizer.params['actions']['scale'][normalizer.params['actions']['scale'] < 1e-8] = 1.0

    # Save normalizer
    np.savez(
        output_dir / "normalizer.npz",
        dof_state_min=stats['dof_state']['min'],
        dof_state_max=stats['dof_state']['max'],
        actions_min=stats['actions']['min'],
        actions_max=stats['actions']['max'],
    )

    # Get dimensions
    sample = dataset[0]
    dof_dim = sample['dof_state'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    print(f"\nDOF state dim: {dof_dim} ({'positions + velocities' if args.include_velocities else 'positions only'})")
    print(f"Action dim: {action_dim}")
    print(f"  - Left hand position (torso frame): 3")
    print(f"  - Right hand position (torso frame): 3")
    print(f"  - Base velocity command (vx, vy, yaw): 3")
    print(f"  - Left hand grasp state: 1")

    # Create model
    print("\nCreating diffusion policy model with image encoder...")
    print(f"  - DOF dim: {dof_dim}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Image backbone: {args.image_backbone}")
    print(f"  - Image pretrained: {args.image_pretrained}")
    
    # On macOS, force simple encoder unless explicitly overridden
    import platform
    is_macos = platform.system() == 'Darwin'
    if is_macos and args.image_backbone != 'simple' and not args.use_simple_encoder:
        import os
        if os.environ.get('FORCE_TORCHVISION', '0') != '1':
            print(f"  ⚠️  macOS detected: Overriding to simple encoder to avoid segfaults")
            print(f"     (Set FORCE_TORCHVISION=1 to try ResNet, but it may segfault)")
            args.image_backbone = 'simple'
            args.use_simple_encoder = True
    
    try:
        model = DiffusionUNetImage(
        dof_dim=dof_dim,
        action_dim=action_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.num_diffusion_iters,
        image_size=tuple(args.image_size),
        image_feature_dim=args.image_feature_dim,
        dof_encoder_layers=tuple(args.dof_encoder_layers),
        down_dims=tuple(args.down_dims),
        image_backbone=args.image_backbone,
        image_pretrained=args.image_pretrained,
        use_simple_encoder=args.use_simple_encoder,
        )
        print("  ✓ Model created")
        model = model.to(device)
        print(f"  ✓ Model moved to device: {device}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, normalizer, epoch)

        # Validate
        val_loss = validate(model, val_loader, device, normalizer)

        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_model_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 60)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion policy with images + 29 DOF for G1 door opening")

    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing trajectory CSV files')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images (if None, uses data_dir/images)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for models and logs')

    # Model architecture
    parser.add_argument('--obs_horizon', type=int, default=2,
                        help='Number of observation frames')
    parser.add_argument('--pred_horizon', type=int, default=16,
                        help='Number of action frames to predict')
    parser.add_argument('--action_horizon', type=int, default=8,
                        help='Number of action frames to execute')
    parser.add_argument('--num_diffusion_iters', type=int, default=100,
                        help='Number of diffusion denoising iterations')
    parser.add_argument('--include_velocities', action='store_true',
                        help='Include joint velocities in DOF state (58 dims vs 29)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Image size (height, width)')
    parser.add_argument('--image_feature_dim', type=int, default=256,
                        help='Feature dimension from image encoder')
    parser.add_argument('--image_backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'simple'],
                        help='Image encoder backbone (default: resnet18, falls back to simple CNN if torchvision fails)')
    parser.add_argument('--image_pretrained', action='store_true',
                        help='Use pretrained ImageNet weights (may cause segfault on some systems)')
    parser.add_argument('--use_simple_encoder', action='store_true',
                        help='Force use of simple CNN encoder (avoids torchvision entirely)')
    parser.add_argument('--dof_encoder_layers', type=int, nargs='+', default=[256, 256],
                        help='DOF state encoder MLP layers')
    parser.add_argument('--down_dims', type=int, nargs='+', default=[256, 512, 1024],
                        help='U-Net downsampling dimensions')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (smaller for images)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')

    # System
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0 or cpu)')
    import platform
    default_num_workers = 0 if platform.system() == 'Darwin' else 4
    parser.add_argument('--num_workers', type=int, default=default_num_workers,
                        help='Number of data loader workers (0 on macOS to avoid segfaults)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    main(args)

