#!/usr/bin/env python3
"""
Training script for diffusion policy on G1 door opening task.

Uses task-specific observations:
- Left/right hand positions in torso frame
- Base velocity commands
- Left hand grasp state
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

from diffusion_policy.models.diffusion_unet import DiffusionUNet
from diffusion_policy.utils.dataset_door_opening import G1DoorOpeningDataset
from diffusion_policy.utils.normalizer import LinearNormalizer


def train_epoch(model, dataloader, optimizer, device, normalizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        obs = batch['obs'].to(device)  # (B, obs_horizon, obs_dim)
        actions = batch['actions'].to(device)  # (B, pred_horizon, action_dim)

        # Normalize data
        B, T_obs, D_obs = obs.shape
        obs_flat = obs.reshape(B * T_obs, D_obs)
        obs_norm = normalizer.normalize(obs_flat, 'obs').reshape(B, T_obs, D_obs)

        _, T_act, D_act = actions.shape
        actions_flat = actions.reshape(B * T_act, D_act)
        actions_norm = normalizer.normalize(actions_flat, 'actions').reshape(B, T_act, D_act)

        # Compute loss
        loss_dict = model.compute_loss(obs_norm, actions_norm)
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
        obs = batch['obs'].to(device)
        actions = batch['actions'].to(device)

        # Normalize data
        B, T_obs, D_obs = obs.shape
        obs_flat = obs.reshape(B * T_obs, D_obs)
        obs_norm = normalizer.normalize(obs_flat, 'obs').reshape(B, T_obs, D_obs)

        _, T_act, D_act = actions.shape
        actions_flat = actions.reshape(B * T_act, D_act)
        actions_norm = normalizer.normalize(actions_flat, 'actions').reshape(B, T_act, D_act)

        # Compute loss
        loss_dict = model.compute_loss(obs_norm, actions_norm)
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

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"door_opening_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = G1DoorOpeningDataset(
        data_dir=args.data_dir,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        augment=args.augment,
    )

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Compute normalization statistics
    print("\nComputing normalization statistics...")
    stats = dataset.get_stats()
    normalizer = LinearNormalizer(mode='limits')

    # Manually set normalizer params from computed statistics
    normalizer.params['obs'] = {
        'min': stats['obs']['min'],
        'max': stats['obs']['max'],
        'scale': stats['obs']['max'] - stats['obs']['min'],
    }
    # Avoid division by zero
    normalizer.params['obs']['scale'][normalizer.params['obs']['scale'] < 1e-8] = 1.0

    normalizer.params['actions'] = {
        'min': stats['actions']['min'],
        'max': stats['actions']['max'],
        'scale': stats['actions']['max'] - stats['actions']['min'],
    }
    # Avoid division by zero
    normalizer.params['actions']['scale'][normalizer.params['actions']['scale'] < 1e-8] = 1.0

    # Save normalizer
    np.savez(
        output_dir / "normalizer.npz",
        obs_min=stats['obs']['min'],
        obs_max=stats['obs']['max'],
        actions_min=stats['actions']['min'],
        actions_max=stats['actions']['max'],
    )

    # Get dimensions
    sample = dataset[0]
    obs_dim = sample['obs'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    print(f"\nObservation dim: {obs_dim}")
    print(f"  - Left hand position (torso frame): 3")
    print(f"  - Right hand position (torso frame): 3")
    print(f"  - Base velocity command (vx, vy, yaw): 3")
    print(f"  - Left hand grasp state: 1")
    print(f"Action dim: {action_dim} (joint positions)")

    # Create model
    print("\nCreating diffusion policy model...")
    model = DiffusionUNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.num_diffusion_iters,
        down_dims=tuple(args.down_dims),
        obs_encoder_layers=tuple(args.obs_encoder_layers),
    ).to(device)

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
    parser = argparse.ArgumentParser(description="Train diffusion policy for G1 door opening")

    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing trajectory CSV files')
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
    parser.add_argument('--down_dims', type=int, nargs='+', default=[256, 512, 1024],
                        help='U-Net downsampling dimensions')
    parser.add_argument('--obs_encoder_layers', type=int, nargs='+', default=[256, 256],
                        help='Observation encoder MLP layers')

    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    main(args)
