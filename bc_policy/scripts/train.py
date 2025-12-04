
import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from bc_policy.models import BCPolicy, BCPolicyWithHistory
from bc_policy.utils import BCDataset, get_data_stats, Normalizer
def train_epoch(model, dataloader, normalizer, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = normalizer.normalize(batch)
        loss = model.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    avg_loss = total_loss / num_batches
    return avg_loss
def eval_epoch(model, dataloader, normalizer, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = normalizer.normalize(batch)
            loss = model.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
    avg_loss = total_loss / num_batches
    return avg_loss
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    print("\nCreating datasets...")
    train_dataset = BCDataset(
        dataset_root=config['data_root'],
        mode='train',
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        sample_stride=config['sample_stride'],
        noise_std=config.get('noise_std', 0.0),
    )
    val_dataset = BCDataset(
        dataset_root=config['data_root'],
        mode='val',
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        sample_stride=config['sample_stride'],
        noise_std=0.0,  
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print("\nComputing normalization statistics...")
    stats = get_data_stats(train_loader)
    normalizer = Normalizer(stats)
    normalizer_path = os.path.join(config['checkpoint_dir'], 'normalizer.pth')
    normalizer.save(normalizer_path)
    print("\nCreating model...")
    model = BCPolicy(
        proprio_dim=config['proprio_dim'],
        action_dim=config['action_dim'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        use_layer_norm=config['use_layer_norm'],
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    if config['optimizer'] == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    elif config['optimizer'] == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'] - config['warmup_epochs'],
        )
    elif config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif config['scheduler'] == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    print("\nStarting training...")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Warmup epochs: {config['warmup_epochs']}")
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        if epoch < config['warmup_epochs']:
            warmup_factor = (epoch + 1) / config['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate'] * warmup_factor
            print(f"Warmup: lr = {optimizer.param_groups[0]['lr']:.6f}")
        train_loss = train_epoch(
            model, train_loader, normalizer, optimizer, device,
            grad_clip=config['grad_clip'],
        )
        print(f"Train loss: {train_loss:.6f}")
        if (epoch + 1) % config['eval_interval'] == 0:
            val_loss = eval_epoch(model, val_loader, normalizer, device)
            print(f"Val loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config['checkpoint_dir'], 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, best_path)
                print(f"✓ Best checkpoint saved: {best_path}")
        if (epoch + 1) % config['save_interval'] == 0:
            epoch_path = os.path.join(config['checkpoint_dir'], f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss if (epoch + 1) % config['eval_interval'] == 0 else None,
                'best_val_loss': best_val_loss,
                'config': config,
            }, epoch_path)
            print(f"Checkpoint saved: {epoch_path}")
        last_path = os.path.join(config['checkpoint_dir'], 'last.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss if (epoch + 1) % config['eval_interval'] == 0 else None,
            'best_val_loss': best_val_loss,
            'config': config,
        }, last_path)
        if scheduler and epoch >= config['warmup_epochs']:
            scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print("="*60)
if __name__ == "__main__":
    main()
