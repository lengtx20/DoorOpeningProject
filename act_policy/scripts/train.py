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
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from act_policy.models import ACTPolicy
from data.dataset import G1Dataset, get_data_stats, Normalizer

def train_epoch(model, dataloader, normalizer, optimizer, device, kl_weight=1.0, grad_clip=None, writer=None, epoch=0, stage_dropout=0.0):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    epoch_mu_values = []
    epoch_logvar_values = []

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        raw_stage = batch['stage']
        
        batch = normalizer.normalize(batch)
        batch['stage'] = raw_stage

        if stage_dropout > 0.0:
            drop_mask = torch.rand(batch['stage'].shape[0], device=device) < stage_dropout
            batch['stage'][drop_mask] = 0.0

        optimizer.zero_grad()
        losses = model.compute_loss(batch, kl_weight=kl_weight)
        loss = losses['loss']
        recon_loss = losses['recon_loss']
        kl_loss = losses['kl_loss']

        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item() if model.use_vae else 0.0
        })

        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/recon_loss_step', recon_loss.item(), global_step)
            if model.use_vae:
                writer.add_scalar('train/kl_loss_step', kl_loss.item(), global_step)

                if 'mu' in losses and 'logvar' in losses:
                    mu = losses['mu']
                    logvar = losses['logvar']
                    variance = torch.exp(logvar)

                    writer.add_scalar('train/mu', mu.mean().item(), global_step)
                    writer.add_scalar('train/variance', variance.mean().item(), global_step)

                    epoch_mu_values.append(mu.detach().cpu())
                    epoch_logvar_values.append(logvar.detach().cpu())

    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches

    if writer is not None and model.use_vae and len(epoch_mu_values) > 0:
        all_mu = torch.cat(epoch_mu_values, dim=0).flatten()
        all_logvar = torch.cat(epoch_logvar_values, dim=0).flatten()
        writer.add_histogram('train/mu_histogram', all_mu, epoch)
        writer.add_histogram('train/logvar_histogram', all_logvar, epoch)

    return avg_loss, avg_recon_loss, avg_kl_loss



def eval_epoch(model, dataloader, normalizer, device, kl_weight=1.0):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = normalizer.normalize(batch)

            losses = model.compute_loss(batch, kl_weight=kl_weight)
            loss = losses['loss']
            recon_loss = losses['recon_loss']
            kl_loss = losses['kl_loss']

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item() if model.use_vae else 0.0
            })

    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches

    return avg_loss, avg_recon_loss, avg_kl_loss


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

    train_dataset = G1Dataset(
        dataset_root=config['data_root'],
        mode='train',
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        use_proprio=True,
        use_image=config.get('use_image', False),
        image_resize_size=config.get('image_resize_size', None),
        sample_stride=config['sample_stride'],
        noise_std=config.get('noise_std', 0.0),
        temporal_dropout=config.get('temporal_dropout', 0.0),
        action_noise_std=config.get('action_noise_std', 0.0),
        smooth_window=config.get('smooth_window', 0),
    )

    val_dataset = G1Dataset(
        dataset_root=config['data_root'],
        mode='val',
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        use_proprio=True,
        use_image=config.get('use_image', False),
        image_resize_size=config.get('image_resize_size', None),
        sample_stride=config['sample_stride'],
        noise_std=0.0,
        temporal_dropout=0.0,
        action_noise_std=0.0,
        smooth_window=config.get('smooth_window', 0),
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

    normalizer_path = os.path.join(config['checkpoint_dir'], 'normalizer.pth')
    if os.path.exists(normalizer_path):
        print(f"\nLoading existing normalization statistics from {normalizer_path}")
        normalizer = Normalizer.load(normalizer_path)
    else:
        print("\nComputing normalization statistics...")
        stats = get_data_stats(train_loader)
        normalizer = Normalizer(stats)
        normalizer.save(normalizer_path)

    print("\nCreating ACT model...")
    model = ACTPolicy(
        proprio_dim=config['proprio_dim'],
        stage_dim=config['stage_dim'],
        action_dim=config['action_dim'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        hidden_dim=config['hidden_dim'],
        nheads=config['nheads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
        use_vae=config['use_vae'],
        use_image=config.get('use_image', False),
        vision_feature_dim=config.get('vision_feature_dim', 512),
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

    # Create TensorBoard writer
    tensorboard_dir = os.path.join(config['checkpoint_dir'], 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"\nTensorBoard logs will be saved to: {tensorboard_dir}")
    print("Run: tensorboard --logdir {} to view".format(tensorboard_dir))

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

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)

        # Get current KL weight (can schedule this)
        kl_weight = config['kl_weight']
        writer.add_scalar('train/kl_weight', kl_weight, epoch)

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, normalizer, optimizer, device,
            kl_weight=kl_weight,
            grad_clip=config['grad_clip'],
            writer=writer,
            epoch=epoch,
            stage_dropout=config.get('stage_dropout', 0.0)
        )
        print(f"Train - Loss: {train_loss:.6f}, Recon: {train_recon:.6f},  KL: {train_kl:.6f}")

        # Log epoch-level train loss
        writer.add_scalar('train/loss_epoch', train_loss, epoch)
        writer.add_scalar('train/recon_loss_epoch', train_recon, epoch)
        writer.add_scalar('train/kl_loss_epoch', train_kl, epoch)

        if (epoch + 1) % config['eval_interval'] == 0:
            val_loss, val_recon, val_kl= eval_epoch(model, val_loader, normalizer, device, kl_weight=kl_weight)
            print(f"Val - Loss: {val_loss:.6f}, Recon: {val_recon:.6f}, KL: {val_kl:.6f}")

            # Log validation loss
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/recon_loss', val_recon, epoch)
            writer.add_scalar('val/kl_loss', val_kl, epoch)

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

    # Close TensorBoard writer
    writer.close()

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
