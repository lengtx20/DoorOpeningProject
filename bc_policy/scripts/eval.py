
import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from bc_policy.models import BCPolicy
from bc_policy.utils import BCDataset, Normalizer
def evaluate(model, dataloader, normalizer, device):
    model.eval()
    all_losses = []
    all_pred_actions = []
    all_gt_actions = []
    per_dim_errors = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_norm = normalizer.normalize(batch)
            loss = model.compute_loss(batch_norm)
            all_losses.append(loss.item())
            proprio_norm = batch_norm['proprio']
            action_pred_norm = model(proprio_norm)
            action_pred = normalizer.denormalize(action_pred_norm, 'action')
            action_gt = batch['action']
            per_dim_error = (action_pred - action_gt).abs().mean(dim=[0, 1])  
            per_dim_errors.append(per_dim_error.cpu())
            all_pred_actions.append(action_pred.cpu().numpy())
            all_gt_actions.append(action_gt.cpu().numpy())
    avg_loss = np.mean(all_losses)
    per_dim_errors = torch.stack(per_dim_errors).mean(dim=0).numpy()
    all_pred_actions = np.concatenate(all_pred_actions, axis=0)  
    all_gt_actions = np.concatenate(all_gt_actions, axis=0)
    mae = np.abs(all_pred_actions - all_gt_actions).mean()
    mse = np.square(all_pred_actions - all_gt_actions).mean()
    action_names = [
        'left_hand_x', 'left_hand_y', 'left_hand_z',
        'right_hand_x', 'right_hand_y', 'right_hand_z',
        'vel_x', 'vel_y', 'yaw_speed', 'grasp'
    ]
    metrics = {
        'avg_loss': avg_loss,
        'mae': mae,
        'mse': mse,
        'per_dim_mae': {action_names[i]: per_dim_errors[i] for i in range(len(action_names))},
        'pred_stats': {
            'mean': all_pred_actions.mean(axis=(0, 1)),
            'std': all_pred_actions.std(axis=(0, 1)),
            'min': all_pred_actions.min(axis=(0, 1)),
            'max': all_pred_actions.max(axis=(0, 1)),
        },
        'gt_stats': {
            'mean': all_gt_actions.mean(axis=(0, 1)),
            'std': all_gt_actions.std(axis=(0, 1)),
            'min': all_gt_actions.min(axis=(0, 1)),
            'max': all_gt_actions.max(axis=(0, 1)),
        }
    }
    return metrics
def print_metrics(metrics):
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Average Loss (normalized): {metrics['avg_loss']:.6f}")
    print(f"  MAE (denormalized):        {metrics['mae']:.6f}")
    print(f"  MSE (denormalized):        {metrics['mse']:.6f}")
    print(f"  RMSE (denormalized):       {np.sqrt(metrics['mse']):.6f}")
    print(f"\nPer-Dimension MAE:")
    for name, error in metrics['per_dim_mae'].items():
        print(f"  {name:15s}: {error:.6f}")
    print(f"\nPredicted Action Statistics:")
    print(f"  {'Dimension':<15s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    action_names = list(metrics['per_dim_mae'].keys())
    for i, name in enumerate(action_names):
        print(f"  {name:<15s} "
              f"{metrics['pred_stats']['mean'][i]:10.4f} "
              f"{metrics['pred_stats']['std'][i]:10.4f} "
              f"{metrics['pred_stats']['min'][i]:10.4f} "
              f"{metrics['pred_stats']['max'][i]:10.4f}")
    print(f"\nGround Truth Action Statistics:")
    print(f"  {'Dimension':<15s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, name in enumerate(action_names):
        print(f"  {name:<15s} "
              f"{metrics['gt_stats']['mean'][i]:10.4f} "
              f"{metrics['gt_stats']['std'][i]:10.4f} "
              f"{metrics['gt_stats']['min'][i]:10.4f} "
              f"{metrics['gt_stats']['max'][i]:10.4f}")
    print("\n" + "="*60)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', type=str, default=None, help='Override data root')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    if args.data_root:
        config['data_root'] = args.data_root
    print("\nCheckpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    checkpoint_dir = os.path.dirname(args.checkpoint)
    normalizer_path = os.path.join(checkpoint_dir, 'normalizer.pth')
    if not os.path.exists(normalizer_path):
        print(f"Warning: Normalizer not found at {normalizer_path}")
        print("Attempting to use normalizer from checkpoint...")
        if 'stats' in checkpoint:
            normalizer = Normalizer(checkpoint['stats'])
        else:
            raise ValueError("No normalizer found in checkpoint or checkpoint directory")
    else:
        normalizer = Normalizer.load(normalizer_path)
    print(f"Normalizer loaded successfully")
    print("\nCreating validation dataset...")
    val_dataset = BCDataset(
        dataset_root=config['data_root'],
        mode='val',
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        sample_stride=config['sample_stride'],
        noise_std=0.0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Val dataset: {len(val_dataset)} samples")
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
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully")
    print("\nStarting evaluation...")
    metrics = evaluate(model, val_loader, normalizer, device)
    print_metrics(metrics)
    metrics_path = os.path.join(checkpoint_dir, 'eval_metrics.npy')
    np.save(metrics_path, metrics)
    print(f"\nMetrics saved to: {metrics_path}")
if __name__ == "__main__":
    main()
