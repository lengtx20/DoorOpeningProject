import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as T
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


from act_policy.models import ACTPolicy
from data.dataset import G1Dataset, Normalizer, compute_stage_from_frame_idx


def stage_to_onehot(stage_label: int, num_stages: int = 5) -> torch.Tensor:
    onehot = torch.zeros(num_stages)
    if 0 <= stage_label < num_stages:
        onehot[stage_label] = 1.0
    return onehot
def compute_overlap_counts(all_predictions, traj_length, pred_horizon):


    overlap = np.zeros(traj_length, dtype=int)

    for (start_idx, pred_actions) in all_predictions:
        end_idx = min(start_idx + pred_horizon, traj_length)
        overlap[start_idx:end_idx] += 1

    return overlap


def visualize_sliding_windows(
    model, normalizer, dataset, config, device,
    trajectory_idx=0, save_dir='visualizations'
):
    os.makedirs(save_dir, exist_ok=True)

    if trajectory_idx >= len(dataset.episode_ids):
        print(f"Warning: trajectory_idx {trajectory_idx} out of range (max: {len(dataset.episode_ids)-1})")
        return

    ep_id = dataset.episode_ids[trajectory_idx]
    print(f"\nVisualizing sliding windows for trajectory: {ep_id}")

    ep_data = dataset.data_cache[ep_id]
    traj_length = ep_data['length']

    gt_actions = ep_data['action_traj']  
    gt_proprio_pos = ep_data['proprio_pos_traj']  
    gt_proprio_vel = ep_data['proprio_vel_traj']  

    print(f"Trajectory length: {traj_length} timesteps")

    use_image = config.get('use_image', False)
    image_transform = None
    if use_image:
        image_transform = T.Resize((224, 224))

    gt_images = None
    if use_image:
        print("Model uses images - loading image data from trajectory...")
        ep_path = os.path.join(dataset.root, ep_id)
        image_files = sorted([f for f in os.listdir(ep_path) if f.endswith('.npy') and f != 'log_dict.npy'])
        if len(image_files) > 0:
            gt_images = []
            for img_file in image_files:
                img_path = os.path.join(ep_path, img_file)
                img_np = np.load(img_path)  
                gt_images.append(img_np)
            gt_images = np.array(gt_images) 
        else:
            print("Warning: Model expects images but none found in trajectory. Using zero images.")
            img_h = config.get('image_resize_size', 224)
            img_w = config.get('image_resize_size', 224)
            gt_images = np.zeros((traj_length, img_h, img_w, 3), dtype=np.uint8) 

    obs_horizon = config['obs_horizon']
    pred_horizon = config['pred_horizon']

    all_predictions = [] 
    stages = []

    print("Running sliding window predictions...")
    for t in tqdm(range(0, traj_length, 1)):
        obs_start = t - obs_horizon + 1
        obs_end = t + 1

        if obs_start < 0:
            continue
        
        # [obs_horizon, 29]
        proprio_pos_seq = gt_proprio_pos[obs_start:obs_end]  
        proprio_vel_seq = gt_proprio_vel[obs_start:obs_end]  

        image_seq = None
        if use_image and gt_images is not None:
            imgs = []
            for i in range(obs_start, obs_end):
                img_np = gt_images[i] 
                img = torch.from_numpy(img_np).float() / 255.0 
                img = img.permute(2, 0, 1) 
                if image_transform is not None:
                    img = image_transform(img)
                imgs.append(img)
            image_seq = torch.stack(imgs).unsqueeze(0).to(device)

        stage_label = compute_stage_from_frame_idx(t)
        stage_onehot = stage_to_onehot(stage_label).unsqueeze(0).to(device)  

        proprio_pos_tensor = torch.from_numpy(proprio_pos_seq).float().unsqueeze(0).to(device) 
        proprio_vel_tensor = torch.from_numpy(proprio_vel_seq).float().unsqueeze(0).to(device) 

    
        proprio_tensor = torch.cat([proprio_pos_tensor, proprio_vel_tensor], dim=-1) 

     
        batch = {
            'proprio': proprio_tensor,  
            'stage': stage_onehot,
        }
        if image_seq is not None:
            batch['image'] = image_seq

        
        batch_norm = normalizer.normalize(batch)

        
        with torch.no_grad():
            model.eval()
            output = model(batch_norm, inference=True)
            action_pred_norm = output['action'] 
            action_pred = normalizer.denormalize(action_pred_norm, 'action')
            action_pred_np = action_pred[0].cpu().numpy() 

      
        pred_start_idx = t + 1
        all_predictions.append((pred_start_idx, action_pred_np))

   
    for t in range(traj_length):
        stages.append(compute_stage_from_frame_idx(t))
    stages = np.array(stages)

    print(f"Generated {len(all_predictions)} sliding window predictions")

    overlap = compute_overlap_counts(all_predictions, traj_length, pred_horizon)

    print("\n=== Overlap Statistics ===")
    print(f"Min overlap: {overlap.min()}")
    print(f"Max overlap: {overlap.max()}")
    print(f"Mean overlap: {overlap.mean():.2f}")
    print(f"Expected ~{pred_horizon} for most timesteps")



    action_names = [
        'Left Hand X', 'Left Hand Y', 'Left Hand Z',
        'Right Hand X', 'Right Hand Y', 'Right Hand Z',
        'Vel X', 'Vel Y', 'Yaw Speed'
    ]

    stage_colors = [
        "#FF6B6B",  # Vibrant Red
        "#6BCB77",  # Fresh Green
        "#4D96FF",  # Bright Blue
        "#FFD93D",  # Warm Yellow
        "#845EC2",  # Strong Purple
    ]


    stage_names = ['Approach', 'Grasp Prep', 'Contact', 'Door Opening', 'Full Open']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Sliding Window Predictions: Trajectory {ep_id}', fontsize=16, fontweight='bold')

    for dim in range(9):
        row = dim % 3
        col = dim // 3
        ax = axes[row, col]

        current_stage = stages[0]
        stage_start = 0

        for t in range(1, traj_length):
            if stages[t] != current_stage:
                ax.axvspan(stage_start, t, facecolor=stage_colors[current_stage], alpha=0.2, linewidth=0)
                stage_start = t
                current_stage = stages[t]
        
        ax.axvspan(stage_start, traj_length, facecolor=stage_colors[current_stage], alpha=0.2, linewidth=0, label='_nolegend_')

        if dim == 0:
            handles = []
            for i, name in enumerate(stage_names):
                handles.append(Rectangle((0, 0), 1, 1, fc=stage_colors[i], alpha=0.5))
            ax.legend(handles, stage_names, title='Stage', loc='upper left', fontsize=8)


        timesteps = np.arange(traj_length)
        ax.plot(timesteps, gt_actions[:, dim], color='black', linestyle='-', label='GT', linewidth=2, alpha=0.9)
        

        prediction_plotted = False
        
        for i, (pred_start_idx, pred_actions) in enumerate(all_predictions):
            pred_end_idx = pred_start_idx + pred_horizon
            plot_end_idx = min(pred_end_idx, traj_length)
            pred_window_timesteps = np.arange(pred_start_idx, plot_end_idx)
            plot_actions = pred_actions[:(plot_end_idx - pred_start_idx), dim]
            
            if not prediction_plotted:
                ax.plot(pred_window_timesteps, plot_actions, color='red', linestyle='--', linewidth=1, alpha=0.4, label='Prediction Window')
                prediction_plotted = True
            else:
                ax.plot(pred_window_timesteps, plot_actions, color='red', linestyle='--', linewidth=1, alpha=0.4)


        first_step_preds = []
        gt_targets = []
        
        for pred_start_idx, pred_actions in all_predictions:
            if pred_start_idx < traj_length:
                first_step_preds.append(pred_actions[0, dim])
                gt_targets.append(gt_actions[pred_start_idx, dim])
        
        if len(first_step_preds) > 0:
            first_step_preds_np = np.array(first_step_preds)
            gt_targets_np = np.array(gt_targets)
            mae = np.abs(first_step_preds_np - gt_targets_np).mean()
            ax.set_title(f'{action_names[dim]} (1-Step MAE: {mae:.4f})', fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'{action_names[dim]}', fontsize=10, fontweight='bold')
            
        ax.set_xlabel('Timestep', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        
        action_lines = []
        action_labels = []
        lines, labels = ax.get_legend_handles_labels()
        for l, lbl in zip(lines, labels):
            if lbl in ('GT', 'Prediction Window'):
                 action_lines.append(l)
                 action_labels.append(lbl)
                 
        ax.legend(action_lines, action_labels, loc='lower right', fontsize=8)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, traj_length)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    summary_path = os.path.join(save_dir, f'traj_{ep_id}_sliding_windows_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig) 

    print(f"\nSummary plot saved: {summary_path}")
    print("---")

def visualize_trajectories(model, dataloader, normalizer, device, num_trajectories=5, save_dir='visualizations'):
   
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    action_names = [
        'left_hand_x', 'left_hand_y', 'left_hand_z',
        'right_hand_x', 'right_hand_y', 'right_hand_z',
        'vel_x', 'vel_y', 'yaw_speed'
    ]

    trajectories_collected = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting trajectories")):
            if trajectories_collected >= num_trajectories:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            batch_norm = normalizer.normalize(batch)

            
            output = model(batch_norm, inference=False)
            action_pred_norm = output['action']

           
            action_pred = normalizer.denormalize(action_pred_norm, 'action')
            action_gt = batch['action']

        
            batch_size = action_pred.shape[0]
            for sample_idx in range(batch_size):
                if trajectories_collected >= num_trajectories:
                    break

                pred = action_pred[sample_idx].cpu().numpy()  
                gt = action_gt[sample_idx].cpu().numpy()



                fig, axes = plt.subplots(3, 3, figsize=(18, 12))
                fig.suptitle(f'Trajectory {trajectories_collected + 1}: Predicted vs Ground Truth Actions (Window)',
                           fontsize=14, fontweight='bold')

                for dim in range(9):
                    row = dim % 3
                    col = dim // 3
                    ax = axes[row, col]
                    timesteps = np.arange(len(pred))

                    ax.plot(timesteps, gt[:, dim], 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
                    ax.plot(timesteps, pred[:, dim], 'r--', label='Predicted', linewidth=2, alpha=0.7)

                    ax.set_title(action_names[dim], fontsize=10, fontweight='bold')
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('Value')
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)


                    mae = np.abs(pred[:, dim] - gt[:, dim]).mean()
                    ax.text(0.02, 0.98, f'MAE: {mae:.4f}',
                          transform=ax.transAxes, fontsize=8,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()


                save_path = os.path.join(save_dir, f'trajectory_window_{trajectories_collected + 1:03d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"Saved: {save_path}")

                trajectories_collected += 1

    print(f"\nVisualized {trajectories_collected} trajectories")
    print(f"Plots saved to: {save_dir}")


def plot_action_distributions(model, dataloader, normalizer, device, save_dir='visualizations'):
   
    model.eval()

    all_pred = []
    all_gt = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting data for distributions"):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_norm = normalizer.normalize(batch)

            output = model(batch_norm, inference=False)
            action_pred_norm = output['action']

            action_pred = normalizer.denormalize(action_pred_norm, 'action')
            action_gt = batch['action']

            all_pred.append(action_pred.cpu().numpy())
            all_gt.append(action_gt.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)  
    all_gt = np.concatenate(all_gt, axis=0)



    all_pred = all_pred.reshape(-1, 9)
    all_gt = all_gt.reshape(-1, 9)

    action_names = [
        'left_hand_x', 'left_hand_y', 'left_hand_z',
        'right_hand_x', 'right_hand_y', 'right_hand_z',
        'vel_x', 'vel_y', 'yaw_speed'
    ]


    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Action Distributions: Predicted vs Ground Truth', fontsize=14, fontweight='bold')

    for dim in range(9):
        row = dim % 3
        col = dim // 3
        ax = axes[row, col]

        ax.hist(all_gt[:, dim], bins=50, alpha=0.5, label='Ground Truth', color='blue', density=True)
        ax.hist(all_pred[:, dim], bins=50, alpha=0.5, label='Predicted', color='red', density=True)

        ax.set_title(action_names[dim], fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        gt_mean, gt_std = all_gt[:, dim].mean(), all_gt[:, dim].std()
        pred_mean, pred_std = all_pred[:, dim].mean(), all_pred[:, dim].std()

        stats_text = f'GT: μ={gt_mean:.3f}, σ={gt_std:.3f}\nPred: μ={pred_mean:.3f}, σ={pred_std:.3f}'
        ax.text(0.98, 0.98, stats_text,
              transform=ax.transAxes, fontsize=7,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'action_distributions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDistribution plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', type=str, default=None, help='Override data root')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--num_trajectories', type=int, default=10, help='Number of trajectories to visualize')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--mode', type=str, default='full_stage',
                       choices=['full_stage', 'window', 'distribution'],
                       help='Visualization mode: full_stage (full trajectory with stages), window (prediction windows), distribution (action distributions)')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']

    if args.data_root:
        config['data_root'] = args.data_root

    
    if args.save_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        save_dir = os.path.join(checkpoint_dir, 'visualizations')
    else:
        save_dir = args.save_dir

    print(f"\nVisualization output directory: {save_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

   
    checkpoint_dir = os.path.dirname(args.checkpoint)
    normalizer_path = os.path.join(checkpoint_dir, 'normalizer.pth')

    if not os.path.exists(normalizer_path):
        print(f"Warning: Normalizer not found at {normalizer_path}")
        if 'stats' in checkpoint:
            normalizer = Normalizer(checkpoint['stats'])
        else:
            raise ValueError("No normalizer found")
    else:
        normalizer = Normalizer.load(normalizer_path)

    print("Creating validation dataset...")
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

    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Val episodes: {len(val_dataset.episode_ids)}")

    print("\nCreating ACT model...")
    model = ACTPolicy(
        proprio_dim=config['proprio_dim'],
        stage_dim=config.get('stage_dim', 5),
        action_dim=config['action_dim'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        hidden_dim=config['hidden_dim'],
        nheads=config['nheads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        latent_dim=config.get('latent_dim', 32),
        dropout=config['dropout'],
        use_vae=config['use_vae'],
        use_image=config.get('use_image', False),
        vision_feature_dim=config.get('vision_feature_dim', 512),
    ).to(device)

  
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("Model loaded successfully (strict mode)")
  
    if args.mode == 'full_stage':
        print("\n=== Visualizing Full Trajectories with All Sliding Windows ===")
        num_episodes = min(args.num_trajectories, len(val_dataset.episode_ids))
        for traj_idx in range(num_episodes):
            visualize_sliding_windows( 
                model, normalizer, val_dataset, config, device,
                trajectory_idx=traj_idx,
                save_dir=save_dir
            )

    elif args.mode == 'window':
        print("\n=== Visualizing Prediction Windows (Original window-based plots) ===")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        visualize_trajectories(
            model, val_loader, normalizer, device,
            num_trajectories=args.num_trajectories,
            save_dir=save_dir
        )

    elif args.mode == 'distribution':
        print("\n=== Plotting Action Distributions ===")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        plot_action_distributions(model, val_loader, normalizer, device, save_dir=save_dir)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()