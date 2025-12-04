
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
bc_root = PROJECT_ROOT / "bc_policy"
sys.path.insert(0, str(bc_root))
sys.path.insert(0, str(PROJECT_ROOT))
import torch
from bc_policy.utils import load_bc_policy_and_normalizer
def load_gt_trajectory(traj_path: str):
    log_path = Path(traj_path) / "log_dict.npy"
    if not log_path.exists():
        raise FileNotFoundError(f"GT trajectory not found: {log_path}")
    log_dict = np.load(log_path, allow_pickle=True).item()
    T = len(log_dict['timestamp'])
    proprio = np.stack([log_dict[f'q_{i}'] for i in range(29)], axis=1)  
    gt_commands = np.stack([
        log_dict['left_wrist_torso_pos_x'],
        log_dict['left_wrist_torso_pos_y'],
        log_dict['left_wrist_torso_pos_z'],
        log_dict['right_wrist_torso_pos_x'],
        log_dict['right_wrist_torso_pos_y'],
        log_dict['right_wrist_torso_pos_z'],
        log_dict['vel_body_x'],
        log_dict['vel_body_y'],
        log_dict['yaw_speed'],
        log_dict['p_pressed'].astype(np.float32),
    ], axis=1)  
    return {
        'proprio': proprio,
        'gt_commands': gt_commands,
        'timestamp': np.array(log_dict['timestamp']),
    }
def predict_bc_commands(bc_policy, normalizer, proprio_sequence, device='cuda:0'):
    T = len(proprio_sequence)
    obs_horizon = 2
    predictions = []
    for t in range(obs_horizon - 1, T - 1):
        obs_window = np.stack([proprio_sequence[t-1], proprio_sequence[t]])
        obs_tensor = torch.from_numpy(obs_window).float().to(device).unsqueeze(0)
        batch = {'proprio': obs_tensor}
        normalized_batch = normalizer.normalize(batch)
        obs_norm = normalized_batch['proprio']
        with torch.no_grad():
            action_pred = bc_policy(obs_norm)  
        action_normalized = action_pred[0, 0]  
        action_denorm = normalizer.denormalize(action_normalized, 'action')
        action = action_denorm.cpu().numpy()
        predictions.append(action)
    predictions = np.array(predictions)  
    return predictions
def compute_errors(bc_predictions, gt_commands, traj_name):
    errors = bc_predictions - gt_commands
    labels = [
        'L_hand_x', 'L_hand_y', 'L_hand_z',
        'R_hand_x', 'R_hand_y', 'R_hand_z',
        'vel_x', 'vel_y', 'yaw_speed', 'grasp'
    ]
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS: {traj_name}")
    print(f"{'='*80}")
    metrics = {}
    for i, label in enumerate(labels):
        mae = np.mean(np.abs(errors[:, i]))
        rmse = np.sqrt(np.mean(errors[:, i]**2))
        max_err = np.max(np.abs(errors[:, i]))
        print(f"\n{label:12s}: MAE={mae:8.4f}  RMSE={rmse:8.4f}  MaxErr={max_err:8.4f}")
        print(f"  BC:  mean={np.mean(bc_predictions[:, i]):8.4f}  std={np.std(bc_predictions[:, i]):8.4f}")
        print(f"  GT:  mean={np.mean(gt_commands[:, i]):8.4f}  std={np.std(gt_commands[:, i]):8.4f}")
        metrics[label] = {
            'mae': mae,
            'rmse': rmse,
            'max_err': max_err
        }
    print(f"\n{'='*80}")
    return metrics
def plot_comparison(bc_predictions, gt_commands, timestamp, traj_name, save_path):
    labels = [
        'L_hand_x', 'L_hand_y', 'L_hand_z',
        'R_hand_x', 'R_hand_y', 'R_hand_z',
        'vel_x', 'vel_y', 'yaw_speed', 'grasp'
    ]
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(timestamp, gt_commands[:, i], 'b-', label='GT', linewidth=1.5, alpha=0.7)
        ax.plot(timestamp, bc_predictions[:, i], 'r--', label='BC', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        mae = np.mean(np.abs(bc_predictions[:, i] - gt_commands[:, i]))
        ax.set_title(f"{label} (MAE={mae:.4f})")
    fig.suptitle(f"BC vs GT: {traj_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved plot to: {save_path}")
    plt.close()
def process_trajectory(traj_name, data_root, bc_policy, normalizer, device, output_dir):
    print(f"\n{'='*80}")
    print(f"Processing: {traj_name}")
    print(f"{'='*80}")
    traj_path = Path(data_root) / traj_name
    data = load_gt_trajectory(traj_path)
    proprio = data['proprio']
    gt_commands = data['gt_commands']
    timestamp = data['timestamp']
    print(f"[INFO] Loaded {len(proprio)} timesteps")
    print("[INFO] Running BC predictions...")
    bc_predictions = predict_bc_commands(bc_policy, normalizer, proprio, device)
    gt_commands_aligned = gt_commands[2:]
    timestamp_aligned = timestamp[2:]
    print(f"[INFO] Aligned BC predictions: {bc_predictions.shape}")
    print(f"[INFO] Aligned GT commands: {gt_commands_aligned.shape}")
    metrics = compute_errors(bc_predictions, gt_commands_aligned, traj_name)
    save_path = output_dir / f"bc_vs_gt_{traj_name}.png"
    plot_comparison(bc_predictions, gt_commands_aligned, timestamp_aligned, traj_name, save_path)
    analysis_path = output_dir / f"bc_vs_gt_{traj_name}.npz"
    np.savez(
        analysis_path,
        bc_predictions=bc_predictions,
        gt_commands=gt_commands_aligned,
        timestamp=timestamp_aligned,
        proprio=proprio[1:-1],  
    )
    print(f"[INFO] Saved analysis data to: {analysis_path}")
    return metrics
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_checkpoint', type=str, required=True, help='Path to BC checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='bc_policy/val_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.data_root) / 'split.json'
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    val_trajs = split_data['val']
    print(f"\n[INFO] Found {len(val_trajs)} validation trajectories:")
    for traj in val_trajs:
        print(f"  - {traj}")
    print(f"\n[INFO] Loading BC policy from: {args.bc_checkpoint}")
    model, normalizer, config = load_bc_policy_and_normalizer(args.bc_checkpoint, args.device)
    print(f"[INFO] BC config: {config}")
    all_metrics = {}
    for traj_name in val_trajs:
        try:
            metrics = process_trajectory(
                traj_name, args.data_root, model, normalizer, args.device, output_dir
            )
            all_metrics[traj_name] = metrics
        except Exception as e:
            print(f"[ERROR] Failed to process {traj_name}: {e}")
            continue
    print(f"\n{'='*80}")
    print("SUMMARY: MAE across all validation trajectories")
    print(f"{'='*80}")
    labels = [
        'L_hand_x', 'L_hand_y', 'L_hand_z',
        'R_hand_x', 'R_hand_y', 'R_hand_z',
        'vel_x', 'vel_y', 'yaw_speed', 'grasp'
    ]
    print(f"\n{'Trajectory':<15}", end='')
    for label in labels:
        print(f"{label:>10}", end='')
    print()
    for traj_name, metrics in all_metrics.items():
        print(f"{traj_name:<15}", end='')
        for label in labels:
            mae = metrics[label]['mae']
            print(f"{mae:10.4f}", end='')
        print()
    print(f"{'AVERAGE':<15}", end='')
    for label in labels:
        avg_mae = np.mean([all_metrics[traj][label]['mae'] for traj in all_metrics])
        print(f"{avg_mae:10.4f}", end='')
    print()
    print(f"\n{'='*80}")
    print(f"[INFO] All results saved to: {output_dir}")
    print(f"{'='*80}\n")
if __name__ == "__main__":
    main()
