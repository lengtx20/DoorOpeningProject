
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
def load_all_trajectories(data_root: str):
    print(f"Loading trajectories from: {data_root}")
    traj_dirs = sorted([d for d in Path(data_root).iterdir() if d.is_dir() and d.name.startswith('traj_')])
    print(f"Found {len(traj_dirs)} trajectory directories")
    trajectories = []
    for traj_dir in traj_dirs:
        log_path = traj_dir / 'log_dict.npy'
        if log_path.exists():
            try:
                log_dict = np.load(log_path, allow_pickle=True).item()
                trajectories.append({
                    'name': traj_dir.name,
                    'data': log_dict,
                    'length': len(log_dict.get('q_0', []))
                })
            except Exception as e:
                print(f"Warning: Failed to load {traj_dir.name}: {e}")
    print(f"Successfully loaded {len(trajectories)} trajectories")
    total_timesteps = sum(t['length'] for t in trajectories)
    print(f"Total timesteps: {total_timesteps:,}")
    return trajectories
def extract_commands(trajectories):
    print("\nExtracting commands from trajectories...")
    command_keys = [
        'left_wrist_torso_pos_x', 'left_wrist_torso_pos_y', 'left_wrist_torso_pos_z',
        'right_wrist_torso_pos_x', 'right_wrist_torso_pos_y', 'right_wrist_torso_pos_z',
        'vel_body_x', 'vel_body_y',
        'yaw_speed',
        'p_pressed'
    ]
    command_names = [
        'Left Hand X', 'Left Hand Y', 'Left Hand Z',
        'Right Hand X', 'Right Hand Y', 'Right Hand Z',
        'Base Vel X', 'Base Vel Y',
        'Yaw Speed',
        'Grasp'
    ]
    all_commands = {key: [] for key in command_keys}
    for traj in trajectories:
        log_dict = traj['data']
        for key in command_keys:
            if key in log_dict:
                all_commands[key].extend(log_dict[key])
    for key in command_keys:
        all_commands[key] = np.array(all_commands[key])
    return all_commands, command_keys, command_names
def analyze_command_ranges(all_commands, command_keys, command_names):
    print("\n" + "="*100)
    print("GROUND TRUTH COMMAND RANGE ANALYSIS")
    print("="*100)
    wbc_ranges = {
        'Left Hand X': (-0.15, 0.5),
        'Left Hand Y': (-0.2, 0.45),
        'Left Hand Z': (-0.05, 0.35),
        'Right Hand X': (-0.2, 0.4),
        'Right Hand Y': (-0.35, 0.2),
        'Right Hand Z': (-0.05, 0.25),
        'Base Vel X': (-0.85, 1.0),
        'Base Vel Y': (-1.15, 0.95),
        'Yaw Speed': (-3.0, 3.0),
        'Grasp': (0.0, 1.0),
    }
    original_wbc_ranges = {
        'Left Hand X': (0.0, 0.3),
        'Left Hand Y': (0.0, 0.3),
        'Left Hand Z': (0.1, 0.4),
        'Right Hand X': (0.0, 0.3),
        'Right Hand Y': (-0.3, 0.0),
        'Right Hand Z': (0.1, 0.4),
        'Base Vel X': (-0.5, 1.0),
        'Base Vel Y': (-0.3, 0.3),
        'Yaw Speed': (-0.2, 0.2),
        'Grasp': (0.0, 1.0),
    }
    print(f"\n{'Dimension':<15s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s} {'Median':>10s} | "
          f"{'Original WBC':>18s} {'New WBC':>18s} {'Coverage':>10s}")
    print("-"*140)
    stats = {}
    coverage_stats = {'original': [], 'new': []}
    for i, (key, name) in enumerate(zip(command_keys, command_names)):
        data = all_commands[key]
        data_min = np.min(data)
        data_max = np.max(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        data_median = np.median(data)
        orig_min, orig_max = original_wbc_ranges[name]
        orig_below = np.sum(data < orig_min)
        orig_above = np.sum(data > orig_max)
        orig_out = orig_below + orig_above
        orig_coverage = 100 * (1 - orig_out / len(data))
        new_min, new_max = wbc_ranges[name]
        new_below = np.sum(data < new_min)
        new_above = np.sum(data > new_max)
        new_out = new_below + new_above
        new_coverage = 100 * (1 - new_out / len(data))
        coverage_stats['original'].append(orig_coverage)
        coverage_stats['new'].append(new_coverage)
        orig_range_str = f"[{orig_min:5.2f}, {orig_max:5.2f}]"
        new_range_str = f"[{new_min:5.2f}, {new_max:5.2f}]"
        if new_coverage >= 99.9:
            coverage_str = f"✓ {new_coverage:5.1f}%"
        elif new_coverage >= 95:
            coverage_str = f"~ {new_coverage:5.1f}%"
        else:
            coverage_str = f"✗ {new_coverage:5.1f}%"
        print(f"{name:<15s} "
              f"{data_min:10.4f} "
              f"{data_max:10.4f} "
              f"{data_mean:10.4f} "
              f"{data_std:10.4f} "
              f"{data_median:10.4f} | "
              f"{orig_range_str:>18s} "
              f"{new_range_str:>18s} "
              f"{coverage_str:>10s}")
        stats[name] = {
            'min': data_min,
            'max': data_max,
            'mean': data_mean,
            'std': data_std,
            'median': data_median,
            'percentile_1': np.percentile(data, 1),
            'percentile_99': np.percentile(data, 99),
            'original_coverage': orig_coverage,
            'new_coverage': new_coverage,
            'data': data
        }
    print("\n" + "="*100)
    print("COVERAGE SUMMARY")
    print("="*100)
    orig_avg = np.mean(coverage_stats['original'])
    new_avg = np.mean(coverage_stats['new'])
    print(f"\nOriginal WBC Ranges:")
    print(f"  Average coverage: {orig_avg:.1f}%")
    dims_below_95 = sum(1 for c in coverage_stats['original'] if c < 95)
    print(f"  Dimensions with <95% coverage: {dims_below_95}/10")
    print(f"\nNew WBC Ranges (Updated):")
    print(f"  Average coverage: {new_avg:.1f}%")
    dims_below_95 = sum(1 for c in coverage_stats['new'] if c < 95)
    print(f"  Dimensions with <95% coverage: {dims_below_95}/10")
    if new_avg >= 99:
        print(f"\n✓ Excellent! New ranges cover {new_avg:.1f}% of logged data")
    elif new_avg >= 95:
        print(f"\n~ Good. New ranges cover {new_avg:.1f}% of logged data")
    else:
        print(f"\n⚠ Warning: New ranges only cover {new_avg:.1f}% of logged data")
    return stats
def plot_command_distributions(stats, command_names, output_file='command_distributions.png'):
    print(f"\nGenerating distribution plots...")
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    for i, name in enumerate(command_names):
        ax = axes[i]
        data = stats[name]['data']
        ax.hist(data, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(stats[name]['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats[name]['mean']:.3f}")
        ax.axvline(stats[name]['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats[name]['median']:.3f}")
        p1 = stats[name]['percentile_1']
        p99 = stats[name]['percentile_99']
        ax.axvspan(p1, p99, alpha=0.2, color='yellow', label=f'1-99 percentile')
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        textstr = f'Range: [{stats[name]["min"]:.3f}, {stats[name]["max"]:.3f}]\n'
        textstr += f'Std: {stats[name]["std"]:.3f}\n'
        textstr += f'Coverage: {stats[name]["new_coverage"]:.1f}%'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to: {output_file}")
    return output_file
def save_statistics(stats, output_file='command_statistics.json'):
    stats_serializable = {}
    for name, data in stats.items():
        stats_serializable[name] = {
            k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in data.items()
            if k != 'data'  
        }
    with open(output_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"\nSaved statistics to: {output_file}")
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract and analyze ground truth commands from logged data')
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/jason/DoorOpeningProject/diffusion_policy/data/torso_rgb_logs',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save output files'
    )
    args = parser.parse_args()
    trajectories = load_all_trajectories(args.data_root)
    if len(trajectories) == 0:
        print("Error: No trajectories found!")
        return
    all_commands, command_keys, command_names = extract_commands(trajectories)
    stats = analyze_command_ranges(all_commands, command_keys, command_names)
    plot_file = os.path.join(args.output_dir, 'command_distributions.png')
    plot_command_distributions(stats, command_names, plot_file)
    stats_file = os.path.join(args.output_dir, 'command_statistics.json')
    save_statistics(stats, stats_file)
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nGenerated files:")
    print(f"  - {plot_file}")
    print(f"  - {stats_file}")
if __name__ == "__main__":
    main()
