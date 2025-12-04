
import os
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from bc_policy.utils import BCDataset
import torch
from torch.utils.data import DataLoader
def analyze_command_ranges(data_root: str):
    print("="*80)
    print("CHECKING LOGGED COMMAND RANGES vs WBC TRAINING RANGES")
    print("="*80)
    print(f"\nLoading dataset from: {data_root}")
    dataset = BCDataset(
        dataset_root=data_root,
        mode='train',
        obs_horizon=2,
        pred_horizon=16,
        sample_stride=1,
    )
    print(f"Dataset size: {len(dataset)} samples")
    print("\nCollecting actions from all samples...")
    all_actions = []
    for i in range(len(dataset)):
        sample = dataset[i]
        action = sample['action'].numpy()  
        all_actions.append(action)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(dataset)} samples...")
    all_actions = np.concatenate(all_actions, axis=0)  
    print(f"Total actions: {all_actions.shape[0]}")
    action_names = [
        'Left Hand X', 'Left Hand Y', 'Left Hand Z',
        'Right Hand X', 'Right Hand Y', 'Right Hand Z',
        'Base Vel X', 'Base Vel Y', 'Yaw Speed', 'Grasp'
    ]
    wbc_ranges = {
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
    print("\n" + "="*80)
    print("COMMAND RANGE ANALYSIS")
    print("="*80)
    print(f"\n{'Dimension':<15s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s} | {'WBC Range':>15s} {'Status':>10s}")
    print("-"*100)
    out_of_range_dims = []
    for i, name in enumerate(action_names):
        data_min = all_actions[:, i].min()
        data_max = all_actions[:, i].max()
        data_mean = all_actions[:, i].mean()
        data_std = all_actions[:, i].std()
        wbc_min, wbc_max = wbc_ranges[name]
        in_range = (data_min >= wbc_min - 0.01) and (data_max <= wbc_max + 0.01)  
        status = "✓ OK" if in_range else "✗ OUT"
        if not in_range:
            out_of_range_dims.append(name)
        print(f"{name:<15s} "
              f"{data_min:10.4f} "
              f"{data_max:10.4f} "
              f"{data_mean:10.4f} "
              f"{data_std:10.4f} | "
              f"[{wbc_min:6.2f}, {wbc_max:6.2f}] "
              f"{status:>10s}")
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if len(out_of_range_dims) == 0:
        print("\n✓ All logged commands are within WBC training ranges!")
        print("  The WBC policy should be able to track your BC policy's outputs.")
    else:
        print(f"\n✗ {len(out_of_range_dims)}/{len(action_names)} dimensions are OUT OF RANGE:")
        for dim in out_of_range_dims:
            data_min = all_actions[:, action_names.index(dim)].min()
            data_max = all_actions[:, action_names.index(dim)].max()
            wbc_min, wbc_max = wbc_ranges[dim]
            print(f"\n  {dim}:")
            print(f"    Logged range:  [{data_min:.4f}, {data_max:.4f}]")
            print(f"    WBC range:     [{wbc_min:.4f}, {wbc_max:.4f}]")
            if data_min < wbc_min:
                print(f"    ⚠ Min value is {wbc_min - data_min:.4f} below WBC range")
            if data_max > wbc_max:
                print(f"    ⚠ Max value is {data_max - wbc_max:.4f} above WBC range")
        print("\n  RECOMMENDATIONS:")
        print("  1. Clip BC policy outputs to WBC ranges during inference")
        print("  2. Retrain WBC policy with wider command ranges")
        print("  3. Normalize logged data to match WBC ranges")
    print("\n" + "="*80)
    print("OUT-OF-RANGE SAMPLE ANALYSIS")
    print("="*80)
    for i, name in enumerate(action_names):
        wbc_min, wbc_max = wbc_ranges[name]
        below_range = (all_actions[:, i] < wbc_min).sum()
        above_range = (all_actions[:, i] > wbc_max).sum()
        total_out = below_range + above_range
        percentage = (total_out / len(all_actions)) * 100
        if total_out > 0:
            print(f"\n{name}:")
            print(f"  {total_out:,} / {len(all_actions):,} samples out of range ({percentage:.2f}%)")
            if below_range > 0:
                print(f"    {below_range:,} below minimum ({wbc_min:.2f})")
            if above_range > 0:
                print(f"    {above_range:,} above maximum ({wbc_max:.2f})")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/jason/DoorOpeningProject/diffusion_policy/data/torso_rgb_logs',
        help='Path to dataset'
    )
    args = parser.parse_args()
    analyze_command_ranges(args.data_root)
