# Assited by Gemini Pro 3.0 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import yaml
import argparse
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffusion_policy.utils.dataset import G1Dataset
from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.utils.normalizer import Normalizer


def load_config(path=None):
    if path is None:
        script_dir = Path(__file__).parent
        path = script_dir.parent / "configs" / "default.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def integrate_trajectory(start_pose, actions, dt=0.02):
    T = len(actions)
    curr_pos = start_pose[:3].copy()
    curr_quat = start_pose[3:].copy()

    l_world_hist, r_world_hist, torso_hist = [], [], []

    for t in range(T):
        r = R.from_quat(curr_quat)

        l_local = actions[t, 0:3]
        r_local = actions[t, 3:6]

        l_world = curr_pos + r.apply(l_local)
        r_world = curr_pos + r.apply(r_local)

        l_world_hist.append(l_world)
        r_world_hist.append(r_world)
        torso_hist.append(curr_pos.copy())

        v_local = np.array([actions[t, 6], actions[t, 7], 0.0])
        w_yaw = actions[t, 8]

        v_world = r.apply(v_local)
        curr_pos = curr_pos + v_world * dt

        r_new = r * R.from_euler('z', w_yaw * dt)
        curr_quat = r_new.as_quat()
        curr_quat = curr_quat / np.linalg.norm(curr_quat)

    return np.array(l_world_hist), np.array(r_world_hist), np.array(torso_hist)


def update_lines(num, data, lines, markers, gt_only=False):

    sl = slice(0, num + 1)

    if gt_only:

        lines[0].set_data(data['gt_l'][sl, 0].flatten(), data['gt_l'][sl, 1].flatten())
        lines[0].set_3d_properties(data['gt_l'][sl, 2].flatten())

        lines[1].set_data(data['gt_r'][sl, 0].flatten(), data['gt_r'][sl, 1].flatten())
        lines[1].set_3d_properties(data['gt_r'][sl, 2].flatten())

        lines[2].set_data(data['gt_torso'][sl, 0].flatten(), data['gt_torso'][sl, 1].flatten())
        lines[2].set_3d_properties(data['gt_torso'][sl, 2].flatten())

        markers[0].set_data([data['gt_l'][num, 0]], [data['gt_l'][num, 1]])
        markers[0].set_3d_properties([data['gt_l'][num, 2]])

        markers[1].set_data([data['gt_r'][num, 0]], [data['gt_r'][num, 1]])
        markers[1].set_3d_properties([data['gt_r'][num, 2]])

        markers[2].set_data([data['gt_torso'][num, 0]], [data['gt_torso'][num, 1]])
        markers[2].set_3d_properties([data['gt_torso'][num, 2]])

        return lines + markers

  
    lines[0].set_data(data['pred_l'][sl, 0].flatten(), data['pred_l'][sl, 1].flatten())
    lines[0].set_3d_properties(data['pred_l'][sl, 2].flatten())

    lines[1].set_data(data['pred_r'][sl, 0].flatten(), data['pred_r'][sl, 1].flatten())
    lines[1].set_3d_properties(data['pred_r'][sl, 2].flatten())

    lines[2].set_data(data['pred_torso'][sl, 0].flatten(), data['pred_torso'][sl, 1].flatten())
    lines[2].set_3d_properties(data['pred_torso'][sl, 2].flatten())

   
    markers[0].set_data([data['pred_l'][num, 0]], [data['pred_l'][num, 1]])
    markers[0].set_3d_properties([data['pred_l'][num, 2]])

    markers[1].set_data([data['pred_r'][num, 0]], [data['pred_r'][num, 1]])
    markers[1].set_3d_properties([data['pred_r'][num, 2]])

    markers[2].set_data([data['pred_torso'][num, 0]], [data['pred_torso'][num, 1]])
    markers[2].set_3d_properties([data['pred_torso'][num, 2]])

    lines[3].set_data(data['gt_l'][sl, 0].flatten(), data['gt_l'][sl, 1].flatten())
    lines[3].set_3d_properties(data['gt_l'][sl, 2].flatten())

    lines[4].set_data(data['gt_r'][sl, 0].flatten(), data['gt_r'][sl, 1].flatten())
    lines[4].set_3d_properties(data['gt_r'][sl, 2].flatten())

    lines[5].set_data(data['gt_torso'][sl, 0].flatten(), data['gt_torso'][sl, 1].flatten())
    lines[5].set_3d_properties(data['gt_torso'][sl, 2].flatten())

    return lines + markers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--sample_idx', type=int, default=-1)
    parser.add_argument('--gt_only', action='store_true')
    args = parser.parse_args()

    config = load_config()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    val_ds = G1Dataset(
        config["data_root"],
        mode='val',
        obs_horizon=config["obs_horizon"],
        pred_horizon=config["pred_horizon"],
        use_proprio=config["use_proprio"]
    )

    idx = args.sample_idx if args.sample_idx >= 0 else np.random.randint(0, len(val_ds))
    sample = val_ds[idx]

    start_pose = sample['start_pose'].cpu().numpy()

    batch = {
        'image': sample['image'].unsqueeze(0).to(device),
        'agent_pos': sample['agent_pos'].unsqueeze(0).to(device),
        'action': sample['action'].unsqueeze(0).to(device)
    }

   
    gt_action = batch['action'].cpu().numpy()[0]

    pred_action = None

    if not args.gt_only:
        ckpt = torch.load(args.checkpoint, map_location=device)

        policy = DiffusionPolicy(
            action_dim=config["action_dim"],
            obs_horizon=config["obs_horizon"],
            pred_horizon=config["pred_horizon"],
            vision_feature_dim=config["vision_feature_dim"],
            proprio_dim=config["proprio_dim"],
            use_proprio=config["use_proprio"],
            num_inference_steps=16
        ).to(device)

        policy.load_state_dict(ckpt['model_state_dict'])
        policy.eval()

        normalizer = Normalizer(ckpt['stats'])

        print("Action Mean:", normalizer.stats['action']['mean'])
        print("Action Std: ", normalizer.stats['action']['std'])


        batch_norm = normalizer.normalize({
            'image': batch['image'],
            'agent_pos': batch['agent_pos'],
            'action': batch['action'].clone()
        })

        with torch.no_grad():
            pred_action_norm = policy.predict_action(
                batch_norm['image'], batch_norm['agent_pos']
            )


        pred_action = normalizer.unnormalize_action(pred_action_norm).cpu().numpy()[0]


    g_l, g_r, g_t = integrate_trajectory(start_pose, gt_action)

    if pred_action is not None:
        p_l, p_r, p_t = integrate_trajectory(start_pose, pred_action)
        print("Trajectory Integrated Successfully.")
        print("\n================ PER-STEP ERROR REPORT ================")

        total_err_l = 0.0
        total_err_r = 0.0
        total_err_t = 0.0

        for t in range(len(gt_action)):
            print(f"\n--- STEP {t} ---")

            # Raw action difference
            print("Action GT:   ", gt_action[t])
            print("Action Pred: ", pred_action[t])
            print("Action Diff: ", pred_action[t] - gt_action[t])

            # Left hand error
            err_l = np.linalg.norm(p_l[t] - g_l[t])
            total_err_l += err_l
            print("Left  GT:", g_l[t])
            print("Left  Pred:", p_l[t])
            print("Left  Err (m):", err_l)

            # Right hand error
            err_r = np.linalg.norm(p_r[t] - g_r[t])
            total_err_r += err_r
            print("Right GT:", g_r[t])
            print("Right Pred:", p_r[t])
            print("Right Err (m):", err_r)


        # Summary
        print("\n================ TOTAL ERROR SUMMARY ================")
        print("Total Left  Error (sum):", total_err_l)
        print("Total Right Error (sum):", total_err_r)
        print("Total Torso Error (sum):", total_err_t)
        print("Mean Left  Error:", total_err_l / len(gt_action))
        print("Mean Right Error:", total_err_r / len(gt_action))
        print("=====================================================\n")
    else:
        p_l = p_r = p_t = None


    # print("Pred Action[0]:", pred_action[0] if pred_action is not None else None)
    # print("GT Action[0]:  ", gt_action[0])

    # print("Start Pose (world):", start_pose)

    # print("\n--- LEFT HAND ---")
    # print("Pred L[0]:", p_l[0] if p_l is not None else None)
    # print("GT   L[0]:", g_l[0])
    # print("Error L (m):", (p_l[0] - g_l[0]) if p_l is not None else None)

    # print("\n--- RIGHT HAND ---")
    # print("Pred R[0]:", p_r[0] if p_r is not None else None)
    # print("GT   R[0]:", g_r[0])
    # print("Error R (m):", (p_r[0] - g_r[0]) if p_r is not None else None)


    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.set_title("Prediction vs Ground Truth" if not args.gt_only else "Ground Truth Only")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=30, azim=-135)

    if pred_action is not None:
        all_pts = np.concatenate([p_l, p_r, p_t, g_l, g_r, g_t])
    else:
        all_pts = np.concatenate([g_l, g_r, g_t])

    center = all_pts.mean(axis=0)
    radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(0, 2.0)

    if pred_action is not None:
        l_line, = ax.plot([], [], [], color='red', linewidth=3, label='Left Pred')
        r_line, = ax.plot([], [], [], color='blue', linewidth=3, label='Right Pred')
        t_line, = ax.plot([], [], [], color='green', linewidth=3, label='Torso Pred')

        l_gt, = ax.plot([], [], [], color='red', linestyle='--', linewidth=2, label='Left GT')
        r_gt, = ax.plot([], [], [], color='blue', linestyle='--', linewidth=2, label='Right GT')
        t_gt, = ax.plot([], [], [], color='green', linestyle='--', linewidth=2, label='Torso GT')

        l_mark, = ax.plot([], [], [], 'o', color='red')
        r_mark, = ax.plot([], [], [], 'o', color='blue')
        t_mark, = ax.plot([], [], [], 'o', color='green')

        lines = [l_line, r_line, t_line, l_gt, r_gt, t_gt]
        markers = [l_mark, r_mark, t_mark]

    else:
        l_gt, = ax.plot([], [], [], color='red', label='Left GT')
        r_gt, = ax.plot([], [], [], color='blue', label='Right GT')
        t_gt, = ax.plot([], [], [], color='black', label='Torso GT')

        l_mark, = ax.plot([], [], [], 'o', color='red')
        r_mark, = ax.plot([], [], [], 'o', color='blue')
        t_mark, = ax.plot([], [], [], '^', color='black')

        lines = [l_gt, r_gt, t_gt]
        markers = [l_mark, r_mark, t_mark]

    ax.legend(loc='upper left')

    ax2 = fig.add_subplot(1, 2, 2)
    img = sample['image'][-1].permute(1, 2, 0).numpy()
    ax2.imshow(img)
    ax2.set_title("Input Vision")
    ax2.axis('off')

    data = {
        'pred_l': p_l, 'pred_r': p_r, 'pred_torso': p_t,
        'gt_l': g_l, 'gt_r': g_r, 'gt_torso': g_t
    }

    ani = animation.FuncAnimation(
        fig,
        update_lines,
        frames=len(gt_action),
        fargs=(data, lines, markers, args.gt_only),
        interval=100
    )
    save_path = f"vis_sample_{idx}.gif"
    ani.save(save_path, writer="pillow", fps=20)
    print("Saved GIF to:", save_path)
    plt.show()

if __name__ == "__main__":
    main()