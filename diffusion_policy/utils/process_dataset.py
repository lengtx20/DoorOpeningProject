import os
import shutil
import numpy as np
import json
import random
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


REPO_ROOT = "/home/jason/DoorOpeningProject"
INPUT_DIR = os.path.join(REPO_ROOT, "diffusion_policy/data/rgb_logs")
OUTPUT_DIR = os.path.join(REPO_ROOT, "diffusion_policy/data/torso_rgb_logs")

VAL_RATIO = 0.1

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def add_torso_features(log_dict):
   
    torso_quat = np.stack([
        log_dict['quat_x'], log_dict['quat_y'], log_dict['quat_z'], log_dict['quat_w']
    ], axis=-1)
    
    torso_pos = np.stack([
        log_dict['pos_x'], log_dict['pos_y'], log_dict['pos_z']
    ], axis=-1)

   
    r_torso = R.from_quat(torso_quat)
    
    r_torso_inv = r_torso.inv()


    vel_world = np.stack([
        log_dict['vel_x'], log_dict['vel_y'], log_dict['vel_z']
    ], axis=-1)
    

    vel_body = r_torso_inv.apply(vel_world)
    
    log_dict['vel_body_x'] = vel_body[:, 0]
    log_dict['vel_body_y'] = vel_body[:, 1]
    log_dict['vel_body_z'] = vel_body[:, 2]


    for prefix in ['left_wrist', 'right_wrist']:
       
        if f'{prefix}_pos_x' not in log_dict:
            continue

  
        w_pos_world = np.stack([
            log_dict[f'{prefix}_pos_x'], 
            log_dict[f'{prefix}_pos_y'], 
            log_dict[f'{prefix}_pos_z']
        ], axis=-1)
        
     
        w_pos_body = r_torso_inv.apply(w_pos_world - torso_pos)
        
        log_dict[f'{prefix}_torso_pos_x'] = w_pos_body[:, 0]
        log_dict[f'{prefix}_torso_pos_y'] = w_pos_body[:, 1]
        log_dict[f'{prefix}_torso_pos_z'] = w_pos_body[:, 2]

     
        w_quat_world = np.stack([
            log_dict[f'{prefix}_quat_x'], 
            log_dict[f'{prefix}_quat_y'], 
            log_dict[f'{prefix}_quat_z'], 
            log_dict[f'{prefix}_quat_w']
        ], axis=-1)
        
        r_wrist = R.from_quat(w_quat_world)
        r_wrist_body = r_torso_inv * r_wrist
        w_quat_body = r_wrist_body.as_quat() 
        
        log_dict[f'{prefix}_torso_quat_x'] = w_quat_body[:, 0]
        log_dict[f'{prefix}_torso_quat_y'] = w_quat_body[:, 1]
        log_dict[f'{prefix}_torso_quat_z'] = w_quat_body[:, 2]
        log_dict[f'{prefix}_torso_quat_w'] = w_quat_body[:, 3]

    return log_dict

def main():
    ensure_dir(OUTPUT_DIR)
    
    if not os.path.exists(INPUT_DIR):
        print("Source directory missing.")
        return

    traj_folders = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])
    processed_episodes = []

    for traj_name in tqdm(traj_folders):
        src_traj_path = os.path.join(INPUT_DIR, traj_name)
        dst_traj_path = os.path.join(OUTPUT_DIR, traj_name)
        
        log_path = os.path.join(src_traj_path, "log_dict.npy")
        
        if not os.path.exists(log_path): continue
        log_dict = np.load(log_path, allow_pickle=True).item()
        log_dict = add_torso_features(log_dict)
        ensure_dir(dst_traj_path)
        
        np.save(os.path.join(dst_traj_path, "log_dict.npy"), log_dict, allow_pickle=True)
        
        for fname in os.listdir(src_traj_path):
            if fname.endswith(".npy") and fname != "log_dict.npy":
                shutil.copy2(
                    os.path.join(src_traj_path, fname),
                    os.path.join(dst_traj_path, fname)
                )
        
        processed_episodes.append(traj_name)

    random.seed(42)
    random.shuffle(processed_episodes)
    
    n_val = int(len(processed_episodes) * VAL_RATIO)
    n_val = max(1, n_val) if processed_episodes else 0
    
    split_stats = {
        "train": processed_episodes[n_val:],
        "val": processed_episodes[:n_val]
    }
    
    with open(os.path.join(OUTPUT_DIR, "split.json"), "w") as f:
        json.dump(split_stats, f, indent=4)
        

if __name__ == "__main__":
    main()