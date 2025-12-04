import os
import av
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

REPO_ROOT = "/home/jason/DoorOpeningProject"
VIDEO_DIR = os.path.join(REPO_ROOT, "diffusion_policy/data/video_traj")
LOG_DIR   = os.path.join(REPO_ROOT, "diffusion_policy/data/logs_csv")
OUT_DIR   = os.path.join(REPO_ROOT, "diffusion_policy/data/rgb_logs")
os.makedirs(OUT_DIR, exist_ok=True)



def visualize_random_rgb(root="/home/jason/DoorOpeningProject/diffusion_policy/data/rgb_logs"):
    episodes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not episodes:
        print("No episodes found.")
        return

    episode = random.choice(episodes)
    episode_path = os.path.join(root, episode)

    frames = [f for f in os.listdir(episode_path) if f.endswith(".npy") and f != "log_dict.npy"]
    if not frames:
        print(f"No frames found in {episode}.")
        return

    frame_file = random.choice(frames)
    frame_path = os.path.join(episode_path, frame_file)


    arr = np.load(frame_path)

    plt.imshow(arr)
    plt.title(f"{episode} — {frame_file}")
    plt.axis("off")
    plt.show()

    print(f"Episode: {episode}")
    print(f"Frame: {frame_file}")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")


def map_video_to_log(video_name):
    base = video_name[:-4] 

    if base == "traj_55_null":
        return "merged_50hz_log.csv"

    idx = int(base.split("_")[-1])
    return f"merged_50hz_log{idx}.csv"


def load_video_rgb(video_path):
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format="rgb24")
        frames.append(rgb)
    return np.stack(frames)


def load_log_csv_as_dict(csv_path):
    df = pd.read_csv(csv_path)
    # print("\nFIRST ROW OF CSV:", csv_path)
    # print(df.iloc[0]) 

    log_dict = {}
    for col in df.columns:
        log_dict[col] = df[col].to_numpy()

    log_dict["length"] = len(df)
    return log_dict


def save_frames(frames, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for i, f in enumerate(frames):
        np.save(os.path.join(out_folder, f"{i:06d}.npy"), f)


def visualize_rgb_folder(folder, step=1):
    files = sorted([x for x in os.listdir(folder) if x.endswith(".npy")])
    for i, fname in enumerate(files[::step]):
        arr = np.load(os.path.join(folder, fname))
        plt.imshow(arr)
        plt.title(f"{folder} — frame {i*step}")
        plt.axis("off")
        plt.pause(0.01)
    plt.show()


def main():
    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])

    for vid in videos:
        video_path = os.path.join(VIDEO_DIR, vid)
        log_name = map_video_to_log(vid)
        log_path = os.path.join(LOG_DIR, log_name)

        if not os.path.exists(log_path):
            print(f"[SKIP] No matching log for {vid}")
            continue

        print(f"[PROCESS] video={vid}  log={log_name}")

        frames = load_video_rgb(video_path)
        log_dict = load_log_csv_as_dict(log_path)

        T_vid = len(frames)
        T_log = log_dict["length"]

        if T_vid != T_log:
            T = min(T_vid, T_log)
            frames = frames[:T]
            for k in log_dict.keys():
                if k != "length":
                    log_dict[k] = log_dict[k][:T]
            log_dict["length"] = T

        out_folder = os.path.join(OUT_DIR, vid[:-4])
        save_frames(frames, out_folder)

        np.save(
            os.path.join(out_folder, "log_dict.npy"),
            log_dict,
            allow_pickle=True
        )

      


if __name__ == "__main__":
    main()
    visualize_random_rgb()