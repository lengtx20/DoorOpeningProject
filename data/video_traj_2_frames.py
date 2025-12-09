import os
import av
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent 


VIDEO_DIR = SCRIPT_DIR / "video_traj_raw"
LOG_DIR   = SCRIPT_DIR / "logs_csv_raw"
OUT_DIR   = SCRIPT_DIR / "processed_data"
os.makedirs(OUT_DIR, exist_ok=True)

VIDEO_DIR = str(VIDEO_DIR)
LOG_DIR = str(LOG_DIR)
OUT_DIR = str(OUT_DIR)



def visualize_random_rgb(root=None):

    if root is None:
        root = str(OUT_DIR)  
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
    log_dict["frame_idx"] = np.arange(len(df), dtype=np.int32)

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
    print("="*60)
    print("Video to Frames Processor")
    print("="*60)
    print(f"Project root: {REPO_ROOT}")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print("="*60)

    if not os.path.exists(VIDEO_DIR):
        print(f"[ERROR] Video directory does not exist: {VIDEO_DIR}")
        return

    if not os.path.exists(LOG_DIR):
        print(f"[ERROR] Log directory does not exist: {LOG_DIR}")
        return

    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])

    if not videos:
        print("[WARNING] No video files found!")
        return

    print(f"\nFound {len(videos)} video(s) to process\n")

    processed_count = 0
    skipped_count = 0

    for vid in videos:
        video_path = os.path.join(VIDEO_DIR, vid)
        log_name = map_video_to_log(vid)
        log_path = os.path.join(LOG_DIR, log_name)

        if not os.path.exists(log_path):
            print(f"[SKIP] No matching log for {vid}")
            skipped_count += 1
            continue

        print(f"[PROCESS] video={vid}  log={log_name}")
        processed_count += 1

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

    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Processed: {processed_count} videos")
    print(f"Skipped: {skipped_count} videos")
    print(f"Output directory: {OUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
    visualize_random_rgb()