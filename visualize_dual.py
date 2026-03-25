import numpy as np
import cv2
import os
import glob
from utils import natural_sort

# Paths
MEAN_PATH = "op1_mean_tracks.npy"
VAR_PATH = "op1_var_tracks.npy"
RAW_DIR = "Tracking_Rigid_Training/Training/OP1/Raw"
OUT_DIR_STD = "tracking_vis_standard"
OUT_DIR_UNC = "tracking_vis_uncertainty"
VIDEO_STD = "op1_standard_tracking.mp4"
VIDEO_UNC = "op1_uncertainty_tracking.mp4"

def create_frames(mean_tracks, var_tracks, frame_paths, out_dir, show_uncertainty=True):
    os.makedirs(out_dir, exist_ok=True)
    var_scalar = np.sum(var_tracks, axis=-1)
    K = 3.0
    
    for t, frame_path in enumerate(frame_paths):
        img = cv2.imread(frame_path)
        for n in range(mean_tracks.shape[1]):
            x, y = mean_tracks[t, n]
            
            # Draw mean point
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            if show_uncertainty:
                var = var_scalar[t, n]
                std = np.sqrt(var)
                radius = int(3 + K * std)
                cv2.circle(img, (int(x), int(y)), radius, (0, 255, 255), 1)
            
        out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
        cv2.imwrite(out_path, img)

def create_video(frame_dir, video_out):
    frame_paths = natural_sort(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frame_paths:
        return
    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out, fourcc, 10.0, (w, h))
    for f in frame_paths:
        out.write(cv2.imread(f))
    out.release()
    print(f"Video saved to {video_out}")

def main():
    if not os.path.exists(MEAN_PATH):
        print("Results not found yet.")
        return
        
    mean_tracks = np.load(MEAN_PATH)
    var_tracks = np.load(VAR_PATH)
    frame_paths = natural_sort(glob.glob(os.path.join(RAW_DIR, "*.png")))[:mean_tracks.shape[0]]
    
    print("Generating standard visualization (no uncertainty)...")
    create_frames(mean_tracks, var_tracks, frame_paths, OUT_DIR_STD, show_uncertainty=False)
    create_video(OUT_DIR_STD, VIDEO_STD)
    
    print("Generating uncertainty visualization (with halos)...")
    create_frames(mean_tracks, var_tracks, frame_paths, OUT_DIR_UNC, show_uncertainty=True)
    create_video(OUT_DIR_UNC, VIDEO_UNC)

if __name__ == "__main__":
    main()
