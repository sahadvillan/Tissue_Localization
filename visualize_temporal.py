import cv2
import os
import glob
import numpy as np
from utils import natural_sort
from temporal_analyzer import TemporalAnalyzer

# Paths
SEQ_NAME = "op1"
MEAN_PATH = f"{SEQ_NAME}_mean_tracks.npy"
VAR_PATH = f"{SEQ_NAME}_var_tracks.npy"
VIS_PATH = f"{SEQ_NAME}_vis_tracks.npy"
RAW_DIR = "Tracking_Rigid_Training/Training/OP1/Raw"
OUT_DIR = "tracking_vis_temporal"

def create_temporal_frames(mean_trajs, var_trajs, vis_trajs, frame_paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    analyzer = TemporalAnalyzer()
    
    # Pre-calculate metrics
    smoothness = analyzer.calculate_smoothness(mean_trajs) # (N,)
    strain_map = analyzer.estimate_deformation_strain(mean_trajs) # (T, N)
    occlusion_events = analyzer.detect_occlusion_events(vis_trajs)
    
    T, N, _ = mean_trajs.shape
    
    for t in range(T):
        img = cv2.imread(frame_paths[t])
        frame_strain = np.mean(strain_map[t])
        
        # Header Info
        cv2.putText(img, f"Frame: {t} | Avg Strain: {frame_strain:.4f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for n in range(N):
            x, y = mean_trajs[t, n]
            v = vis_trajs[t, n]
            s = smoothness[n]
            st = strain_map[t, n]
            
            # 1. Color by Smoothness (Green=Stable, Red=Jittery)
            # Use HLS or simple RGB interpolation
            # Smoothness is 0-1. 1.0 (stable) -> Green (0, 255, 0), 0.0 (jittery) -> Red (0, 0, 255)
            color = (0, int(255 * s), int(255 * (1-s))) # BGR: 1.0 -> (0, 255, 0), 0.0 -> (0, 0, 255)
            
            # 2. Indicate Occlusion
            if v < 0.5:
                # Point is lost/occluded
                cv2.drawMarker(img, (int(x), int(y)), (50, 50, 50), cv2.MARKER_TILTED_CROSS, 10, 2)
                continue # Don't draw halo for lost points
            
            # 3. Draw Point and Uncertainty Halo
            var = np.mean(var_trajs[t, n])
            std = np.sqrt(var)
            radius = int(3 + 2.0 * std)
            
            # 4. Deformation visualization (optional: circle thickness or size)
            # Let's use circle size for strain if it's significant
            strain_radius = int(radius + 10 * st)
            
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            cv2.circle(img, (int(x), int(y)), strain_radius, (255, 255, 0), 1) # Cyan ring for strain
            
        out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
        cv2.imwrite(out_path, img)

def create_video(frame_dir, video_out):
    frame_paths = natural_sort(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frame_paths: return
    first = cv2.imread(frame_paths[0])
    h, w, _ = first.shape
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))
    for f in frame_paths:
        out.write(cv2.imread(f))
    out.release()
    print(f"Video saved to {video_out}")

def main():
    if not os.path.exists(MEAN_PATH):
        print("Results not found. Run run_tracking.py first.")
        return
        
    mean = np.load(MEAN_PATH)
    var = np.load(VAR_PATH)
    vis = np.load(VIS_PATH)
    frame_paths = natural_sort(glob.glob(os.path.join(RAW_DIR, "*.png")))[:mean.shape[0]]
    
    print("Generating Temporal Intelligence Visualization...")
    create_temporal_frames(mean, var, vis, frame_paths, OUT_DIR)
    create_video(OUT_DIR, f"{SEQ_NAME}_temporal_tracking.mp4")

if __name__ == "__main__":
    main()
