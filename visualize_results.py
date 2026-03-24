import numpy as np
import cv2
import os
import glob
from temporal_analyzer import TemporalAnalyzer

# Paths
MEAN_PATH = "op1_mean_tracks.npy"
VAR_PATH = "op1_var_tracks.npy"
RAW_DIR = "Tracking_Rigid_Training/Training/OP1/Raw"
OUT_DIR = "tracking_vis"

def main():
    if not os.path.exists(MEAN_PATH):
        print("Results not found yet.")
        return
        
    mean_tracks = np.load(MEAN_PATH) # (T, N, 2)
    var_tracks = np.load(VAR_PATH)   # (T, N, 2)
    vis_tracks = np.load("op1_vis_tracks.npy") if os.path.exists("op1_vis_tracks.npy") else np.ones(mean_tracks.shape[:2])
    
    # 2. Analyze for Augmentation
    analyzer = TemporalAnalyzer()
    # Load masks if they exist for better occlusion markers
    mask_dir = "Tracking_Rigid_Training/Training/OP1/Masks"
    mask_seq = None
    if os.path.exists(mask_dir):
        mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        if mask_paths:
            m0 = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
            mask_seq = np.zeros((len(mask_paths), m0.shape[0], m0.shape[1]), dtype=np.uint8)
            for i, p in enumerate(mask_paths[:mean_tracks.shape[0]]):
                mask_seq[i] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    _, state_mask = analyzer.detect_occlusion_events(mean_tracks, vis_tracks, mask_seq)
    motion_vectors = analyzer.compute_motion_vectors(mean_tracks, vis_tracks) # (T-1, N, 2)
    strain = analyzer.estimate_deformation_strain(mean_tracks)
    
    # Uncertainty as sum of variances
    var_scalar = np.sum(var_tracks, axis=-1) 
    
    os.makedirs(OUT_DIR, exist_ok=True)
    frame_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))[:mean_tracks.shape[0]]
    
    # K-factor for uncertainty halo (e.g., 2 standard deviations)
    K = 3.0
    
    for t, frame_path in enumerate(frame_paths):
        img = cv2.imread(frame_path)
        for n in range(mean_tracks.shape[1]):
            x, y = mean_tracks[t, n]
            var = var_scalar[t, n]
            std = np.sqrt(var)
            state = state_mask[t, n]
            s_val = strain[t, n]
            
            # Color coding based on state
            if state == 0: color = (0, 255, 0)      # Visible: Green
            elif state == 1: color = (0, 255, 255)  # Tool-Occluded: Yellow
            else: color = (0, 0, 255)               # Lost/Other: Red
            
            # Draw mean point
            cv2.circle(img, (int(x), int(y)), 3, color, -1)
            
            # Draw uncertainty halo
            radius = int(3 + K * std)
            cv2.circle(img, (int(x), int(y)), radius, color, 1)
            
            # Draw motion vector arrow (if t < T-1 and visible)
            if t < mean_tracks.shape[0] - 1 and vis_tracks[t, n] > 0.5:
                mv = motion_vectors[t, n]
                if np.linalg.norm(mv) > 0.5: # Only draw significant motion
                    end_pt = (int(x + mv[0] * 5), int(y + mv[1] * 5)) # Scale for visibility
                    cv2.arrowedLine(img, (int(x), int(y)), end_pt, (255, 0, 0), 1, tipLength=0.3)
                    
        # Add labels
        cv2.putText(img, f"Frame: {t}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Green: Visible | Yellow: Tool | Red: Lost", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        out_path = os.path.join(OUT_DIR, f"frame_{t:04d}.png")
        cv2.imwrite(out_path, img)
        if t % 10 == 0:
            print(f"Processed {t}/{len(frame_paths)} frames.")
            
    print(f"Visualization frames saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
