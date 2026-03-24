import numpy as np
import os
from temporal_analyzer import TemporalAnalyzer
import cv2

def main():
    # 1. Load existing data if available, or create synthetic data
    mean_tracks_path = "op1_mean_tracks.npy"
    vis_tracks_path = "op1_vis_tracks.npy"
    
    if os.path.exists(mean_tracks_path) and os.path.exists(vis_tracks_path):
        print(f"Loading real data from {mean_tracks_path}...")
        trajectories = np.load(mean_tracks_path)
        visibility = np.load(vis_tracks_path)
    else:
        print("Real data not found, creating synthetic test data...")
        # (T=50, N=5, 2)
        T, N = 50, 5
        trajectories = np.zeros((T, N, 2))
        visibility = np.ones((T, N))
        
        for n in range(N):
            # Base position
            start_pos = np.array([100 + n*50, 100 + n*50])
            # Linear motion + noise
            for t in range(T):
                trajectories[t, n] = start_pos + t * 2 + np.random.normal(0, 0.5, 2)
            
        # Add an outlier (unstable track)
        trajectories[:, 0] += np.random.normal(0, 10, (T, 2)) 
        
        # Add a "drift" track
        trajectories[:, 1] += np.linspace(0, 150, T)[:, None]
        
        # Add an occlusion
        visibility[20:30, 2] = 0.0
        
    # 2. Initialize Analyzer
    analyzer = TemporalAnalyzer()
    
    # 3. Test Module A: Consistency & Rejection
    print("\n--- Testing Module A: Consistency ---")
    valid_mask = analyzer.filter_trajectories(trajectories, visibility, smoothness_thresh=0.5, drift_thresh=100.0)
    smoothness = analyzer.calculate_smoothness(trajectories)
    drift = analyzer.calculate_drift(trajectories)
    
    for i in range(len(valid_mask)):
        status = "VALID" if valid_mask[i] else "REJECTED"
        print(f"Point {i}: Smoothness={smoothness[i]:.4f}, Drift={drift[i]:.2f} -> {status}")

    # 4. Test Module B: Occlusion (Synthetic tool mask)
    print("\n--- Testing Module B: Occlusion ---")
    T, H, W = trajectories.shape[0], 512, 512
    # Create dummy mask sequence where tool is a rectangle in the middle
    mask_seq = np.zeros((T, H, W), dtype=np.uint8)
    mask_seq[:, 200:300, 200:300] = 255
    
    events, state_mask = analyzer.detect_occlusion_events(trajectories, visibility, mask_seq)
    for i, e in enumerate(events):
        print(f"Point {i}: Lost {len(e['lost_at'])} times, Tool Occ: {e['tool_occlusions']}, Other: {e['other_losses']}")

    # 5. Test Module C: Deformation
    print("\n--- Testing Module C: Deformation ---")
    strain = analyzer.estimate_deformation_strain(trajectories)
    motion = analyzer.compute_motion_vectors(trajectories, visibility)
    
    print(f"Mean Strain: {np.mean(strain):.4f}")
    print(f"Motion Vector Shape: {motion.shape}")
    
    # 6. Summary
    summary = analyzer.get_temporal_metrics_summary(trajectories, visibility, mask_seq)
    print("\n--- Overall Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
