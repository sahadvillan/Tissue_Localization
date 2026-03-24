import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from temporal_analyzer import TemporalAnalyzer

def main():
    # 1. Load data
    mean_tracks_path = "op1_mean_tracks.npy"
    vis_tracks_path = "op1_vis_tracks.npy"
    
    if not os.path.exists(mean_tracks_path) or not os.path.exists(vis_tracks_path):
        print("Data files not found. Please run tracking first.")
        return
        
    trajectories = np.load(mean_tracks_path)
    visibility = np.load(vis_tracks_path)
    T, N, _ = trajectories.shape
    
    # Load Masks for occlusion analysis
    mask_dir = "Tracking_Rigid_Training/Training/OP1/Masks"
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])[:T]
    
    mask_seq = None
    if mask_paths:
        print(f"Loading {len(mask_paths)} masks...")
        # Load and resize/convert if needed. CoTracker usually works with 512x880 or similar.
        # We just need them to match the trajectory space.
        sample_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
        mask_seq = np.zeros((len(mask_paths), sample_mask.shape[0], sample_mask.shape[1]), dtype=np.uint8)
        for i, p in enumerate(mask_paths):
            mask_seq[i] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            
    # 2. Analyze
    analyzer = TemporalAnalyzer()
    smoothness = analyzer.calculate_smoothness(trajectories)
    drift = analyzer.calculate_drift(trajectories)
    strain = analyzer.estimate_deformation_strain(trajectories)
    _, state_mask = analyzer.detect_occlusion_events(trajectories, visibility, mask_seq)
    
    # 3. Plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.suptitle(f"Tissue Localization Enhanced Analysis: OP1 ({N} points, {T} frames)", fontsize=16)
    
    # Plot A: Smoothness Distribution
    axes[0, 0].bar(range(N), smoothness, color='skyblue')
    axes[0, 0].set_title("Trajectory Smoothness")
    axes[0, 0].set_ylabel("Smoothness Score")
    
    # Plot B: Mean Tissue Strain
    mean_strain = np.mean(strain, axis=1)
    axes[0, 1].plot(mean_strain, color='forestgreen', linewidth=2)
    axes[0, 1].set_title("Mean Tissue Strain over Time")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot C: Survival Rate
    survival = analyzer.calculate_survival_rate(visibility)
    axes[1, 0].plot(survival * 100, color='blue', linewidth=2)
    axes[1, 0].set_title("Track Survival Rate (%)")
    axes[1, 0].set_ylabel("% Visible Tracks")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot D: Cumulative Drift
    drift_profile = analyzer.calculate_drift_over_time(trajectories)
    mean_drift_profile = np.mean(drift_profile, axis=1)
    axes[1, 1].plot(mean_drift_profile, color='purple', linewidth=2)
    axes[1, 1].set_title("Average Cumulative Drift (pixels)")
    axes[1, 1].set_ylabel("Drift (px)")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot E: Occlusion Stats (Stackplot)
    tool_occluded_count = np.sum(state_mask == 1, axis=1)
    other_lost_count = np.sum(state_mask == 2, axis=1)
    axes[2, 0].stackplot(range(T), [tool_occluded_count, other_lost_count], 
                        labels=['Tool Occluded', 'Lost Other'], colors=['#ff7f0e', '#d62728'], alpha=0.7)
    axes[2, 0].set_title("Occlusion/Disappearance Events")
    axes[2, 0].legend(loc='upper left')
    
    # Plot F: Drift Distribution (Scatter)
    axes[2, 1].scatter(range(N), drift, c=drift, cmap='viridis')
    axes[2, 1].set_title("Net Drift per Point (Final Displacement)")
    axes[2, 1].set_ylabel("px")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = "temporal_analysis_dashboard.png"
    plt.savefig(out_path)
    print(f"Analysis dashboard saved to {out_path}")
    
    # Also print summary
    summary = analyzer.get_temporal_metrics_summary(trajectories, visibility)
    print("\n--- Clinical Metrics Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    main()
