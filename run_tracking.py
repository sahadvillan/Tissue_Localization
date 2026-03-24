import os
import torch
import numpy as np
from tissue_localizer import TissueLocalizer
import glob
from utils import natural_sort
from temporal_analyzer import TemporalAnalyzer

# Paths
SEQ_NAME = "op1"
CHECKPOINT = "checkpoints/scaled_online.pth"
RAW_DIR = "Tracking_Rigid_Training/Training/OP1/Raw"
MASK_PATH = "Tracking_Rigid_Training/Training/OP1/Masks/img_00_instrument.png"

def main():
    # 1. Initialize
    localizer = TissueLocalizer(CHECKPOINT, device="cpu")
    # No longer need to call enable_mc_dropout here if track_sequence handles it,
    # but we can do it via the wrapper if we want to be explicit.
    localizer.mc_wrapper.enable_mc_dropout()

    # 2. Sample Points
    points = localizer.sample_tissue_points(MASK_PATH, num_points=50)
    print(f"Sampled {len(points)} points.")

    # 3. Prepare Sequence (Preview: first 200 frames)
    frame_paths = natural_sort(glob.glob(os.path.join(RAW_DIR, "*.png")))[:200]
    print(f"Tracking across {len(frame_paths)} frames...")

    # 4. Track with MC-Dropout (N=3 for preview)
    mean_tracks, var_tracks, vis_tracks = localizer.track_sequence(frame_paths, points, mc_samples=3)
    
    print("Tracking complete.")
    print(f"Mean trajectory shape: {mean_tracks.shape}")
    print(f"Variance shape: {var_tracks.shape}")
    print(f"Visibility shape: {vis_tracks.shape}")

    # 5. Temporal Analysis
    analyzer = TemporalAnalyzer()
    metrics = analyzer.get_temporal_metrics_summary(mean_tracks, vis_tracks)
    print("\n--- Temporal Intelligence Report ---")
    print(f"Mean Smoothness Score: {metrics['mean_smoothness']:.4f}")
    print(f"Mean Visibility Ratio: {metrics['mean_visibility']:.4f}")
    print(f"Total Occlusion Events: {metrics['total_occlusion_events']}")
    print(f"Mean Deformation (Strain): {metrics['mean_strain']:.4f}")

    # 6. Save results
    np.save(f"{SEQ_NAME}_mean_tracks.npy", mean_tracks)
    np.save(f"{SEQ_NAME}_var_tracks.npy", var_tracks)
    np.save(f"{SEQ_NAME}_vis_tracks.npy", vis_tracks)
    print(f"Saved trajectories and metrics to .npy files.")

if __name__ == "__main__":
    main()
