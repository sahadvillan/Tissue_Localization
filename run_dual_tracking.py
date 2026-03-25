import os
import torch
import numpy as np
from tissue_localizer import TissueLocalizer
import glob

# Paths
CHECKPOINT = "checkpoints/scaled_online.pth"
RAW_DIR = "Tracking_Rigid_Training/Training/OP1/Raw"
MASK_PATH = "Tracking_Rigid_Training/Training/OP1/Masks/img_00_instrument.png"

def run_tracking_mode(mode_name, use_mc_dropout, n_samples):
    print(f"\n--- Running {mode_name} ---")
    localizer = TissueLocalizer(CHECKPOINT, device="cpu")
    
    # Sample Points
    points = localizer.sample_tissue_points(MASK_PATH, num_points=50)
    frame_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))
    
    # Track
    mean_tracks, var_tracks = localizer.track_sequence(
        frame_paths, points, mc_samples=n_samples, use_mc_dropout=use_mc_dropout
    )
    
    # Save results
    suffix = "mc" if use_mc_dropout else "std"
    np.save(f"op1_mean_tracks_{suffix}.npy", mean_tracks)
    np.save(f"op1_var_tracks_{suffix}.npy", var_tracks)
    print(f"Saved {mode_name} results.")

def main():
    # 1. Standard Tracking (No MC-Dropout)
    run_tracking_mode("Standard Tracking", use_mc_dropout=False, n_samples=1)
    
    # 2. MC-Dropout Tracking (N=10)
    run_tracking_mode("MC-Dropout Tracking", use_mc_dropout=True, n_samples=10)

if __name__ == "__main__":
    main()
