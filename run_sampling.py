import os
from tissue_localizer import TissueLocalizer
import torch

# Paths
CHECKPOINT = "checkpoints/scaled_online.pth"
OP1_RAW = "Tracking_Rigid_Training/Training/OP1/Raw/img_00_raw.png"
OP1_MASK = "Tracking_Rigid_Training/Training/OP1/Masks/img_00_instrument.png"
VIS_OUT = "sampled_points_op1.png"

def main():
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        return

    # Initialize
    localizer = TissueLocalizer(CHECKPOINT, device="cpu")
    
    # 1. Enable UQ (optional for sampling, but good to test)
    localizer.enable_mc_dropout(dropout_p=0.1)

    # 2. Sample Points
    print(f"Sampling points from {OP1_RAW} using mask {OP1_MASK}...")
    try:
        points = localizer.sample_tissue_points(OP1_MASK, num_points=50)
        print(f"Successfully sampled {len(points)} points.")
        
        # 3. Visualize
        localizer.visualize_sampling(OP1_RAW, points, VIS_OUT)
        print(f"Check {VIS_OUT} for the sampled points.")
        
    except Exception as e:
        print(f"Error during sampling: {e}")

if __name__ == "__main__":
    main()
