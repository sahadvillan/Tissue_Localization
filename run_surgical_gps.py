"""
run_surgical_gps.py
===================
Unified Clinical Pipeline: "Surgical GPS"
Combines:
  1. Dense Tissue Tracking (with uncertainty halos and strain vectors)
  2. Instrument Tip & Shaft Tracking (Phantom View AR with orientation)
Outputs a single, fully augmented surgical video.
"""
import os
import re
import glob
import cv2
import numpy as np
from tissue_localizer import TissueLocalizer
from surgical_phantom_layer import SurgicalPhantomLayer
from temporal_analyzer import TemporalAnalyzer

def main():
    # --- Configuration ---
    SEQ_NAME       = "op1"
    CHECKPOINT     = "checkpoints/scaled_online.pth"
    RAW_DIR        = "Tracking_Rigid_Training/Training/OP1/Raw"
    MASK_PATH      = "Tracking_Rigid_Training/Training/OP1/Masks/img_00_instrument.png"
    INSTRUMENT_CSV = "Tracking_Rigid_Training/Training/OP1/Instruments_OP1.csv"
    OUTPUT_DIR     = "surgical_gps_results"
    MAX_FRAMES     = 200

    if not os.path.exists(CHECKPOINT):
        print(f"Error: Checkpoint {CHECKPOINT} not found.")
        return

    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame_paths = sorted(
        glob.glob(os.path.join(RAW_DIR, "*.png")),
        key=lambda x: int(re.search(r"(\d+)", os.path.basename(x)).group(1))
    )[:MAX_FRAMES]
    T = len(frame_paths)

    # 1. Initialize localizers (we can use one for both, but wrapper handles dropout state)
    print("=" * 50)
    print("SURGICAL GPS INITIALIZATION")
    print("=" * 50)
    
    localizer = TissueLocalizer(CHECKPOINT, device="cuda" if torch.cuda.is_available() else "cpu")
    phantom_layer = SurgicalPhantomLayer(CHECKPOINT)
    analyzer = TemporalAnalyzer()

    # ==========================================================
    # PHASE 1: TISSUE TRACKING (Standard, no MC-Dropout)
    # ==========================================================
    print("\n[PHASE 1] Tracking Surgical Tissue (Standard Mode)...")
    
    tissue_points = localizer.sample_tissue_points(MASK_PATH, num_points=50)
    tissue_mean, tissue_var, tissue_vis = localizer.track_sequence(frame_paths, tissue_points, mc_samples=1)
    
    # Analyze tissue mechanics
    state_mask = analyzer.detect_occlusion_events(tissue_mean, tissue_vis)[1]
    motion_vectors = analyzer.compute_motion_vectors(tissue_mean, tissue_vis)
    var_scalar = np.sum(tissue_var, axis=-1)

    # ==========================================================
    # PHASE 2: INSTRUMENT TRACKING (Shaft Method)
    # ==========================================================
    print("\n[PHASE 2] Tracking Instruments (Self-Supervised Shaft Orientation)...")
    # Disable dropout for stable instrument tracking
    phantom_layer.localizer.mc_wrapper.disable_mc_dropout()
    
    tip_tracks, tip_vis, shaft_oris = phantom_layer.track_with_shaft_orientation(
        frame_paths, INSTRUMENT_CSV, shaft_offset=40
    )

    # ==========================================================
    # PHASE 3: UNIFIED RENDERING ("SURGICAL GPS")
    # ==========================================================
    print("\n[PHASE 3] Rendering Unified Surgical GPS AR Overlay...")
    N_inst = tip_tracks.shape[1]
    N_tiss = tissue_mean.shape[1]

    for t in range(T):
        img = cv2.imread(frame_paths[t])

        # A. Draw Tissue Points & Uncertainty
        for n in range(N_tiss):
            x, y = tissue_mean[t, n]
            std = np.sqrt(var_scalar[t, n])
            state = state_mask[t, n]
            vis = tissue_vis[t, n]
            
            if state == 0: color = (0, 255, 0)      # Visible: Green
            elif state == 1: color = (0, 255, 255)  # Occluded: Yellow
            else: color = (0, 0, 255)               # Lost: Red
                
            cv2.circle(img, (int(x), int(y)), 2, color, -1)
            cv2.circle(img, (int(x), int(y)), int(3 + 3.0 * std), color, 1) # Halo
            
            # Short Motion Vectors
            if t < T - 1 and vis > 0.5:
                mv = motion_vectors[t, n]
                if np.linalg.norm(mv) > 0.5:
                    end_pt = (int(x + mv[0]*4), int(y + mv[1]*4))
                    cv2.arrowedLine(img, (int(x), int(y)), end_pt, (255, 255, 0), 1, tipLength=0.3)

        # B. Draw Instrument Phantom AR
        for k in range(N_inst):
            pos = tip_tracks[t, k]
            ori = shaft_oris[t, k]
            v = tip_vis[t, k]
            
            phantom_layer.draw_phantom_indicator(img, pos, ori, v)
            
            status = "VISIBLE" if v > 0.5 else "PHANTOM"
            c = (0, 255, 0) if v > 0.5 else (0, 0, 255)
            cv2.putText(img, f"INSTRUMENT {k+1}: {status}", 
                        (20, 30 + k*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)

        out_path = os.path.join(OUTPUT_DIR, f"frame_{t:04d}.png")
        cv2.imwrite(out_path, img)

        if t % 20 == 0:
            print(f"  Rendered frame {t}/{T}")

    # ==========================================================
    # PHASE 4: VIDEO EXPORT
    # ==========================================================
    print("\n[PHASE 4] Exporting Final Video...")
    img0 = cv2.imread(os.path.join(OUTPUT_DIR, "frame_0000.png"))
    h, w, _ = img0.shape
    video_out = "surgical_gps_op1.mp4"
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))

    for t in range(T):
        writer.write(cv2.imread(os.path.join(OUTPUT_DIR, f"frame_{t:04d}.png")))
    writer.release()

    print(f"\nSUCCESS! Surgical GPS AR Video saved to: {video_out}")


if __name__ == "__main__":
    import torch # import here for the main config check
    main()
