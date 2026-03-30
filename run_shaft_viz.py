"""
run_shaft_viz.py
================
NEW approach: derive instrument orientation per-frame from CoTracker
tracking of both the tip AND a secondary shaft point.

This is separate from run_phantom_viz.py (CSV-based orientation).
Output goes to: shaft_results/   and   shaft_view_op1.mp4
"""
import os
import re
import glob
import cv2
import numpy as np
from surgical_phantom_layer import SurgicalPhantomLayer


def main():
    RAW_DIR        = "Tracking_Rigid_Training/Training/OP1/Raw"
    INSTRUMENT_CSV = "Tracking_Rigid_Training/Training/OP1/Instruments_OP1.csv"
    CHECKPOINT     = "checkpoints/scaled_online.pth"
    OUTPUT_DIR     = "shaft_results"

    if not os.path.exists(CHECKPOINT):
        print(f"Error: Checkpoint not found at {CHECKPOINT}")
        return

    phantom_layer = SurgicalPhantomLayer(CHECKPOINT)

    # Numerically sorted frames
    frame_paths = sorted(
        glob.glob(os.path.join(RAW_DIR, "*.png")),
        key=lambda x: int(re.search(r"(\d+)", os.path.basename(x)).group(1))
    )
    frame_paths = frame_paths[:200]

    # --- NEW: shaft-tracking orientation ---
    tip_tracks, tip_vis, shaft_oris = phantom_layer.track_with_shaft_orientation(
        frame_paths, INSTRUMENT_CSV, shaft_offset=40
    )

    # Save for optional downstream analysis
    np.save("op1_shaft_tip_tracks.npy", tip_tracks)
    np.save("op1_shaft_vis.npy",        tip_vis)
    np.save("op1_shaft_oris.npy",       shaft_oris)
    print("Shaft orientation tracks saved.")

    # Render frames
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Rendering to {OUTPUT_DIR}...")
    T, N, _ = tip_tracks.shape

    for t in range(T):
        img = cv2.imread(frame_paths[t])

        for n in range(N):
            pos = tip_tracks[t, n]
            ori = shaft_oris[t, n]     # <-- per-frame shaft-derived orientation
            v   = tip_vis[t, n]
            phantom_layer.draw_phantom_indicator(img, pos, ori, v)

            status = "VISIBLE" if v > 0.5 else "PHANTOM"
            color  = (0, 255, 0) if v > 0.5 else (0, 0, 255)
            cv2.putText(img, f"TIP {n+1}: {status} [SHAFT-ORI]",
                        (20, 30 + n * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{t:04d}.png"), img)
        if t % 20 == 0:
            print(f"  Frame {t}/{T}")

    # Export video
    print("Exporting video...")
    img0 = cv2.imread(os.path.join(OUTPUT_DIR, "frame_0000.png"))
    h, w, _ = img0.shape
    writer = cv2.VideoWriter(
        "shaft_view_op1.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0, (w, h)
    )
    for t in range(T):
        writer.write(cv2.imread(os.path.join(OUTPUT_DIR, f"frame_{t:04d}.png")))
    writer.release()
    print("Done → shaft_view_op1.mp4")


if __name__ == "__main__":
    main()
