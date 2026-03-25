import os
import glob
import cv2
import numpy as np
from surgical_phantom_layer import SurgicalPhantomLayer

def main():
    # 1. Setup paths
    RAW_DIR = "Tracking_Rigid_Training/Training/OP1/Raw"
    INSTRUMENT_CSV = "Tracking_Rigid_Training/Training/OP1/Instruments_OP1.csv"
    CHECKPOINT = "checkpoints/scaled_online.pth"
    OUTPUT_DIR = "phantom_results"
    
    if not os.path.exists(CHECKPOINT):
        print(f"Error: Checkpoint {CHECKPOINT} not found.")
        return

    # 2. Initialize Layer
    phantom_layer = SurgicalPhantomLayer(CHECKPOINT)
    
    # 3. Prepare frame paths
    frame_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))
    # Limit for demo purposes or process all
    frame_paths = frame_paths[:200] 
    
    # 4. Track Instrument Tips
    # tracks: (T, N, 2), vis: (T, N), oris: (T, N, 2)
    tracks, vis, oris = phantom_layer.track_tips(frame_paths, INSTRUMENT_CSV)
    
    # Save for Interaction Analysis (Module F)
    np.save("op1_instrument_tracks.npy", tracks)
    np.save("op1_instrument_vis.npy", vis)
    np.save("op1_instrument_oris.npy", oris)
    print("Instrument tracks saved for Interaction Analysis.")
    
    # 5. Generate Phantom Visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Phantom Visualization in {OUTPUT_DIR}...")
    
    T, N, _ = tracks.shape
    for t in range(T):
        img = cv2.imread(frame_paths[t])
        
        # Add a subtle HUD for "Phantom Mode"
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(img, "DA VINCI AUGMENTED REALITY | PHANTOM MODE ACTIVE", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        for n in range(N):
            pos = tracks[t, n]
            ori = oris[t, n]
            v = vis[t, n]
            
            # Draw the phantom indicator
            phantom_layer.draw_phantom_indicator(img, pos, ori, v)
            
            # Label based on visibility
            status = "VISIBLE" if v > 0.5 else "PHANTOM (HIDDEN)"
            color = (0, 255, 0) if v > 0.5 else (0, 0, 255)
            cv2.putText(img, f"TIP {n+1}: {status}", (20, 80 + n*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        out_path = os.path.join(OUTPUT_DIR, f"frame_{t:04d}.png")
        cv2.imwrite(out_path, img)
        if t % 20 == 0:
            print(f"  Exported frame {t}/{T}...")

    # 6. Create Video
    # Use ffmpeg or simple cv2 writer
    print("Exporting video...")
    video_out = "phantom_view_op1.mp4"
    img0 = cv2.imread(os.path.join(OUTPUT_DIR, "frame_0000.png"))
    h, w, _ = img0.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, 30.0, (w, h))
    
    for t in range(T):
        writer.write(cv2.imread(os.path.join(OUTPUT_DIR, f"frame_{t:04d}.png")))
    writer.release()
    
    print(f"Final video saved: {video_out}")

if __name__ == "__main__":
    main()
