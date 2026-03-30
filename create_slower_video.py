import cv2
import os
import glob

# Paths
FRAME_DIR = "surgical_gps_results"
VIDEO_OUT = "surgical_gps_op1_slow.mp4"
FPS = 5.0  # Slower version (previous was 15)

def main():
    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    if not frame_paths:
        print(f"No frames found in {FRAME_DIR}.")
        return
        
    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, FPS, (w, h))
    
    for i, f in enumerate(frame_paths):
        img = cv2.imread(f)
        out.write(img)
        if i % 50 == 0:
            print(f"Processed {i}/{len(frame_paths)} frames...")
        
    out.release()
    print(f"Video saved to {VIDEO_OUT} at {FPS} FPS")

if __name__ == "__main__":
    main()
