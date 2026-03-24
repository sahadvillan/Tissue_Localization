import cv2
import os
import glob

# Paths
FRAME_DIR = "tracking_vis"
VIDEO_OUT = "op1_augmented_analysis.mp4"

def main():
    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    if not frame_paths:
        print("No frames found.")
        return
        
    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, 10.0, (w, h))
    
    for f in frame_paths:
        img = cv2.imread(f)
        out.write(img)
        
    out.release()
    print(f"Video saved to {VIDEO_OUT}")

if __name__ == "__main__":
    main()
