"""
evaluate_orientation.py
=======================
An objective comparison between two instrument orientation methods:
1. Phantom Method: CSV ground truth, carried forward (stale between labels).
2. Shaft Method: CoTracker predicting tip and shaft points per-frame.

Metrics calculated:
- Absolute angular error at Ground Truth (CSV) frames (degrees)
- Maximum inter-frame "Snap" (degrees) - measures staleness
- Average Angular Velocity (degrees/frame) - measures smoothness vs jitter
"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

def angular_difference(v1, v2):
    """Returns absolute angle in degrees between two 2D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def main():
    print("Loading data...")
    phantom_oris = np.load('op1_instrument_oris.npy')  # (T, N, 2)
    shaft_oris = np.load('op1_shaft_oris.npy')         # (T, N, 2)
    
    df = pd.read_csv('Tracking_Rigid_Training/Training/OP1/Instruments_OP1.csv', header=None)
    
    # 1. Parse all Ground Truth
    gt_frames = []
    gt_oris = {} # frame -> list of [v1, v2]
    
    for _, row in df.iterrows():
        fnum = int(re.search(r'(\d+)', str(row.iloc[0])).group(1))
        if fnum < 200:
            gt_frames.append(fnum)
            
            vx1, vy1 = float(row.iloc[3]), float(row.iloc[4])
            frame_gt = [[vx1, vy1]]
            
            if len(row) > 8 and not pd.isna(row.iloc[7]):
                vx2, vy2 = float(row.iloc[7]), float(row.iloc[8])
                frame_gt.append([vx2, vy2])
            else:
                frame_gt.append([0.0, 0.0])
                
            gt_oris[fnum] = np.array(frame_gt)

    T, N, _ = phantom_oris.shape
    
    print("\n==============================================")
    print("   INSTRUMENT ORIENTATION EVALUATION REPORT   ")
    print("==============================================\n")

    for n in range(N):
        print(f"--- TIP {n+1} ---")
        
        # Metric 1: Angular Error at GT Frames
        shaft_errors = []
        phantom_errors = []
        
        for t in gt_frames:
            gt_v = gt_oris[t][n]
            if np.linalg.norm(gt_v) < 1e-4: continue
            
            shaft_errors.append(angular_difference(shaft_oris[t, n], gt_v))
            phantom_errors.append(angular_difference(phantom_oris[t, n], gt_v))

        print("1. Accuracy at Ground Truth frames (Mean Angular Error):")
        print(f"   Phantom (CSV) : {np.mean(phantom_errors):>5.1f}° ± {np.std(phantom_errors):>4.1f}°  <- (Note: This is inherently 0 because it copies GT)")
        print(f"   Shaft (Track) : {np.mean(shaft_errors):>5.1f}° ± {np.std(shaft_errors):>4.1f}°")

        # Metric 2: Staleness ("Snap" when GT updates)
        phantom_snaps = []
        shaft_snaps = []
        
        for i in range(1, len(gt_frames)):
            curr_gt = gt_frames[i]
            prev_frame = curr_gt - 1
            if prev_frame < 0: continue
            
            # How much does the predicted vector change between t-1 and t(GT)?
            p_snap = angular_difference(phantom_oris[prev_frame, n], phantom_oris[curr_gt, n])
            s_snap = angular_difference(shaft_oris[prev_frame, n], shaft_oris[curr_gt, n])
            
            phantom_snaps.append(p_snap)
            shaft_snaps.append(s_snap)
            
        print("\n2. Staleness (Average sudden angle 'snap' when new GT arrives):")
        print(f"   Phantom (CSV) : {np.mean(phantom_snaps):>5.1f}° (Max jump: {max(phantom_snaps):.1f}°)")
        print(f"   Shaft (Track) : {np.mean(shaft_snaps):>5.1f}° (Max jump: {max(shaft_snaps):.1f}°)")

        # Metric 3: Smoothness (Average frame-to-frame change)
        p_smooth = [angular_difference(phantom_oris[t, n], phantom_oris[t-1, n]) for t in range(1, T)]
        s_smooth = [angular_difference(shaft_oris[t, n], shaft_oris[t-1, n]) for t in range(1, T)]
        
        print("\n3. Smoothness (Average frame-to-frame rotation):")
        print(f"   Phantom (CSV) : {np.mean(p_smooth):>5.2f}°/frame (Blocky, stale updates)")
        print(f"   Shaft (Track) : {np.mean(s_smooth):>5.2f}°/frame (Continuous movement)")
        print("\n")

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.title(f'Tip {n+1} Angle Change Over Time (Smoothness vs Snapping)')
        plt.plot(p_smooth, label='Phantom (CSV)', drawstyle='steps-mid', alpha=0.7)
        plt.plot(s_smooth, label='Shaft (Tracked)', alpha=0.7)
        for gt in gt_frames:
            plt.axvline(x=gt, color='r', linestyle='--', alpha=0.2)
        plt.ylabel('Change in Angle (Degrees)')
        plt.xlabel('Frame')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'orientation_comparison_tip{n+1}.png')
        
    print("Plots saved as orientation_comparison_tip1.png and tip2.png")

if __name__ == "__main__":
    main()
