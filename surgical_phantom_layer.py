import os
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import torchvision.transforms as T_trans
from tissue_localizer import TissueLocalizer

class SurgicalPhantomLayer:
    def __init__(self, checkpoint_path="checkpoints/scaled_online.pth"):
        self.localizer = TissueLocalizer(checkpoint_path)
        self.device = self.localizer.device

    def track_tips(self, frame_paths, csv_path):
        """Track instrument tips from CSV across all frames."""
        print(f"Tracking tips from {csv_path}...")
        df = pd.read_csv(csv_path, header=None)
        T = len(frame_paths)
        
        # 1. Get initial query points from the first frame in the CSV
        first_frame_name = df.iloc[0, 0]
        first_idx = next(i for i, f in enumerate(frame_paths) if os.path.basename(f) == first_frame_name)
        
        initial_points = []
        initial_oris = []
        
        # Tip 1
        initial_points.append([float(df.iloc[0, 1]), float(df.iloc[0, 2])])
        initial_oris.append([float(df.iloc[0, 3]), float(df.iloc[0, 4])])
        # Tip 2 if exists
        if len(df.columns) > 8 and not pd.isna(df.iloc[0, 5]):
            initial_points.append([float(df.iloc[0, 5]), float(df.iloc[0, 6])])
            initial_oris.append([float(df.iloc[0, 7]), float(df.iloc[0, 8])])
            
        initial_points = np.array(initial_points, dtype=np.float32)
        initial_oris = np.array(initial_oris, dtype=np.float32)
        N = len(initial_points)
        
        # 2. Load Video Tensor
        imgs = []
        for f in frame_paths:
            img = Image.open(f).convert("RGB")
            imgs.append(T_trans.ToTensor()(img))
        video_tensor = torch.stack(imgs).unsqueeze(0).to(self.device)
        
        # 3. Track with CoTracker (Chunked to avoid OOM)
        # We'll use a sliding window approach
        block_size = 50
        overlap = 16
        
        all_pred_tracks = np.zeros((T, N, 2))
        all_pred_vis = np.zeros((T, N))
        
        # Initial queries
        curr_queries = initial_points.copy()
        start = first_idx
        
        while start < T:
            end = min(start + block_size, T)
            print(f"  Processing phantom block {start}-{end}...")
            
            chunk_imgs = []
            for i in range(start, end):
                img = Image.open(frame_paths[i]).convert("RGB")
                chunk_imgs.append(T_trans.ToTensor()(img))
            video_chunk = torch.stack(chunk_imgs).unsqueeze(0).to(self.device).float()
            
            queries = torch.zeros((1, N, 3), device=self.device)
            # queries[0, :, 0] is 0 relative to this chunk
            queries[0, :, 1:] = torch.from_numpy(curr_queries).to(self.device)
            
            with torch.no_grad():
                tracks, vis = self.localizer.model(video_chunk, queries=queries, grid_size=0)
                tracks_np = tracks.cpu().numpy()[0]
                vis_np = vis.cpu().numpy()[0]
                
                # Copy to results
                actual_len = tracks_np.shape[0]
                all_pred_tracks[start:start+actual_len] = tracks_np
                all_pred_vis[start:start+actual_len] = vis_np
                
                if end < T:
                    # Update queries for next chunk from the overlap point
                    rel_idx = (end - start) - overlap
                    curr_queries = tracks_np[rel_idx]
                    start = end - overlap
                else:
                    break
            
            del video_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        tracks_np = all_pred_tracks
        vis_np = all_pred_vis
        
        # 4. Handle Orientation (carrying forward/interpolating)
        # We'll use CSV orientation when available, otherwise last known.
        oriented_tracks = np.zeros((T, N, 2))
        for n in range(N):
            curr_ori = initial_oris[n]
            for t in range(T):
                f_name = os.path.basename(frame_paths[t])
                row = df[df[0] == f_name]
                if not row.empty:
                    if n == 0:
                        curr_ori = [float(row.iloc[0][3]), float(row.iloc[0][4])]
                    elif len(row.columns) > 8:
                        curr_ori = [float(row.iloc[0][7]), float(row.iloc[0][8])]
                oriented_tracks[t, n] = curr_ori
                
        return tracks_np, vis_np, oriented_tracks

    def draw_phantom_indicator(self, img, pos, orientation, visibility):
        """
        Draws an oriented indicator. 
        Style changes based on visibility (Solid vs Phantom).
        """
        if np.isnan(pos).any() or np.isnan(orientation).any():
            return
            
        x, y = int(pos[0]), int(pos[1])
        vx, vy = orientation
        angle = np.arctan2(vy, vx)
        
        # Indicator parameters
        length = 25
        width = 15
        
        # Colors: Cyan for visible, Translucent Red for hidden
        if visibility > 0.5:
            color = (255, 255, 0) # Cyan-ish (BGR: 255, 255, 0 is Yellow, wait)
            color = (255, 255, 0) # Let's use Cyan (255, 255, 0 in BGR is Cyan if Blue is 255... wait)
            color = (255, 255, 0) # BGR Cyan is (255, 255, 0)
            line_style = cv2.LINE_AA
            thickness = 2
        else:
            color = (0, 0, 255) # Red for Phantom
            line_style = cv2.LINE_4 # Dotted effect via drawing logic
            thickness = 1
            
        # Create a "V" shape gripper outline
        # Tip point is (x, y). Two points back at angle +/- 30 deg
        p1 = (int(x - length * np.cos(angle - 0.5)), int(y - length * np.sin(angle - 0.5)))
        p2 = (int(x - length * np.cos(angle + 0.5)), int(y - length * np.sin(angle + 0.5)))
        
        if visibility > 0.5:
            cv2.line(img, (x, y), p1, color, thickness, line_style)
            cv2.line(img, (x, y), p2, color, thickness, line_style)
            # Add a glow circle
            cv2.circle(img, (x, y), 5, color, -1)
        else:
            # Phantom: Draw dashed lines manually
            self.draw_dashed_line(img, (x, y), p1, color, 1)
            self.draw_dashed_line(img, (x, y), p2, color, 1)
            cv2.circle(img, (x, y), 3, color, 1)

    def draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=5):
        dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start_dash = (
                int(pt1[0] + (pt2[0]-pt1[0]) * i / dashes),
                int(pt1[1] + (pt2[1]-pt1[1]) * i / dashes)
            )
            end_dash = (
                int(pt1[0] + (pt2[0]-pt1[0]) * (i+0.5) / dashes),
                int(pt1[1] + (pt2[1]-pt1[1]) * (i+0.5) / dashes)
            )
            cv2.line(img, start_dash, end_dash, color, thickness)

if __name__ == "__main__":
    import glob
    layer = SurgicalPhantomLayer()
    frames = sorted(glob.glob("Tracking_Rigid_Training/Training/OP1/Raw/*.png"))[:100]
    csv = "Tracking_Rigid_Training/Training/OP1/Instruments_OP1.csv"
    
    tracks, vis, oris = layer.track_tips(frames, csv)
    
    os.makedirs("phantom_vis", exist_ok=True)
    for t, f in enumerate(frames):
        img = cv2.imread(f)
        for n in range(tracks.shape[1]):
            layer.draw_phantom_indicator(img, tracks[t, n], oris[t, n], vis[t, n])
            
        cv2.putText(img, f"PHANTOM VIEW - Frame {t}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite(f"phantom_vis/frame_{t:04d}.png", img)
        if t % 20 == 0: print(f"Saved phantom frame {t}")
