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
        
        # 2. Build a GT lookup: frame_name -> (x1,y1, x2,y2) when available
        gt_positions = {}  # fname -> np array (N, 2)
        for _, row in df.iterrows():
            fname = str(row.iloc[0])
            tip_pos = [float(row.iloc[1]), float(row.iloc[2])]
            tips = [tip_pos]
            if N == 2 and len(row) > 8 and not pd.isna(row.iloc[5]):
                tips.append([float(row.iloc[5]), float(row.iloc[6])])
            gt_positions[fname] = np.array(tips[:N], dtype=np.float32)
        
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

        # 3b. GT Re-Anchoring: override predictions with ground-truth positions
        # when CSV data is available for that frame. Mark those frames as 'visible'.
        for t, fpath in enumerate(frame_paths):
            fname = os.path.basename(fpath)
            if fname in gt_positions:
                tracks_np[t] = gt_positions[fname]  # lock to GT
                vis_np[t] = 1.0                      # GT frame = always visible
        print(f"  GT re-anchoring applied to {len(gt_positions)} frames out of {T}.")
        
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

    def track_with_shaft_orientation(self, frame_paths, csv_path, shaft_offset=40):
        """
        NEW METHOD: Track tip + shaft point per instrument.
        Orientation derived per-frame from (tip - shaft) direction — no CSV staleness.

        Args:
            frame_paths:   sorted list of frame file paths
            csv_path:      Instruments CSV (initial positions + GT re-anchoring)
            shaft_offset:  pixels back along shaft to place secondary tracking point

        Returns:
            tip_tracks  (T, N, 2)  — GT-anchored tip positions
            tip_vis     (T, N)     — CoTracker visibility per tip
            shaft_oris  (T, N, 2)  — per-frame normalized (vx,vy) from shaft geometry
        """
        print(f"[ShaftTrack] Initializing from {csv_path}...")
        df = pd.read_csv(csv_path, header=None)
        T = len(frame_paths)

        # 1. Parse initial GT positions and orientations
        first_frame_name = df.iloc[0, 0]
        first_idx = next(i for i, f in enumerate(frame_paths)
                         if os.path.basename(f) == first_frame_name)

        tip_init, shaft_init, ori_init = [], [], []

        x1, y1 = float(df.iloc[0, 1]), float(df.iloc[0, 2])
        vx1, vy1 = float(df.iloc[0, 3]), float(df.iloc[0, 4])
        tip_init.append([x1, y1])
        shaft_init.append([x1 - shaft_offset * vx1, y1 - shaft_offset * vy1])
        ori_init.append([vx1, vy1])
        N = 1

        if len(df.columns) > 8 and not pd.isna(df.iloc[0, 5]):
            x2, y2 = float(df.iloc[0, 5]), float(df.iloc[0, 6])
            vx2, vy2 = float(df.iloc[0, 7]), float(df.iloc[0, 8])
            tip_init.append([x2, y2])
            shaft_init.append([x2 - shaft_offset * vx2, y2 - shaft_offset * vy2])
            ori_init.append([vx2, vy2])
            N = 2

        tip_init   = np.array(tip_init,   dtype=np.float32)
        shaft_init = np.array(shaft_init, dtype=np.float32)

        # 2. Interleave: [tip_0, shaft_0, tip_1, shaft_1]
        M = N * 2
        all_init = np.empty((M, 2), dtype=np.float32)
        for n in range(N):
            all_init[2 * n]     = tip_init[n]
            all_init[2 * n + 1] = shaft_init[n]

        # 3. GT lookup for tip re-anchoring
        gt_positions = {}
        for _, row in df.iterrows():
            fname = str(row.iloc[0])
            pts = [[float(row.iloc[1]), float(row.iloc[2])]]
            if N == 2 and len(row) > 8 and not pd.isna(row.iloc[5]):
                pts.append([float(row.iloc[5]), float(row.iloc[6])])
            gt_positions[fname] = np.array(pts, dtype=np.float32)

        # 4. Chunked CoTracker over all points
        block_size, overlap = 50, 16
        all_tracks = np.zeros((T, M, 2))
        all_vis    = np.zeros((T, M))
        curr_q     = all_init.copy()
        start      = first_idx

        while start < T:
            end = min(start + block_size, T)
            print(f"  [ShaftTrack] Block {start}-{end}...")
            imgs = [T_trans.ToTensor()(Image.open(frame_paths[i]).convert("RGB"))
                    for i in range(start, end)]
            vid = torch.stack(imgs).unsqueeze(0).to(self.device).float()
            q   = torch.zeros((1, M, 3), device=self.device)
            q[0, :, 1:] = torch.from_numpy(curr_q).to(self.device)

            with torch.no_grad():
                t_out, v_out = self.localizer.model(vid, queries=q, grid_size=0)
                t_np = t_out.cpu().numpy()[0]
                v_np = v_out.cpu().numpy()[0]
                L    = t_np.shape[0]
                all_tracks[start:start + L] = t_np
                all_vis[start:start + L]    = v_np
                if end < T:
                    all_tracks[start:start + L] = t_np
                    curr_q = t_np[(end - start) - overlap]
                    start  = end - overlap
                else:
                    break

            del vid
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 5. Separate tip / shaft
        tip_tracks   = all_tracks[:, list(range(0, M, 2)), :]
        shaft_tracks = all_tracks[:, list(range(1, M, 2)), :]
        tip_vis      = all_vis[:,   list(range(0, M, 2))]

        # 6. GT re-anchor tips
        for t, fpath in enumerate(frame_paths):
            fname = os.path.basename(fpath)
            if fname in gt_positions:
                tip_tracks[t] = gt_positions[fname]
                tip_vis[t]    = 1.0
        print(f"  [ShaftTrack] GT re-anchoring: {len(gt_positions)}/{T} frames.")

        # 7. Compute orientation from (tip - shaft) per frame
        shaft_oris = np.zeros((T, N, 2))
        for n in range(N):
            for t in range(T):
                diff = tip_tracks[t, n] - shaft_tracks[t, n]
                norm = np.linalg.norm(diff)
                if norm > 1e-6:
                    shaft_oris[t, n] = diff / norm
                else:
                    shaft_oris[t, n] = shaft_oris[t - 1, n] if t > 0 else ori_init[n]

        # 8. Smooth with 5-frame moving average, then re-normalize
        kernel = np.ones(5) / 5.0
        for n in range(N):
            for c in range(2):
                shaft_oris[:, n, c] = np.convolve(shaft_oris[:, n, c], kernel, mode='same')
            norms = np.linalg.norm(shaft_oris[:, n, :], axis=1, keepdims=True)
            shaft_oris[:, n, :] /= np.maximum(norms, 1e-6)

        print(f"  [ShaftTrack] Per-frame orientation ready for {N} instruments.")
        return tip_tracks, tip_vis, shaft_oris

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
        
        # Indicator parameters - larger and more visible
        length = 50
        
        if visibility > 0.5:
            # TRUE Cyan in BGR = (255, 255, 0) is YELLOW. BGR Cyan = (255, 255, 0) NO.
            # Cyan in RGB(0,255,255) -> BGR(255,255,0) is YELLOW.
            # BGR: B=255, G=255, R=0 -> displayed as Cyan ✓
            color = (255, 255, 0)  # BGR Cyan = Blue+Green channels lit
            line_style = cv2.LINE_AA
            thickness = 3
            label = "VISIBLE"
        else:
            color = (0, 0, 255)  # BGR Red for Phantom
            line_style = cv2.LINE_4
            thickness = 2
            label = "PHANTOM"
            
        # Create a "V" shape gripper outline (U-gripper jaws)
        # Tip point is (x, y). Two arms spread back at +/- 0.5 rad
        p1 = (int(x - length * np.cos(angle - 0.5)), int(y - length * np.sin(angle - 0.5)))
        p2 = (int(x - length * np.cos(angle + 0.5)), int(y - length * np.sin(angle + 0.5)))
        
        if visibility > 0.5:
            cv2.line(img, (x, y), p1, color, thickness, line_style)
            cv2.line(img, (x, y), p2, color, thickness, line_style)
            # Bright glow circle at tip
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 12, color, 1)
        else:
            # Phantom: Draw dashed lines
            self.draw_dashed_line(img, (x, y), p1, color, thickness)
            self.draw_dashed_line(img, (x, y), p2, color, thickness)
            cv2.circle(img, (x, y), 6, color, 1)

        # Label next to indicator
        label_x = min(x + 15, img.shape[1] - 80)
        label_y = max(y - 10, 15)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

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
