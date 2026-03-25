import os
import torch
import numpy as np
import cv2
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from mc_dropout import MCDropoutWrapper

class TissueLocalizer:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        # For scaled_online.pth, use offline=False and window_len=16
        self.model = CoTrackerPredictor(
            checkpoint=checkpoint_path, 
            offline=False, 
            window_len=16
        )
        self.model.to(self.device)
        self.mc_wrapper = MCDropoutWrapper(self.model)
        # pass frame_paths to the constructor or set self.frames
        # before calling track_instrument_tips.
        # Example: self.frames = frame_paths
        
    def sample_tissue_points(self, mask_path, num_points=50):
        """Sample points from background (mask == 0)."""
        mask = np.array(Image.open(mask_path))
        # If mask is RGB, convert to grayscale
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
        bg_y, bg_x = np.where(mask == 0)
        if len(bg_y) == 0:
            raise ValueError("No background pixels found in mask!")
            
        indices = np.random.choice(len(bg_y), num_points, replace=False)
        sampled_points = np.stack([bg_x[indices], bg_y[indices]], axis=1) # (N, 2)
        return sampled_points

    def track_sequence(self, frame_paths, query_points, mc_samples=1, block_size=50, overlap=16, use_mc_dropout=True):
        """
        Track points across a long sequence using overlapping offline chunks,
        optionally using MC-Dropout for uncertainty quantification.
        """
        T = len(frame_paths)
        N = query_points.shape[0]
        
        if use_mc_dropout:
            self.mc_wrapper.enable_mc_dropout()
            n_runs = mc_samples
        else:
            n_runs = 1
            
        all_samples = np.zeros((n_runs, T, N, 2))
        all_vis = np.zeros((n_runs, T, N))
        
        # We process each run independently for simplicity with block-based tracking
        for s in range(n_runs):
            if n_runs > 1:
                print(f"MC Sample {s+1}/{n_runs}...")
            
            sample_tracks = np.zeros((T, N, 2))
            sample_vis = np.zeros((T, N))
            start = 0
            curr_queries = query_points.copy()
            
            while start < T:
                end = min(start + block_size, T)
                print(f"  Processing block {start}-{end}...")
                
                chunk_paths = frame_paths[start:end]
                # Load chunk as uint8 first to save memory
                chunk_bytes = [np.array(Image.open(p)) for p in chunk_paths]
                video_chunk = torch.from_numpy(np.stack(chunk_bytes)).permute(0, 3, 1, 2)[None]
                
                # Queries for this chunk: at t=0 of this chunk
                queries = torch.zeros((1, N, 3), device=self.device)
                queries[0, :, 1:] = torch.from_numpy(curr_queries).to(self.device)
                
                with torch.no_grad():
                    # Convert to float and move to device only for the active chunk
                    # Then immediately back to CPU/del
                    tracks, vis = self.model(video_chunk.float().to(self.device), queries=queries)
                    tracks_np = tracks.cpu().numpy()[0] # (block_T, N, 2)
                    vis_np = vis.cpu().numpy()[0] # (block_T, N)
                    
                    # Fill result
                    sample_tracks[start:end] = tracks_np
                    sample_vis[start:end] = vis_np
                    
                    if end < T:
                        next_start = end - overlap
                        rel_idx = next_start - start
                        curr_queries = tracks_np[rel_idx]
                        start = next_start
                    else:
                        start = end
                
                del video_chunk
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            all_samples[s] = sample_tracks
            all_vis[s] = sample_vis
            
        total_mean = np.mean(all_samples, axis=0)
        total_var = np.var(all_samples, axis=0)
        total_vis = np.mean(all_vis, axis=0) # Mean visibility across samples
        
        return total_mean, total_var, total_vis

    def visualize_sampling(self, frame_path, points, save_path):
        """Visualize sampled points on the first frame."""
        image = cv2.imread(frame_path)
        for i, (x, y) in enumerate(points):
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(save_path, image)
        print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    # Test initialization
    checkpoint = "checkpoints/scaled_online.pth"
    if os.path.exists(checkpoint):
        localizer = TissueLocalizer(checkpoint)
        print("TissueLocalizer initialized successfully.")
