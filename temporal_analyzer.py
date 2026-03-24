import numpy as np
from scipy.spatial import KDTree

class TemporalAnalyzer:
    def __init__(self):
        pass

    def calculate_smoothness(self, trajectories):
        """
        Calculate smoothness as the inverse of the mean acceleration magnitude.
        trajectories: (T, N, 2)
        returns: (N,) smoothness scores (higher is smoother)
        """
        if trajectories.shape[0] < 3:
            return np.ones(trajectories.shape[1])
            
        # Velocity: (T-1, N, 2)
        vel = np.diff(trajectories, axis=0)
        # Acceleration: (T-2, N, 2)
        acc = np.diff(vel, axis=0)
        
        acc_mag = np.linalg.norm(acc, axis=-1) # (T-2, N)
        mean_acc = np.mean(acc_mag, axis=0) # (N,)
        
        # Smoothness = 1 / (1 + mean_acc) to bound between 0 and 1
        smoothness = 1.0 / (1.0 + mean_acc)
        return smoothness

    def calculate_drift(self, trajectories):
        """
        Calculate drift as the net displacement from the initial position.
        trajectories: (T, N, 2)
        returns: (N,) drift values (Euclidean distance)
        """
        if trajectories.shape[0] < 2:
            return np.zeros(trajectories.shape[1])
            
        initial_pos = trajectories[0]
        final_pos = trajectories[-1]
        drift = np.linalg.norm(final_pos - initial_pos, axis=-1)
        return drift

    def calculate_drift_over_time(self, trajectories):
        """
        Calculate cumulative drift over time.
        returns: (T, N) cumulative displacement
        """
        if trajectories.shape[0] < 2:
            return np.zeros(trajectories.shape[:2])
            
        initial_pos = trajectories[0]
        # Distance from origin at each frame
        drift_profile = np.linalg.norm(trajectories - initial_pos, axis=-1)
        return drift_profile

    def calculate_survival_rate(self, visibility):
        """
        Percentage of tracks that remain visible (vis > 0.5) over time.
        returns: (T,) percentage of alive tracks
        """
        alive = (visibility > 0.5).astype(float)
        return np.mean(alive, axis=1)

    def calculate_epe(self, trajectories, gt_trajectories):
        """
        Endpoint Error: Euclidean distance to ground truth.
        returns: (T, N) error map
        """
        # Ensure shapes match (T might differ if GT is sparse)
        T_pred = trajectories.shape[0]
        T_gt = gt_trajectories.shape[0]
        T = min(T_pred, T_gt)
        
        error = np.linalg.norm(trajectories[:T] - gt_trajectories[:T], axis=-1)
        return error

    def filter_trajectories(self, trajectories, visibility, smoothness_thresh=0.5, drift_thresh=100.0):
        """
        Mark tracks as rejected if they are unstable based on smoothness or excessive drift.
        returns: (N,) boolean mask (True for valid, False for rejected)
        """
        N = trajectories.shape[1]
        smoothness = self.calculate_smoothness(trajectories)
        drift = self.calculate_drift(trajectories)
        
        # Valid if smoothness is high AND drift is not astronomical (unless legitimate motion)
        # Note: drift_thresh depends on image scale; 100px is a decent default for 512px images
        valid_mask = (smoothness >= smoothness_thresh) & (drift <= drift_thresh)
        
        # Also reject if visibility is too low
        mean_vis = np.mean(visibility, axis=0)
        valid_mask = valid_mask & (mean_vis > 0.5)
        
        return valid_mask

    def detect_occlusion_events(self, trajectories: np.ndarray, visibility: np.ndarray, mask_sequence: np.ndarray = None):
        """
        Detect when points are lost and when they are recovered,
        differentiating between tool occlusion and generic disappearance.
        
        trajectories: (T, N, 2)
        visibility: (T, N)
        mask_sequence: (T, H, W) binary masks where 255 (or >0) is the instrument
        
        returns:
            events: List of dicts per point
            state_mask: (T, N) integer mask (0:Visible, 1:Tool-Occluded, 2:Lost-Other)
        """
        T, N = visibility.shape
        events = []
        state_mask = np.zeros((T, N), dtype=int)
        
        for n in range(N):
            v = visibility[:, n]
            traj = trajectories[:, n]
            
            p_events = {
                "point_idx": n,
                "lost_at": [],
                "recovered_at": [],
                "tool_occlusions": 0,
                "other_losses": 0
            }
            
            for t in range(T):
                if v[t] < 0.5:
                    # Point is lost
                    is_tool = False
                    if mask_sequence is not None and t < mask_sequence.shape[0]:
                        x, y = int(traj[t, 0]), int(traj[t, 1])
                        # Check bounds
                        if 0 <= y < mask_sequence.shape[1] and 0 <= x < mask_sequence.shape[2]:
                            if mask_sequence[t, y, x] > 0:
                                is_tool = True
                    
                    if is_tool:
                        state_mask[t, n] = 1 # Tool-Occluded
                    else:
                        state_mask[t, n] = 2 # Lost-Other
                        
                    # Transition detection
                    if t > 0 and v[t-1] >= 0.5:
                        p_events["lost_at"].append(t)
                        if is_tool: p_events["tool_occlusions"] += 1
                        else: p_events["other_losses"] += 1
                else:
                    state_mask[t, n] = 0 # Visible
                    if t > 0 and v[t-1] < 0.5:
                        p_events["recovered_at"].append(t)
            
            p_events["visibility_ratio"] = np.mean(v)
            events.append(p_events)
            
        return events, state_mask

    def estimate_deformation_strain(self, trajectories, k_neighbors=5):
        """
        Estimate local strain by measuring changes in distance to K-nearest neighbors.
        trajectories: (T, N, 2)
        returns: (T, N) strain values
        """
        T, N, _ = trajectories.shape
        if N <= k_neighbors:
            return np.zeros((T, N))
            
        # 1. Identity neighbors at t=0
        initial_pos = trajectories[0]
        tree = KDTree(initial_pos)
        _, neighbor_indices = tree.query(initial_pos, k=k_neighbors + 1)
        neighbor_indices = neighbor_indices[:, 1:] # Exclude self
        
        # 2. Calculate initial distances: (N, K)
        initial_dists = np.zeros((N, k_neighbors))
        for i in range(N):
            neighbors = initial_pos[neighbor_indices[i]]
            initial_dists[i] = np.linalg.norm(neighbors - initial_pos[i], axis=-1)
            
        # 3. Calculate distances over time and compute strain
        # Strain = |curr_dist - init_dist| / init_dist
        strain_map = np.zeros((T, N))
        
        for t in range(T):
            curr_pos = trajectories[t]
            for i in range(N):
                neighbors = curr_pos[neighbor_indices[i]]
                curr_dists = np.linalg.norm(neighbors - curr_pos[i], axis=-1)
                
                # Element-wise strain relative to initial
                # We take the mean strain across K neighbors
                point_strain = np.mean(np.abs(curr_dists - initial_dists[i]) / (initial_dists[i] + 1e-6))
                strain_map[t, i] = point_strain
                
        return strain_map

    def compute_motion_vectors(self, trajectories, visibility):
        """
        Compute local motion vectors (flow) for visible points.
        trajectories: (T, N, 2)
        returns: (T-1, N, 2) velocity vectors
        """
        vel = np.diff(trajectories, axis=0)
        # Mask out velocities for occluded points (both start and end frame must be visible)
        vis_mask = visibility[:-1] * visibility[1:]
        vel = vel * vis_mask[:, :, None]
        return vel

    def get_temporal_metrics_summary(self, trajectories, visibility, mask_sequence=None, gt_trajectories=None):
        """Combined metrics for easy reporting."""
        smoothness = self.calculate_smoothness(trajectories)
        occlusions, state_mask = self.detect_occlusion_events(trajectories, visibility, mask_sequence)
        strain = self.estimate_deformation_strain(trajectories)
        drift = self.calculate_drift(trajectories)
        survival = self.calculate_survival_rate(visibility)
        
        summary = {
            "mean_smoothness": np.mean(smoothness),
            "mean_visibility": np.mean(visibility),
            "mean_strain": np.mean(strain),
            "mean_drift": np.mean(drift),
            "final_survival_rate": survival[-1],
            "total_occlusion_events": sum(len(e["lost_at"]) for e in occlusions),
            "total_recovery_events": sum(len(e["recovered_at"]) for e in occlusions),
            "tool_occlusion_ratio": np.mean(state_mask == 1)
        }
        
        if gt_trajectories is not None:
            epe = self.calculate_epe(trajectories, gt_trajectories)
            summary["mean_epe"] = np.mean(epe)
            
        return summary
