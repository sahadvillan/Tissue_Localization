import numpy as np
import os
import matplotlib.pyplot as plt
from temporal_analyzer import TemporalAnalyzer

class SurgicalInteractionAnalysis:
    def __init__(self):
        self.analyzer = TemporalAnalyzer()

    def analyze_interactions(self, tissue_tracks, instrument_tracks, interaction_threshold=50.0):
        """
        tissue_tracks: (T, N_tissue, 2)
        instrument_tracks: (T, N_instr, 2)
        """
        T, N_tissue, _ = tissue_tracks.shape
        T, N_instr, _ = instrument_tracks.shape
        
        # 1. Calculate proximity for every frame
        # distances: (T, N_tissue, N_instr)
        distances = np.zeros((T, N_tissue, N_instr))
        for t in range(T):
            for n_i in range(N_instr):
                tip_pos = instrument_tracks[t, n_i]
                dists = np.linalg.norm(tissue_tracks[t] - tip_pos, axis=1)
                distances[t, :, n_i] = dists
                
        # 2. Minimum distance to any instrument for each tissue point
        min_dists = np.min(distances, axis=2) # (T, N_tissue)
        
        # 3. Identify Interaction Events
        interaction_mask = min_dists < interaction_threshold
        interaction_frequency = np.mean(interaction_mask, axis=0) # (N_tissue,)
        
        return min_dists, interaction_mask, interaction_frequency

    def correlate_with_strain(self, min_dists, strain_tracks):
        """Check if higher strain happens when tools are closer."""
        # Flat arrays for correlation
        valid_indices = min_dists < 500 # Ignore outliers
        dists_flat = min_dists[valid_indices]
        strain_flat = strain_tracks[valid_indices]
        
        correlation = np.corrcoef(dists_flat, strain_flat)[0, 1]
        return correlation

    def plot_interaction_report(self, min_dists, interaction_mask, strain, save_path="interaction_report.png"):
        """Generate plots showing how proximity affects tissue."""
        T = min_dists.shape[0]
        avg_prox = np.mean(min_dists, axis=1)
        avg_strain = np.mean(strain, axis=1)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(T), avg_prox, label="Avg Tool Proximity", color="blue")
        plt.title("Tool Proximity over Time")
        plt.xlabel("Frame")
        plt.ylabel("Distance (px)")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(min_dists.flatten()[::100], strain.flatten()[::100], alpha=0.1, color="red")
        plt.title("Proximity vs. Local Strain")
        plt.xlabel("Dist to Tool (px)")
        plt.ylabel("Local Strain")
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Interaction report saved to {save_path}")

if __name__ == "__main__":
    # Example usage (requires existing tracks)
    if os.path.exists("op1_mean_tracks.npy"):
        tissue = np.load("op1_mean_tracks.npy")
        # For demo, if we don't have instrument tracks yet, we'll wait for the runner
        print("SurgicalInteractionAnalysis ready for production integration.")
