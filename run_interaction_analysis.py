import numpy as np
import os
from surgical_interaction_analysis import SurgicalInteractionAnalysis

def main():
    # 1. Load Data
    TISSUE_TRACKS = "op1_mean_tracks.npy"
    INSTRUMENT_TRACKS = "op1_instrument_tracks.npy"
    STRAIN_TRACKS = "op1_strain.npy" # We need to make sure this exists
    
    if not all(os.path.exists(f) for f in [TISSUE_TRACKS, INSTRUMENT_TRACKS]):
        print("Required track files not found. Run tracking scripts first.")
        return
        
    tissue = np.load(TISSUE_TRACKS)
    instr = np.load(INSTRUMENT_TRACKS)
    
    # Check if we have strain cached, otherwise compute it
    if os.path.exists(STRAIN_TRACKS):
        strain = np.load(STRAIN_TRACKS)
    else:
        from temporal_analyzer import TemporalAnalyzer
        analyzer = TemporalAnalyzer()
        strain = analyzer.estimate_deformation_strain(tissue)
        np.save(STRAIN_TRACKS, strain)

    # 2. Analyze Interactions
    analysis = SurgicalInteractionAnalysis()
    min_dists, interaction_mask, freq = analysis.analyze_interactions(tissue, instr)
    
    # 3. Correlation
    correlation = analysis.correlate_with_strain(min_dists, strain)
    print(f"Interaction Analysis Complete:")
    print(f"  - Correlation (Proximity vs Strain): {correlation:.4f}")
    
    # 4. Save Report
    analysis.plot_interaction_report(min_dists, interaction_mask, strain, "op1_interaction_report.png")
    print("Report saved: op1_interaction_report.png")

if __name__ == "__main__":
    main()
