# Surgical Tissue Localization & Deformation Tracking

This project provides a robust framework for **anatomical landmark localization** and **tissue deformation tracking** in complex surgical environments.

## The Surgical Problem
Tissue localization in surgery is notoriously difficult due to:
- **Elastic Deformation**: Organs change shape during manipulation.
- **Occlusions**: Tools, bleeding, and smoke often block the line of sight.
- **Dynamic Lighting**: Camera motion and light reflection create visual noise.

This pipeline uses **CoTracker3** to maintain long-term point correspondence and **MC-Dropout** to quantify tracking reliability, ensuring that surgical navigation or robotic assistance systems can "trust" the localized coordinates.

## Tech Stack
- **Tracking Engine**: CoTracker3 (`scaled_online.pth`)
- **Uncertainty Method**: MC-Dropout (Stochastic Inference)
- **Frameworks**: PyTorch, OpenCV, PIL

## Project Architecture
The project is designed to be modular and reusable:

| Module | Description |
| :--- | :--- |
| `mc_dropout.py` | **Generic UQ Wrapper**: Handles dropout-layer identification and stochastic inference loops. |
| `tissue_localizer.py` | **Surgical Context**: Manages point sampling from masks and sequence-level tracking orchestration. |
| `run_tracking.py` | **Main Entry Point**: Configures paths and runs the full tracking pipeline. |
| `visualize_dual.py` | **Visualization**: Generates separate videos for standard tracking and uncertainty (with halos). |
| `utils.py` | **Utilities**: Contains natural sorting and other helper functions. |
| `create_linkedin_gif.py` | **Social Media**: Generates high-quality GIFs for sharing results. |

## Clinical Analysis Features
- **Temporal Consistency**: Automated rejection of unstable trajectories via smoothness and drift thresholds.
- **Occlusion Handling**: Differentiation between tool-tissue interaction and generic tracking loss using instrument masks.
- **Deformation Modeling**: Estimation of local tissue flow (Motion Fields) and stretching/compression (Local Strain).
- **Performance Scoring**: Tracking of Survival Rate, Cumulative Drift Profile, and EPE framework support.

## Current Status
- [x] **Uncertainty Quantification**: Full MC-Dropout integration with CoTracker3.
- [x] **Clinical Analysis Modules**: Smoothness, Occlusion, and Deformation logic complete.
- [x] **Visualization Dashboard**: Automated PNG dashboard for all temporal metrics.
- [x] **Augmented Video**: State-aware overlays and motion fields integrated into the visualizer.
- [x] **OP1 Verification**: Successful metric validation on the first 200 frames.

## Next Steps
1.  **Full Scale Tracking**: Run `run_tracking.py` on the complete 1122-frame `OP1` sequence with 10 MC-samples.
2.  **Dataset Expansion**: Process sequences `OP2`, `OP3`, and `OP4`.
3.  **Localization refinement**: Use the variance data (`op1_var_tracks.npy`) to filter out low-confidence trajectories.

## 📁 Key File Locations
- **Raw Images**: `Tracking_Rigid_Training/Training/OP1/Raw/`
- **Masks**: `Tracking_Rigid_Training/Training/OP1/Mask/`
- **Tracking Results**: `op1_mean_tracks.npy`, `op1_var_tracks.npy`
- **Videos/GIFs**: Root directory (`.mp4`, `.gif`)
