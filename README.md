# 3D Cell Tracking during Embryonic Development (Kaggle Biohub)

This repository contains our state-of-the-art solution for the **Biohub - Cell Tracking During Development** Kaggle competition. The goal is to detect, track, and reconstruct cell lineages (including cell division/mitosis events) from high-resolution 3D time-lapse fluorescent microscopy images of zebrafish embryos.

Our pipeline progressed from a baseline Hungarian matching tracker to a robust, biologically-constrained lineage reconstruction engine, achieving a public leaderboard score of **`0.672`**.

---

## 📈 Leaderboard Progression

| Version | Public Score | Key Improvements |
| :--- | :--- | :--- |
| **V20 / V22** | **0.629** | Baseline Difference of Gaussians (DoG) detector + frame-to-frame Hungarian matching. |
| **V24** | **0.633** | Integrated **Sibling constraints** ($\le 5.0$ µm) for mitosis splitting to filter out false positive division links. |
| **V25** | **0.658** | Added **Robust 99th-percentile adaptive thresholding** to prevent single outlier hyper-bright spots from suppressing cell detections. |
| **V26** | **0.672** | Implemented **Bipartite Gap-Closing with linear position interpolation** to stitch together tracks broken by temporary cell detection dropouts. |

---

## 🛠️ Architecture Overview

The pipeline is composed of two primary modules: an anisotropic 3D cell detector and a temporal lineage tracker.

### 1. Anisotropic 3D Cell Detector (`detector.py`)
- **Difference of Gaussians (DoG)**: Smooths normalized 3D volumes using anisotropic Sigmas based on physical voxel spacing (`1.625` µm in Z, `0.40625` µm in Y and X) and expected cell radius (`3.0` µm).
- **99th-Percentile Adaptive Thresholding**: Calculates dynamic thresholds based on the 99th percentile of positive DoG intensities rather than the absolute maximum. This makes the detector invariant to light attenuation and local bleaching.
- **Physical Non-Maximum Suppression (NMS)**: Suppresses peak detections within a physical radius of `1.5` µm in 3D.

### 2. Sibling-Constrained Gap-Closing Tracker (`tracker.py`)
- **Bipartite Hungarian Matching**: Matches cells frame-to-frame ($t \to t+1$) by minimizing physical migration distances.
- **Sibling-Constrained Mitosis Detection**: Unmatched nodes at $t+1$ are allowed to match back to parents at $t$ as division daughters. Crucially, a parent is only allowed to split if its two candidate daughters are physically close to each other ($\le 5.0$ µm) at birth.
- **Bipartite Gap-Closing**: Identifies track terminations at $t$ and track initiations at $t+2$. If the spatial distance between them is small ($\le 4.0$ µm), the gap is closed.
- **Midpoint Position Interpolation**: For closed gaps, a virtual node is inserted at the missing frame $t+1$ with coordinates calculated as the exact midpoint between $t$ and $t+2$. The two missing edges are reconstructed, healing track fragmentation.

---

## 📂 Repository Structure

- [detector.py](detector.py): Anisotropic 3D cell detector.
- [tracker.py](tracker.py): Bipartite tracking with sibling constraints and gap-closing.
- [metrics.py](metrics.py): Lineage evaluation metric (Edge Jaccard + 0.1 * Division Jaccard).
- [synthetic_data.py](synthetic_data.py): Generates simulated 3D cell volumes and ground-truth lineages for local verification.
- [test_gap_closing.py](test_gap_closing.py): Local test runner comparing tracking with and without gap-closing.
- [make_notebook.py](make_notebook.py): Python script that automatically builds the final production `.ipynb` notebook for Kaggle.
- [wait_and_submit.py](wait_and_submit.py): Background daemon that polls Kaggle kernel status and submits upon successful compile.

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install numpy pandas scipy scikit-image networkx kaggle nbformat
```

### 2. Run Local Lineage Tracking Verification
To run the hyperparameter sweep and verify the gap-closing algorithm performance against simulated ground-truth lineages:
```bash
python test_gap_closing.py
```

### 3. Generate and Push Kaggle Notebook
To compile the production notebook and push it to Kaggle:
```bash
python make_notebook.py
python -c "from kaggle import api; api.authenticate(); api.kernels_push('temp_kernel')"
```
