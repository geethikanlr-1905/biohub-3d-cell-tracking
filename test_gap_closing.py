import numpy as np
import pandas as pd
from synthetic_data import generate_synthetic_dataset
from metrics import evaluate_lineage
from detector import detect_cells_3d
from tracker import link_cells_with_gap_closing

def main():
    print("=== Verification of Sibling-Constrained Tracking WITH Gap Closing ===")
    T = 8
    Z, Y, X = 32, 128, 128
    voxel_scale = (1.625, 0.40625, 0.40625)
    
    scores_baseline = []
    scores_gap = []
    
    for run in range(3):
        print(f"\n--- Run {run + 1} ---")
        images, gt_nodes, gt_edges = generate_synthetic_dataset(
            T=T, Z=Z, Y=Y, X=X,
            num_initial_cells=20,
            division_prob=0.10,
            noise_sigma=0.07,  # Substantial noise to cause occasional missed detections
            voxel_scale=voxel_scale,
            cell_radius_um=3.0,
            diffusion_um=0.7
        )
        
        detected_centroids = {}
        for t in range(T):
            detected_centroids[t] = detect_cells_3d(
                images[t], voxel_scale, cell_radius_um=3.0, rel_threshold=0.20, min_dist_um=1.5
            )
            
        # 1. Tracker WITHOUT Gap Closing
        nodes1, edges1 = link_cells_with_gap_closing(
            detected_centroids, voxel_scale, max_dist_um=6.0, div_max_dist_um=5.0, max_sibling_dist_um=5.0, max_gap_dist_um=0.0
        )
        res1 = evaluate_lineage(nodes1, edges1, gt_nodes, gt_edges, voxel_scale)
        scores_baseline.append(res1['score'])
        print(f"  Without Gap Closing Score: {res1['score']:.4f}")
        
        # 2. Tracker WITH Gap Closing & Interpolation
        nodes2, edges2 = link_cells_with_gap_closing(
            detected_centroids, voxel_scale, max_dist_um=6.0, div_max_dist_um=5.0, max_sibling_dist_um=5.0, max_gap_dist_um=4.0
        )
        res2 = evaluate_lineage(nodes2, edges2, gt_nodes, gt_edges, voxel_scale)
        scores_gap.append(res2['score'])
        print(f"  With Gap Closing Score:    {res2['score']:.4f}")
        
    print("\n=========================================")
    print(f"Average Score WITHOUT Gap Closing: {np.mean(scores_baseline):.4f}")
    print(f"Average Score WITH Gap Closing:    {np.mean(scores_gap):.4f}")
    print("=========================================")

if __name__ == '__main__':
    main()
