import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment

def calculate_pairwise_distances(coords1, coords2, voxel_scale):
    """
    Computes physical pairwise distances between two sets of coordinates.
    coords1: (N, 3), coords2: (M, 3)
    voxel_scale: (sz, sy, sx)
    Returns: (N, M) distance matrix
    """
    sz, sy, sx = voxel_scale
    c1 = np.array(coords1, dtype=np.float32)
    c2 = np.array(coords2, dtype=np.float32)
    
    if len(c1) == 0 or len(c2) == 0:
        return np.zeros((len(c1), len(c2)))
        
    diff = c1[:, None, :] - c2[None, :, :]
    diff[:, :, 0] *= sz
    diff[:, :, 1] *= sy
    diff[:, :, 2] *= sx
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def match_nodes_per_timepoint(pred_nodes_df, gt_nodes_df, voxel_scale, max_dist_um=7.0):
    """
    Matches predicted nodes to ground-truth nodes per timepoint.
    Returns:
    - pred_to_gt: dict mapping pred_node_id (int) -> gt_node_id (int)
    - gt_to_pred: dict mapping gt_node_id (int) -> pred_node_id (int)
    """
    pred_to_gt = {}
    gt_to_pred = {}
    
    T_max = max(pred_nodes_df['t'].max() if len(pred_nodes_df) > 0 else 0,
                gt_nodes_df['t'].max() if len(gt_nodes_df) > 0 else 0)
                
    for t in range(T_max + 1):
        p_t = pred_nodes_df[pred_nodes_df['t'] == t]
        g_t = gt_nodes_df[gt_nodes_df['t'] == t]
        
        if len(p_t) == 0 or len(g_t) == 0:
            continue
            
        p_coords = p_t[['z', 'y', 'x']].values
        g_coords = g_t[['z', 'y', 'x']].values
        p_ids = p_t['node_id'].values
        g_ids = g_t['node_id'].values
        
        dist_matrix = calculate_pairwise_distances(p_coords, g_coords, voxel_scale)
        
        # Bipartite matching
        cost_matrix = dist_matrix.copy()
        cost_matrix[cost_matrix > max_dist_um] = 1e9
        
        p_ind, g_ind = linear_sum_assignment(cost_matrix)
        
        for p_idx, g_idx in zip(p_ind, g_ind):
            if dist_matrix[p_idx, g_idx] <= max_dist_um:
                p_id = int(p_ids[p_idx])
                g_id = int(g_ids[g_idx])
                pred_to_gt[p_id] = g_id
                gt_to_pred[g_id] = p_id
                
    return pred_to_gt, gt_to_pred

def evaluate_lineage(pred_nodes_df, pred_edges_df, gt_nodes_df, gt_edges_df, voxel_scale, max_dist_um=7.0):
    """
    Computes Edge Jaccard and Division Jaccard metrics using networkx for graph analysis.
    """
    # 1. Match nodes
    pred_to_gt, gt_to_pred = match_nodes_per_timepoint(pred_nodes_df, gt_nodes_df, voxel_scale, max_dist_um)
    
    # Build degree maps for GT
    gt_in_degrees = {}
    gt_out_degrees = {}
    for gt_id in gt_nodes_df['node_id']:
        gt_in_degrees[int(gt_id)] = 0
        gt_out_degrees[int(gt_id)] = 0
    for _, edge in gt_edges_df.iterrows():
        s, t = int(edge['source_id']), int(edge['target_id'])
        if s in gt_out_degrees:
            gt_out_degrees[s] += 1
        if t in gt_in_degrees:
            gt_in_degrees[t] += 1
            
    gt_edges_set = set(zip(gt_edges_df['source_id'].astype(int), gt_edges_df['target_id'].astype(int)))
    
    # 2. Edge Matching (Jaccard)
    edge_tp = 0
    edge_fp = 0
    
    for _, edge in pred_edges_df.iterrows():
        u_pred, v_pred = int(edge['source_id']), int(edge['target_id'])
        u_gt = pred_to_gt.get(u_pred)
        v_gt = pred_to_gt.get(v_pred)
        
        is_tp = False
        if u_gt is not None and v_gt is not None:
            if (u_gt, v_gt) in gt_edges_set:
                edge_tp += 1
                is_tp = True
                
        if not is_tp:
            u_valid = (u_gt is not None and gt_out_degrees.get(u_gt, 0) > 0)
            v_valid = (v_gt is not None and gt_in_degrees.get(v_gt, 0) > 0)
            if u_valid or v_valid:
                edge_fp += 1
                
    pred_edges_mapped = set()
    for _, edge in pred_edges_df.iterrows():
        u_pred, v_pred = int(edge['source_id']), int(edge['target_id'])
        u_gt = pred_to_gt.get(u_pred)
        v_gt = pred_to_gt.get(v_pred)
        if u_gt is not None and v_gt is not None:
            pred_edges_mapped.add((u_gt, v_gt))
            
    edge_fn = len(gt_edges_set - pred_edges_mapped)
    
    # 3. Division Matching (Jaccard)
    # Build NetworkX DiGraphs for GT and prediction
    gt_graph = nx.DiGraph()
    for gt_id in gt_nodes_df['node_id']:
        gt_graph.add_node(int(gt_id))
    for _, edge in gt_edges_df.iterrows():
        gt_graph.add_edge(int(edge['source_id']), int(edge['target_id']))
        
    pred_graph = nx.DiGraph()
    for pred_id in pred_nodes_df['node_id']:
        pred_graph.add_node(int(pred_id))
    for _, edge in pred_edges_df.iterrows():
        pred_graph.add_edge(int(edge['source_id']), int(edge['target_id']))
        
    # Find GT division events (nodes with out-degree >= 2)
    gt_divisions = [node for node in gt_graph.nodes() if gt_graph.out_degree(node) >= 2]
    
    # Find predicted division events
    pred_divisions = [node for node in pred_graph.nodes() if pred_graph.out_degree(node) >= 2]
    
    div_tp = 0
    div_fn = 0
    div_fp = 0
    
    # Get connected components of the undirected predicted graph
    pred_components = list(nx.weakly_connected_components(pred_graph))
    
    # For each GT division, check if there is a predicted component that covers it
    matched_pred_divisions = set()
    
    for gt_div in gt_divisions:
        # Pre-split stage: ancestors and the division node itself
        pre_split_gt = nx.ancestors(gt_graph, gt_div) | {gt_div}
        pre_split_pred = {gt_to_pred[n] for n in pre_split_gt if n in gt_to_pred}
        
        # Daughter lineages
        children = list(gt_graph.successors(gt_div))
        if len(children) < 2:
            continue
            
        lineage1_gt = nx.descendants(gt_graph, children[0]) | {children[0]}
        lineage2_gt = nx.descendants(gt_graph, children[1]) | {children[1]}
        
        lineage1_pred = {gt_to_pred[n] for n in lineage1_gt if n in gt_to_pred}
        lineage2_pred = {gt_to_pred[n] for n in lineage2_gt if n in gt_to_pred}
        
        # Check if any predicted connected component covers pre-split and touches both daughter lineages
        covered = False
        for comp in pred_components:
            has_pre_split = len(comp & pre_split_pred) > 0
            has_l1 = len(comp & lineage1_pred) > 0
            has_l2 = len(comp & lineage2_pred) > 0
            
            if has_pre_split and has_l1 and has_l2:
                covered = True
                
                # Check if there is a predicted division node in this component
                comp_divisions = [d for d in pred_divisions if d in comp]
                if comp_divisions:
                    # Mark these predicted divisions as matched
                    for d in comp_divisions:
                        matched_pred_divisions.add(d)
                break
                
        if covered:
            div_tp += 1
        else:
            div_fn += 1
            
    # For each predicted division, check if it's a False Positive
    for pred_div in pred_divisions:
        if pred_div not in matched_pred_divisions:
            # Only count as FP if the parent matches a GT node with out-degree > 0
            gt_node = pred_to_gt.get(pred_div)
            if gt_node is not None and gt_out_degrees.get(gt_node, 0) > 0:
                div_fp += 1
                
    # 4. Adjusted Edge Jaccard calculation
    T_pred = len(pred_nodes_df)
    T_true = len(gt_nodes_df)
    
    edge_denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / edge_denom if edge_denom > 0 else 0.0
    
    if T_true > 0:
        total_node_ratio = (T_pred - T_true) / T_true
        adj_edge_jaccard = max(0.0, edge_jaccard * (1.0 - 0.1 * total_node_ratio))
    else:
        total_node_ratio = 0.0
        adj_edge_jaccard = edge_jaccard
        
    # Division Jaccard
    div_denom = div_tp + div_fp + div_fn
    div_jaccard = div_tp / div_denom if div_denom > 0 else 0.0
    
    # Combined score
    score = adj_edge_jaccard + 0.1 * div_jaccard
    
    return {
        'edge_tp': edge_tp,
        'edge_fp': edge_fp,
        'edge_fn': edge_fn,
        'edge_jaccard': edge_jaccard,
        'adj_edge_jaccard': adj_edge_jaccard,
        'division_tp': div_tp,
        'division_fp': div_fp,
        'division_fn': div_fn,
        'division_jaccard': div_jaccard,
        'score': score
    }
