import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import networkx as nx

def link_cells_with_gap_closing(
    detections_by_time,
    voxel_scale=(1.625, 0.40625, 0.40625),
    max_dist_um=6.0,
    div_max_dist_um=5.0,
    max_sibling_dist_um=5.0,
    max_gap_dist_um=4.0
):
    """
    Sibling-Constrained Temporal Bipartite Matching with Gap-Closing and Linear Position Interpolation.
    
    Parameters:
    -----------
    detections_by_time : dict
        Dict mapping time t -> list of (z, y, x) centroid tuples in voxel coordinates.
    voxel_scale : tuple of float
        (sz, sy, sx) physical voxel spacing in micrometers.
    max_dist_um : float
        Maximum frame-to-frame cell migration distance in micrometers.
    div_max_dist_um : float
        Maximum distance from parent to a dividing daughter cell in micrometers.
    max_sibling_dist_um : float
        Maximum permitted physical distance between sister cell daughters at birth.
    max_gap_dist_um : float
        Maximum migration distance across a 1-frame gap (t -> t+2) for gap closing.
    """
    sz, sy, sx = voxel_scale
    nodes_list = []
    node_id_map = {}
    next_node_id = 1
    T = max(detections_by_time.keys()) + 1 if detections_by_time else 0
    
    # 1. Create nodes representing cell centroids at all timepoints
    for t in range(T):
        t_detections = detections_by_time.get(t, [])
        for idx, (z, y, x) in enumerate(t_detections):
            node_id = next_node_id
            next_node_id += 1
            node_id_map[(t, idx)] = node_id
            nodes_list.append({
                'node_id': node_id, 't': t, 'z': int(z), 'y': int(y), 'x': int(x)
            })
    nodes_df = pd.DataFrame(nodes_list)
    edges_list = []
    
    # 2. Perform frame-by-frame matching using Hungarian algorithm
    for t in range(T - 1):
        dets_t = detections_by_time.get(t, [])
        dets_t1 = detections_by_time.get(t + 1, [])
        if not dets_t or not dets_t1:
            continue
        arr_t = np.array(dets_t, dtype=np.float32)
        arr_t1 = np.array(dets_t1, dtype=np.float32)
        diff = arr_t[:, None, :] - arr_t1[None, :, :]
        
        diff_phys = diff.copy()
        diff_phys[:, :, 0] *= sz
        diff_phys[:, :, 1] *= sy
        diff_phys[:, :, 2] *= sx
        dist_matrix = np.sqrt(np.sum(diff_phys ** 2, axis=-1))
        
        cost_matrix = dist_matrix.copy()
        cost_matrix[cost_matrix > max_dist_um] = 1e9
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_t1 = set()
        parent_to_children = {i: [] for i in range(len(dets_t))}
        parent_child_links = []
        
        # Link 1-to-1 matching links
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= max_dist_um:
                parent_id = node_id_map[(t, r)]
                child_id = node_id_map[(t + 1, c)]
                parent_child_links.append((parent_id, child_id))
                matched_t1.add(c)
                parent_to_children[r].append(c)
                
        # Link unmatched cells to parents (cell division detection)
        unmatched_t1 = [c for c in range(len(dets_t1)) if c not in matched_t1]
        for c in unmatched_t1:
            parent_dists = dist_matrix[:, c]
            best_r = np.argmin(parent_dists)
            if parent_dists[best_r] <= div_max_dist_um:
                existing_children = parent_to_children[best_r]
                # Enforce physical proximity of division siblings
                if len(existing_children) == 1:
                    sibling_idx = existing_children[0]
                    sib_diff = arr_t1[sibling_idx] - arr_t1[c]
                    sib_dist = np.sqrt((sib_diff[0]*sz)**2 + (sib_diff[1]*sy)**2 + (sib_diff[2]*sx)**2)
                    if sib_dist <= max_sibling_dist_um:
                        parent_id = node_id_map[(t, best_r)]
                        child_id = node_id_map[(t + 1, c)]
                        parent_child_links.append((parent_id, child_id))
                        parent_to_children[best_r].append(c)
                        
        for parent_id, child_id in parent_child_links:
            edges_list.append({'source_id': parent_id, 'target_id': child_id})
            
    # 3. Gap Closing with Linear Coordinate Interpolation
    G = nx.DiGraph()
    for _, node in nodes_df.iterrows():
        G.add_node(int(node['node_id']), t=int(node['t']), z=node['z'], y=node['y'], x=node['x'])
    for edge in edges_list:
        G.add_edge(int(edge['source_id']), int(edge['target_id']))
        
    # Terminating and initiating endpoints of tracks
    terminating = [n for n in G.nodes() if G.out_degree(n) == 0 and G.nodes[n]['t'] < T - 1]
    initiating = [n for n in G.nodes() if G.in_degree(n) == 0 and G.nodes[n]['t'] > 0]
    
    added_nodes = []
    added_edges = []
    matched_initiating = set()
    
    for u in terminating:
        t_u = G.nodes[u]['t']
        u_coords = np.array([G.nodes[u]['z'] * sz, G.nodes[u]['y'] * sy, G.nodes[u]['x'] * sx])
        
        best_v = None
        best_dist = 1e9
        candidates = [v for v in initiating if G.nodes[v]['t'] == t_u + 2 and v not in matched_initiating]
        
        for v in candidates:
            v_coords = np.array([G.nodes[v]['z'] * sz, G.nodes[v]['y'] * sy, G.nodes[v]['x'] * sx])
            dist = np.sqrt(np.sum((u_coords - v_coords) ** 2))
            if dist <= max_gap_dist_um and dist < best_dist:
                best_dist = dist
                best_v = v
                
        if best_v is not None:
            matched_initiating.add(best_v)
            
            # Midpoint linear position interpolation
            interp_z = int(round((G.nodes[u]['z'] + G.nodes[best_v]['z']) / 2))
            interp_y = int(round((G.nodes[u]['y'] + G.nodes[best_v]['y']) / 2))
            interp_x = int(round((G.nodes[u]['x'] + G.nodes[best_v]['x']) / 2))
            
            interp_id = next_node_id
            next_node_id += 1
            
            added_nodes.append({
                'node_id': interp_id,
                't': t_u + 1,
                'z': interp_z,
                'y': interp_y,
                'x': interp_x
            })
            added_edges.append({'source_id': u, 'target_id': interp_id})
            added_edges.append({'source_id': interp_id, 'target_id': best_v})
            
    if added_nodes:
        nodes_df = pd.concat([nodes_df, pd.DataFrame(added_nodes)], ignore_index=True)
    edges_df = pd.DataFrame(edges_list + added_edges) if (edges_list or added_edges) else pd.DataFrame(columns=['source_id', 'target_id'])
    
    return nodes_df, edges_df

def link_cells(
    detections_by_time,
    voxel_scale=(1.625, 0.40625, 0.40625),
    max_dist_um=6.0,
    div_max_dist_um=5.0,
    max_sibling_dist_um=5.0,
    max_gap_dist_um=4.0
):
    return link_cells_with_gap_closing(
        detections_by_time=detections_by_time,
        voxel_scale=voxel_scale,
        max_dist_um=max_dist_um,
        div_max_dist_um=div_max_dist_um,
        max_sibling_dist_um=max_sibling_dist_um,
        max_gap_dist_um=max_gap_dist_um
    )
