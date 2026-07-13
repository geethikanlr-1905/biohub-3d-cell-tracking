import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

def generate_synthetic_dataset(
    T=10, Z=32, Y=128, X=128,
    num_initial_cells=20,
    division_prob=0.08,
    noise_sigma=0.05,
    voxel_scale=(1.625, 0.40625, 0.40625),
    cell_radius_um=3.0,
    diffusion_um=0.8
):
    """
    Generates a synthetic 3D+time cell tracking dataset.
    
    Parameters:
    - T: Number of timepoints
    - Z, Y, X: Volume shape in voxels
    - num_initial_cells: Number of cells at t=0
    - division_prob: Probability of a cell dividing at each timepoint
    - noise_sigma: Standard deviation of Gaussian noise added to images
    - voxel_scale: Physical size of a voxel (z, y, x) in micrometers
    - cell_radius_um: Physical radius of cell nuclei in micrometers
    - diffusion_um: Diffusion step size (standard deviation of motion) per frame in micrometers
    
    Returns:
    - images: Numpy array of shape (T, Z, Y, X)
    - nodes: DataFrame with columns [node_id, t, z, y, x]
    - edges: DataFrame with columns [source_id, target_id]
    """
    # Scale conversion
    sz, sy, sx = voxel_scale
    
    # Track cell states
    # A cell state is represented by (node_id, z, y, x) where coordinates are in physical micrometers
    next_node_id = 1
    
    # Initialize cells at t=0
    active_cells = []
    nodes_list = []
    edges_list = []
    
    # Random initial placement in the volume (away from boundaries)
    margin_um = 10.0
    z_min, z_max = margin_um, (Z - 1) * sz - margin_um
    y_min, y_max = margin_um, (Y - 1) * sy - margin_um
    x_min, x_max = margin_um, (X - 1) * sx - margin_um
    
    for _ in range(num_initial_cells):
        z = np.random.uniform(z_min, z_max)
        y = np.random.uniform(y_min, y_max)
        x = np.random.uniform(x_min, x_max)
        node_id = next_node_id
        next_node_id += 1
        active_cells.append((node_id, z, y, x))
        nodes_list.append({
            'node_id': node_id,
            't': 0,
            'z': int(round(z / sz)),
            'y': int(round(y / sy)),
            'x': int(round(x / sx))
        })
        
    # Track cell positions over time
    cell_history = {0: list(active_cells)}
    
    for t in range(1, T):
        prev_cells = cell_history[t - 1]
        current_cells = []
        
        for parent_id, p_z, p_y, p_x in prev_cells:
            # Decide if cell divides
            if np.random.rand() < division_prob:
                # Mitosis: splits into two daughters
                d1_id = next_node_id
                d2_id = next_node_id + 1
                next_node_id += 2
                
                # Split direction: random unit vector
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                dx = np.sin(phi) * np.cos(theta)
                dy = np.sin(phi) * np.sin(theta)
                dz = np.cos(phi)
                
                # Distance of split (e.g., 2.0 micrometers apart)
                split_dist = cell_radius_um * 0.7
                
                # Daughter 1
                d1_z = np.clip(p_z + dz * split_dist, 0.0, (Z - 1) * sz)
                d1_y = np.clip(p_y + dy * split_dist, 0.0, (Y - 1) * sy)
                d1_x = np.clip(p_x + dx * split_dist, 0.0, (X - 1) * sx)
                
                # Daughter 2
                d2_z = np.clip(p_z - dz * split_dist, 0.0, (Z - 1) * sz)
                d2_y = np.clip(p_y - dy * split_dist, 0.0, (Y - 1) * sy)
                d2_x = np.clip(p_x - dx * split_dist, 0.0, (X - 1) * sx)
                
                # Add drift/diffusion
                noise = np.random.normal(0, diffusion_um, (2, 3))
                d1_z = np.clip(d1_z + noise[0, 0], 0.0, (Z - 1) * sz)
                d1_y = np.clip(d1_y + noise[0, 1], 0.0, (Y - 1) * sy)
                d1_x = np.clip(d1_x + noise[0, 2], 0.0, (X - 1) * sx)
                
                d2_z = np.clip(d2_z + noise[1, 0], 0.0, (Z - 1) * sz)
                d2_y = np.clip(d2_y + noise[1, 1], 0.0, (Y - 1) * sy)
                d2_x = np.clip(d2_x + noise[1, 2], 0.0, (X - 1) * sx)
                
                current_cells.append((d1_id, d1_z, d1_y, d1_x))
                current_cells.append((d2_id, d2_z, d2_y, d2_x))
                
                nodes_list.append({'node_id': d1_id, 't': t, 'z': int(round(d1_z / sz)), 'y': int(round(d1_y / sy)), 'x': int(round(d1_x / sx))})
                nodes_list.append({'node_id': d2_id, 't': t, 'z': int(round(d2_z / sz)), 'y': int(round(d2_y / sy)), 'x': int(round(d2_x / sx))})
                
                edges_list.append({'source_id': parent_id, 'target_id': d1_id})
                edges_list.append({'source_id': parent_id, 'target_id': d2_id})
            else:
                # Regular motion
                child_id = next_node_id
                next_node_id += 1
                
                noise = np.random.normal(0, diffusion_um, 3)
                c_z = np.clip(p_z + noise[0], 0.0, (Z - 1) * sz)
                c_y = np.clip(p_y + noise[1], 0.0, (Y - 1) * sy)
                c_x = np.clip(p_x + noise[2], 0.0, (X - 1) * sx)
                
                current_cells.append((child_id, c_z, c_y, c_x))
                nodes_list.append({'node_id': child_id, 't': t, 'z': int(round(c_z / sz)), 'y': int(round(c_y / sy)), 'x': int(round(c_x / sx))})
                edges_list.append({'source_id': parent_id, 'target_id': child_id})
                
        cell_history[t] = current_cells

    # Generate 3D+time intensity images
    images = np.zeros((T, Z, Y, X), dtype=np.float32)
    
    # Coordinates grid in voxel index space
    grid_z, grid_y, grid_x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    
    for t in range(T):
        # We build the volume for time t by summing Gaussian intensity profiles for all cells at time t
        t_cells = cell_history[t]
        volume = np.zeros((Z, Y, X), dtype=np.float32)
        
        for _, c_z, c_y, c_x in t_cells:
            # Voxel center coordinate
            vz = c_z / sz
            vy = c_y / sy
            vx = c_x / sx
            
            # Radii in voxel units
            rz = cell_radius_um / sz
            ry = cell_radius_um / sy
            rx = cell_radius_um / sx
            
            # Sub-volume to draw in (for efficiency, only draw in local neighborhood of size 3 * r)
            z_start = int(max(0, vz - 3 * rz))
            z_end = int(min(Z, vz + 3 * rz + 1))
            y_start = int(max(0, vy - 3 * ry))
            y_end = int(min(Y, vy + 3 * ry + 1))
            x_start = int(max(0, vx - 3 * rx))
            x_end = int(min(X, vx + 3 * rx + 1))
            
            if z_start >= z_end or y_start >= y_end or x_start >= x_end:
                continue
                
            local_z = grid_z[z_start:z_end, y_start:y_end, x_start:x_end]
            local_y = grid_y[z_start:z_end, y_start:y_end, x_start:x_end]
            local_x = grid_x[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Squared distance normalized by radii
            dist_sq = ((local_z - vz) / rz) ** 2 + ((local_y - vy) / ry) ** 2 + ((local_x - vx) / rx) ** 2
            
            # Gaussian profile
            blob = np.exp(-0.5 * dist_sq)
            volume[z_start:z_end, y_start:y_end, x_start:x_end] += blob
            
        # Add background and noise
        volume = np.clip(volume, 0.0, 1.0)
        noise = np.random.normal(0, noise_sigma, volume.shape)
        volume_noisy = volume + noise
        images[t] = np.clip(volume_noisy, 0.0, 1.0)
        
    nodes_df = pd.DataFrame(nodes_list)
    edges_df = pd.DataFrame(edges_list)
    
    return images, nodes_df, edges_df

if __name__ == '__main__':
    # Test generation
    images, nodes, edges = generate_synthetic_dataset(T=3, Z=16, Y=64, X=64)
    print("Images shape:", images.shape)
    print("Nodes count:", len(nodes))
    print("Edges count:", len(edges))
    print(nodes.head())
    print(edges.head())
