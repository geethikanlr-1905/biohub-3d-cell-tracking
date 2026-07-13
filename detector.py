import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

def detect_cells_3d(
    image,
    voxel_scale=(1.625, 0.40625, 0.40625),
    cell_radius_um=3.0,
    rel_threshold=0.20,
    min_dist_um=1.5
):
    """
    Anisotropic 3D Blob Detector using Difference of Gaussians (DoG)
    with robust 99th-percentile dynamic thresholding.
    
    Parameters:
    -----------
    image : numpy.ndarray
        3D volume of shape (Z, Y, X).
    voxel_scale : tuple of float
        (sz, sy, sx) physical voxel spacing in micrometers.
    cell_radius_um : float
        Expected physical cell nucleus radius in micrometers.
    rel_threshold : float
        Threshold factor relative to the 99th percentile of positive DoG values.
    min_dist_um : float
        Minimum physical distance between cell centroids in micrometers.
        
    Returns:
    --------
    centroids : list of (z, y, x) tuples
        Detected cell centroid voxel coordinates.
    """
    sz, sy, sx = voxel_scale
    
    # Min-max normalization
    img_min = np.min(image)
    img_max = np.max(image)
    denom = img_max - img_min
    if denom < 1e-5:
        denom = 1.0
    img_norm = (image - img_min) / denom
    
    # Compute anisotropic sigmas directly on voxel coordinates
    sigma_low = (cell_radius_um * 0.5 / sz, cell_radius_um * 0.5 / sy, cell_radius_um * 0.5 / sx)
    sigma_high = (cell_radius_um * 1.0 / sz, cell_radius_um * 1.0 / sy, cell_radius_um * 1.0 / sx)
    
    # Difference of Gaussians (DoG) filter
    img_smooth_low = gaussian_filter(img_norm, sigma=sigma_low)
    img_smooth_high = gaussian_filter(img_norm, sigma=sigma_high)
    dog = img_smooth_low - img_smooth_high
    
    # Robust dynamic threshold based on 99th percentile of positive DoG values
    pos_dog = dog[dog > 0]
    ref_val = np.percentile(pos_dog, 99) if len(pos_dog) > 0 else 1e-3
    threshold_abs = max(0.005, rel_threshold * ref_val)
    
    # Extract local peaks in voxel units
    peaks = peak_local_max(dog, min_distance=1, threshold_abs=threshold_abs, exclude_border=False)
    if len(peaks) == 0:
        return []
        
    # Apply physical distance non-maximum suppression (NMS)
    peaks_physical = []
    for p in peaks:
        pz, py, px = p
        peaks_physical.append((pz * sz, py * sy, px * sx, pz, py, px))
        
    # Sort by DoG intensity (descending)
    intensities = [dog[p[3], p[4], p[5]] for p in peaks_physical]
    sorted_idx = np.argsort(intensities)[::-1]
    
    keep = []
    keep_coords_physical = []
    
    for idx in sorted_idx:
        p = peaks_physical[idx]
        p_phys = np.array([p[0], p[1], p[2]])
        too_close = False
        for k_phys in keep_coords_physical:
            dist = np.sqrt(np.sum((p_phys - k_phys) ** 2))
            if dist < min_dist_um:
                too_close = True
                break
        if not too_close:
            keep.append((int(p[3]), int(p[4]), int(p[5])))
            keep_coords_physical.append(p_phys)
            
    return keep
