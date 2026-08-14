import numpy as np
from numba import njit
from sklearn.cluster import KMeans

# =====================================================================
# 1. NUMBA KERNEL: I leave the definition but I don't use numba anymore
# as the amount of operations per experiment is small
# =====================================================================
@njit
def _aggregate_voxel_means_numba(points, bins_per_dim, min_val, max_val):
    n = len(points)
    range_val = max_val - min_val
    
    voxel_sums = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim, 3), dtype=np.float64)
    voxel_counts = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), dtype=np.int32)
    
    for i in range(n):
        ix = int(((points[i, 0] - min_val) / range_val) * bins_per_dim)
        iy = int(((points[i, 1] - min_val) / range_val) * bins_per_dim)
        iz = int(((points[i, 2] - min_val) / range_val) * bins_per_dim)
        
        # Clamp boundaries
        if ix >= bins_per_dim: ix = bins_per_dim - 1
        if iy >= bins_per_dim: iy = bins_per_dim - 1
        if iz >= bins_per_dim: iz = bins_per_dim - 1
        
        if ix >= 0 and iy >= 0 and iz >= 0:
            voxel_sums[ix, iy, iz, 0] += points[i, 0]
            voxel_sums[ix, iy, iz, 1] += points[i, 1]
            voxel_sums[ix, iy, iz, 2] += points[i, 2]
            voxel_counts[ix, iy, iz] += 1
            
    return voxel_sums, voxel_counts


# =====================================================================
# 2. MAIN FUNCTION: bin each point in all the experiments. 
# Then compute the average per bin. Then quantize the averages globally
# using k-means 
# =====================================================================
def quantize_dataset_globally(data, k_clusters, bins_per_dim=32, min_val=-1.0, max_val=1.0):
    exp_cached_voxels = []
    global_voxel_pool_list = []
    
    print("Pass 1: Computing voxel averages...")
    
    for exp in data:
        # Standardize input shape
        pts = np.array(exp, dtype=np.float64)
        if pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.T
        pts = np.ascontiguousarray(pts)
        
        # Local normalization
        centered = pts - np.mean(pts, axis=0)
        max_mod = np.max(np.abs(centered))
        norm_pts = centered / max_mod if max_mod > 0 else centered
            
        # Numba kernel
        voxel_sums, voxel_counts = _aggregate_voxel_means_numba(
            norm_pts, bins_per_dim, min_val, max_val
        )
        
        # Extract active voxels
        active_mask = voxel_counts > 0
        active_indices = np.argwhere(active_mask)
        
        if len(active_indices) > 0:
            active_sums = voxel_sums[active_mask]
            active_counts = voxel_counts[active_mask][:, np.newaxis]
            active_means = active_sums / active_counts
            
            global_voxel_pool_list.append(active_means)
        else:
            active_means = np.empty((0, 3))
            
        exp_cached_voxels.append((active_indices, active_means))
                
    if not global_voxel_pool_list:
        raise ValueError("No active voxels were found.")
        
    # Fit KMeans globally
    global_voxel_pool = np.vstack(global_voxel_pool_list)
    print(f"Pooled {len(global_voxel_pool)} active voxel averages. Fitting KMeans (K={k_clusters})...")
    
    global_kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    global_kmeans.fit(global_voxel_pool)
    
    # Print assigned states and their frequencies
    unique_states, counts = np.unique(global_kmeans.labels_, return_counts=True)
    print("\n--- Global State Assignment Summary ---")
    for state, count in zip(unique_states, counts):
        print(f"State {state:2d}: {count:6d} voxels")
    print("---------------------------------------\n")

    # Assign states
    print("Pass 2: Quantizing voxels...")
    all_experiment_matrices = []
    
    for active_indices, active_means in exp_cached_voxels:
        state_matrix = np.full((bins_per_dim, bins_per_dim, bins_per_dim), fill_value=-1, dtype=np.int32)
        
        if len(active_indices) > 0:
            assigned_states = global_kmeans.predict(active_means)
            for idx, (ix, iy, iz) in enumerate(active_indices):
                state_matrix[ix, iy, iz] = assigned_states[idx]
                
        all_experiment_matrices.append(state_matrix)
    
    return all_experiment_matrices, global_kmeans