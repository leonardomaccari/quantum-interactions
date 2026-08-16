import numpy as np
from numba import njit
from sklearn.cluster import KMeans
import subprocess
import os
import re

# =====================================================================
# 1. NUMBA KERNEL: I leave the definition but I don't use numba anymore
# as the amount of operations per experiment is small
# =====================================================================
@njit
def _aggregate_voxel_means_numba(points, bins_per_dim, min_val, max_val):
    n = len(points)
    range_val = max_val - min_val
    
    voxel_sums = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim, 3), dtype=np.float64)
    voxel_counts = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), dtype=np.int16)
    
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
                        norm_pts, bins_per_dim, min_val, max_val)
        
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
    all_experiment_sparse_matrices = []
    for active_indices, active_means in exp_cached_voxels:
        if len(active_indices) > 0:
            assigned_states = global_kmeans.predict(active_means)
            # Combine [x, y, z] indices with the state ID into an (M, 4) array
            sparse_matrix = np.column_stack((active_indices, assigned_states)).astype(np.int16)
        else:
            sparse_matrix = np.empty((0, 4), dtype=np.int16)
            
        all_experiment_sparse_matrices.append(sparse_matrix)
        
    return all_experiment_sparse_matrices, global_kmeans

def generate_amie_facts_tsv(sparse_matrices, centroids, epsilon, output_filename="amie_facts.tsv"):
    """
    Takes the integer sparse matrices and float centroids, computes the zero-sum 
    triplets based on epsilon, and writes a TSV file for AMIE.
    
    sparse_matrices : list of (M, 4) int16 arrays [ix, iy, iz, state_id]
    centroids       : (K, 3) float array of cluster centers
    epsilon         : float, threshold for the zero-sum condition
    """
    facts = []
    
    # =====================================================================
    # 1. GENERATE EXPERIMENT FACTS (Data -> States)
    # =====================================================================
    for exp_id, matrix in enumerate(sparse_matrices):
        if len(matrix) > 0:
            # Extract unique state_ids (column 3) present in this experiment
            unique_states = np.unique(matrix[:, 3])
            for state_id in unique_states:
                # Format: Subject    Predicate    Object
                facts.append(f"exp_{exp_id}\tcontains\tstate_{state_id}")
                
    # =====================================================================
    # 2. GENERATE BACKGROUND KNOWLEDGE (Zero-Sum Triplets)
    # =====================================================================
    # Vectorized 3D broadcasting to compute all possible A + B + C combinations
    # sums shape: (K, K, K, 3)
    # result is that sums[i,j,k] = centroids[i] + centroids[j] + centroids[k] 
    # (each element is a list of 3 values x/y/z)

    sums = centroids[:, None, None, :] + centroids[None, :, None, :] + centroids[None, None, :, :]
    
    # Compute vector norm on the: magnitudes[i,j,k] = norm of the sum of centroids
    magnitudes = np.linalg.norm(sums, axis=-1)
    
    # Find all indices where the sum magnitude is less than epsilon
    valid_i, valid_j, valid_k = np.where(magnitudes < epsilon)
    
    # Filter to unique combinations (i <= j <= k) to avoid duplicate permutations 
    # (e.g., we want 1-2-3, but not 2-1-3 or 3-2-1)
    unique_mask = (valid_i <= valid_j) & (valid_j <= valid_k)
    triplet_i = valid_i[unique_mask]
    triplet_j = valid_j[unique_mask]
    triplet_k = valid_k[unique_mask]

    #TODO: abort if there are no useful triplets

    # Because AMIE uses binary relations (Subject-Predicate-Object), 
    # the standard way to represent a 3-way connection is a "Hyperedge" (a triplet node).
    for t_id, (s_A, s_B, s_C) in enumerate(zip(triplet_i, triplet_j, triplet_k)):
        triplet_name = f"triplet_{t_id}"
        
        # Connect the states to the shared triplet ID
        facts.append(f"state_{s_A}\tpart_of\t{triplet_name}")
        facts.append(f"state_{s_B}\tpart_of\t{triplet_name}")
        facts.append(f"state_{s_C}\tpart_of\t{triplet_name}")
        
    # =====================================================================
    # 3. WRITE TO TSV
    # =====================================================================
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(facts))
        f.write("\n")
        
    print(f"Exported {len(facts)} facts to {output_filename} (Epsilon: {epsilon:.4f})")
    return output_filename


def run_amie(tsv_filepath, amie_jar="amie/bin/amie4.0.0.jar", min_c=0.1, min_hc=0.1, maxad=4):
    """
    Executes AMIE 4.0.0 on the generated TSV facts.
    min_c: Minimum standard confidence
    min_hc: Minimum head coverage
    maxad: Maximum number of atoms in the rule (default in AMIE is 3, we often need 4 or 5)
    """
    cmd = [
        "java", "-jar", amie_jar,
        "-minc", str(min_c),
        "-minhc", str(min_hc),
        "-maxad", str(maxad),   
        tsv_filepath
    ]
    
    print(f"Launching AMIE: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("AMIE Error Log:\n", result.stderr)
        raise RuntimeError("AMIE execution failed.")
    print(result.stdout)
    return result.stdout

def canonicalize_rule(raw_rule):
    """
    Takes an AMIE rule and converts it into a sorted, directionless signature.
    This ensures that A + B => C and A + C => B evaluate to the exact same string.
    """
    # 1. Strip the implication arrow to treat the body and head equally
    rule_no_arrow = raw_rule.replace("=>", " ")
    
    # 2. Extract all atoms (e.g., "?a contains ?b")
    # AMIE separates atoms with double spaces
    atoms = [a.strip() for a in re.split(r'  +', rule_no_arrow.strip()) if a.strip()]
    
    canonical_atoms = []
    for atom in atoms:
        parts = atom.split()
        if len(parts) == 3:
            sub, pred, obj = parts
            
            # 3. Standardize variable names
            # We don't care if AMIE called it ?a or ?b, we just care about the shape.
            # We will replace all variables with a generic placeholder 'VAR'
            # to capture the purely structural relationship.
            # (If your rules contain constants like 'state_14', they will be preserved)
            sub_clean = "VAR" if sub.startswith("?") else sub
            obj_clean = "VAR" if obj.startswith("?") else obj
            
            # For symmetric relations (like interactions), alphabetical sorting 
            # prevents A-interacts-B from looking different than B-interacts-A
            if pred in ["interacts", "part_of"]:  
                elements = sorted([sub_clean, obj_clean])
                canonical_atoms.append(f"{pred}({elements[0]}, {elements[1]})")
            else:
                canonical_atoms.append(f"{pred}({sub_clean}, {obj_clean})")

    # 4. Sort all atoms alphabetically so the order of the rule doesn't matter
    canonical_atoms.sort()
    
    # 5. Join them back together into a single unique signature
    return " AND ".join(canonical_atoms)


def parse_and_deduplicate_amie(amie_stdout):
    """
    Parses AMIE output and merges symmetric/duplicate rules by averaging 
    their confidences and taking their max support.
    """
    unique_rules = {}
    
    for line in amie_stdout.splitlines():
        if not line.strip().startswith("?") or "=>" not in line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        print(parts)
        try:
            raw_rule = parts[0].strip()
            std_conf = float(parts[2])
            support = int(parts[4])
            
            # Generate the canonical directionless signature
            canonical_sig = canonicalize_rule(raw_rule)
            
            if canonical_sig in unique_rules:
                # If we've seen this physics interaction before (e.g. A+C=>B instead of A+B=>C)
                # We update the metrics. (Taking the max support and average confidence)
                unique_rules[canonical_sig]["support"] = max(unique_rules[canonical_sig]["support"], support)
                unique_rules[canonical_sig]["std_conf_sum"] += std_conf
                unique_rules[canonical_sig]["count"] += 1
            else:
                # First time seeing this interaction
                unique_rules[canonical_sig] = {
                    "raw_rule_example": raw_rule,
                    "canonical_signature": canonical_sig,
                    "support": support,
                    "std_conf_sum": std_conf,
                    "count": 1
                }
        except ValueError:
            continue
            
    # Calculate final averaged metrics and return as a list
    final_rules = []
    for sig, data in unique_rules.items():
        avg_conf = data["std_conf_sum"] / data["count"]
        final_rules.append({
            "canonical_signature": sig,
            "example_raw_rule": data["raw_rule_example"],
            "std_conf": avg_conf,
            "support": data["support"],
            "permutations_found": data["count"]
        })
        
    return sorted(final_rules, key=lambda x: x["std_conf"], reverse=True)


## wrapper function 

import os
import uuid
import json

def run_physics_discovery_pipeline(
    data, 
    bins_per_dim, 
    k_clusters, 
    epsilon, 
    cache_dict=None,
    amie_jar="AI/amie/bin/amie4.0.0.jar", 
    min_c=0.1, 
    min_hc=0.1, 
    maxad=6,
    base_output_dir="pipeline_runs"
):
    """
    Executes the full end-to-end toolchain and saves all intermediate data 
    into a unique temporary folder for auditing and debugging.
    """
    # 1. CREATE UNIQUE DIRECTORY FOR THIS RUN
    # Naming convention: b{bins}_k{k}_eps{epsilon}_{uuid}
    run_id = f"b{bins_per_dim}_k{k_clusters}_eps{epsilon:.3f}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"--- Starting Pipeline: Output saved to {run_dir} ---")
    
    # 2. QUANTIZATION & CACHE HANDLING
    if cache_dict is not None and (bins_per_dim, k_clusters) in cache_dict:
        sparse_matrices = cache_dict[(bins_per_dim, k_clusters)]["sparse_matrices"]
        centroids = cache_dict[(bins_per_dim, k_clusters)]["centroids"]
    else:
        sparse_matrices, kmeans = quantize_dataset_globally(
            data, 
            k_clusters=k_clusters, 
            bins_per_dim=bins_per_dim
        )
        centroids = kmeans.cluster_centers_
        if cache_dict is not None:
            cache_dict[(bins_per_dim, k_clusters)] = {
                "sparse_matrices": sparse_matrices,
                "centroids": centroids
            }

    # 3. GENERATE AMIE FACTS
    tsv_filepath = os.path.join(run_dir, "amie_facts.tsv")
    generate_amie_facts_tsv(
        sparse_matrices=sparse_matrices, 
        centroids=centroids, 
        epsilon=epsilon, 
        output_filename=tsv_filepath
    )

    # 4. RUN AMIE LOGIC MINER
    try:
        amie_stdout = run_amie(
            tsv_filepath=tsv_filepath, 
            amie_jar=amie_jar, 
            min_c=min_c, 
            min_hc=min_hc, 
            maxad=maxad
        )
        
        # Save raw AMIE output (pre-parsed rules and logs)
        raw_output_path = os.path.join(run_dir, "amie_raw_output.txt")
        with open(raw_output_path, "w", encoding="utf-8") as f:
            f.write(amie_stdout)
            
    except RuntimeError as e:
        print(f"AMIE failed. Check the directory {run_dir} for details.")
        return []

    # 5. PARSE & DEDUPLICATE RESULTS
    final_rules = parse_and_deduplicate_amie(amie_stdout)
    
    # Save the final parsed rules to a JSON file for easy loading later
    parsed_rules_path = os.path.join(run_dir, "parsed_rules.json")
    with open(parsed_rules_path, "w", encoding="utf-8") as f:
        json.dump(final_rules, f, indent=4)
        
    print(f"--- Pipeline Complete. Discovered {len(final_rules)} rules. ---")
    
    return final_rules