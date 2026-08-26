import optuna
import compute_triplets
import compute_baseline

def compare_experiments(data, bins_per_dim):
    rvalue = compute_triplets.compute_triplets_numba(data, 
                                                    bins_per_dim)
    true_samples = rvalue['data']
    hist_true = rvalue['data'].sum(axis=0)
    rvalue_norm = compute_baseline.compute_triplets_numba_norm(data, 
                                                    bins_per_dim,
                                                    rvalue['power'],
                                                    rvalue['max_mod'])
    norm_samples = rvalue_norm['data']
    rvalue_cross = compute_baseline.compute_triplets_numba_cross(data, 
                                                    bins_per_dim,
                                                    rvalue['power'],
                                                    rvalue['max_mod'])
    base_samples = rvalue_cross['data']

    hist_base = rvalue_norm['data'].sum(axis=0)
    hist_base += rvalue_cross['data'].sum(axis=0)
    return hist_true - hist_base, true_samples[:,0,0,0], norm_samples[:,0,0,0] + base_samples[:,0,0,0]

