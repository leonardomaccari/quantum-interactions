import optuna
import compute_triplets
import compute_baseline


def compare_experiments(data, bins_per_dim):
    # 1. Compute raw counts
    rvalue = compute_triplets.compute_triplets_numba(data, bins_per_dim)
    true_samples = rvalue['data']
    hist_true = true_samples.sum(axis=0)
    
    norm_factor = 0.001
    rvalue_norm = compute_baseline.compute_triplets_numba_norm(data, bins_per_dim, 
                                                               rvalue['power'], 
                                                               rvalue['max_mod'], 
                                                               norm_factor_norm=norm_factor)
    rvalue_cross = compute_baseline.compute_triplets_numba_cross(data, bins_per_dim, 
                                                                 rvalue['power'], 
                                                                 rvalue['max_mod'], 
                                                                 norm_factor_cross=norm_factor)
    
    # Reconstruct un-downsampled background shape
    raw_base_samples = (rvalue_norm['data'] + rvalue_cross['data'])
    hist_base_raw = raw_base_samples.sum(axis=0)
    
    # 2. Compute Normalization Weight (Scale background area to match data area)
    total_true_triplets = hist_true.sum()
    total_base_triplets = hist_base_raw.sum()
    weight = total_true_triplets / total_base_triplets
    
    # Properly scaled background
    hist_base = hist_base_raw * weight
    base_samples = raw_base_samples * weight

    # 3. Target Center Bin (Physical Zero)
    c = bins_per_dim // 2
    print(f"True Zero Bin: {hist_true[c,c,c]}")
    print(f"Scaled Background Zero Bin: {hist_base[c,c,c]:.2f}")
    print(f"Net Signal Zero Bin: {hist_true[c,c,c] - hist_base[c,c,c]:.2f}")

    # Fixed [c,c,c] slicing instead of [0,0,0]
    return hist_true - hist_base, true_samples[:, c, c, c], base_samples[:, c, c, c]

def run_optuna_pipeline(data, compute_triplets_fn, compute_cross_fn, 
                        compute_norm_fn, n_trials=50, train_ratio=0.8, 
                        norm_factor_cross=0.001, norm_factor_norm=0.001):
    """
    Optimizes grid resolution using Optuna with Area Normalization for the background.
    Enforces odd bin counts, multi-fidelity pruning, and an 80/20 train/test split.
    """
    # 1. Lock in 80/20 Train/Test Split
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    def objective(trial):
        # Only search space parameter: odd grid resolutions (15, 17, ..., 127)
        bins_per_dim = trial.suggest_int("bins_per_dim", 15, 127, step=2)
        c = bins_per_dim // 2  # Physical center / zero-momentum bin
        
        # Multi-fidelity checkpoints (10%, 30%, 100% of training pool)
        sample_ratios = [0.10, 0.30, 1.00]
        pure_signal = 0.0
        
        for step, ratio in enumerate(sample_ratios):
            n_sub = max(1, int(len(train_data) * ratio))
            sub_data = train_data[:n_sub]
            
            # 1. Compute True Data
            res_true = compute_triplets_fn(sub_data, bins_per_dim=bins_per_dim)
            true_samples = res_true['data']
            
            total_true_triplets = true_samples.sum()
            signal_zeroes = true_samples[:, c, c, c].sum()
            
            # Retrieve scaling parameters
            power = res_true.get('power', 0)
            max_mod = res_true.get('max_mod', 1)
            
            # 2. Compute Raw Sampled Background
            res_cross = compute_cross_fn(
                sub_data, bins_per_dim, power, max_mod, norm_factor_cross=norm_factor_cross
            )
            res_norm = compute_norm_fn(
                sub_data, bins_per_dim, power, max_mod, norm_factor_norm=norm_factor_norm
            )
            
            raw_base_samples = res_cross['data'] + res_norm['data']
            total_base_triplets = raw_base_samples.sum()
            
            # 3. Direct Area Normalization (Shape matching)
            # Prevent division by zero early in search if a grid is too coarse/sparse
            if total_base_triplets > 0:
                weight = total_true_triplets / total_base_triplets
            else:
                weight = 0.0
            
            # Scale background zeroes using the computed weight
            raw_base_zeroes = raw_base_samples[:, c, c, c].sum()
            base_zeroes = raw_base_zeroes * weight
            
            # Net Pure Signal in zero bin
            pure_signal = signal_zeroes - base_zeroes
            
            # Report progress for Hyperband pruning
            trial.report(pure_signal, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return pure_signal

    # 2. Setup Study with Multi-Fidelity Pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=3,
            reduction_factor=3
        )
    )
    
    # 3. Optimize on Training Set
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n[Optuna] Optimization Complete.")
    print(f"Optimal Bins Per Dim: {study.best_params['bins_per_dim']}")
    print(f"Best Training Pure Signal: {study.best_value:.2f}")
    
    return study, train_data, test_data

def optimize(data, n_trials=50, train_ratio=0.8):
    study, train_data, test_data = run_optuna_pipeline(data,
        compute_triplets_fn=compute_triplets.compute_triplets_numba,
        compute_cross_fn=compute_baseline.compute_triplets_numba_cross,
        compute_norm_fn=compute_baseline.compute_triplets_numba_norm,
        n_trials=n_trials,train_ratio=train_ratio)

    # Access optimal parameters
    best_bins = study.best_params["bins_per_dim"]


    print(f"Optimal Grid: {best_bins}x{best_bins}x{best_bins}")