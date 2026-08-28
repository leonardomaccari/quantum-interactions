import optuna
import compute_triplets
import compute_baseline
import numpy as np


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
            
            # Per-experiment total true triplets (Shape: E,)
            total_true_per_exp = true_samples.sum(axis=(1, 2, 3))
            
            # True zeroes per experiment, and the global sum
            signal_zeroes_per_exp = true_samples[:, c, c, c]
            signal_zeroes = signal_zeroes_per_exp.sum()
            
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
            
            # Per-experiment total base triplets (Shape: E,)
            total_base_per_exp = raw_base_samples.sum(axis=(1, 2, 3))
            
            # 3. Per-Experiment Area Normalization
            # Prevent division by zero for sparse grids using a mask
            weights = np.zeros_like(total_true_per_exp, dtype=float)
            valid_mask = total_base_per_exp > 0
            weights[valid_mask] = total_true_per_exp[valid_mask] / total_base_per_exp[valid_mask]
            
            # 4. Scale background zeroes PER EXPERIMENT using the weight array
            raw_base_zeroes_per_exp = raw_base_samples[:, c, c, c]
            base_zeroes_per_exp = raw_base_zeroes_per_exp * weights
            
            # Now we can safely sum the properly scaled zeroes to get the total
            base_zeroes = base_zeroes_per_exp.sum()
            
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
    print(study.trials_dataframe())


    print(f"Optimal Grid: {best_bins}x{best_bins}x{best_bins}")