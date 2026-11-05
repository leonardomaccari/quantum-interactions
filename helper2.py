"""Third-order factorial cumulants with experiment and sampling normalization.

Use ``compare_experiments(data, bins_per_dim)`` as in helper_functions.
The original counting kernels and plotting function are reused unchanged.
Experiments must be independent realizations under comparable conditions.
"""

import numpy as np

import compute_baseline
import compute_triplets
from helper_functions import plot_heatmaps


def _inclusion_probabilities(multiplicities, fraction, *, pair):
    """Match the kernels' fixed-size, without-replacement external sampling."""
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Sampling fractions must be finite and in (0, 1].")
    external = multiplicities.sum() - multiplicities
    sampled = np.array([int(round(int(n) * fraction)) for n in external])
    required = 2 if pair else 1
    if np.any(sampled < required):
        raise ValueError(
            f"Each experiment needs at least {required} sampled external atoms; "
            "increase the sampling fraction (up to 1.0)."
        )
    external = external.astype(np.float64)
    sampled = sampled.astype(np.float64)
    if pair:
        return sampled * (sampled - 1) / (external * (external - 1))
    return sampled / external


def normalize_histograms(
    true_raw, cross_raw, norm_raw, multiplicities,
    *, norm_factor_cross=0.001, norm_factor_norm=0.001,
):
    """Return corrected true, pair-plus-single, and independent samples.

    Inputs are raw kernel histograms of shape (E, B, B, B), with one
    multiplicity per experiment. Sampling fractions must match the kernel
    calls that produced the histograms. Each returned array retains the
    experiment axis; its mean estimates A, B, or D respectively, where
    C3 = A - 3*B + 2*D. No histogram is normalized to unit integral.
    """
    counts = np.asarray(multiplicities)
    if (
        counts.ndim != 1 or counts.size < 3
        or not np.all(np.isfinite(counts))
        or np.any(counts < 0) or np.any(counts != np.floor(counts))
    ):
        raise ValueError("Provide nonnegative integer multiplicities for at least 3 experiments.")
    counts = counts.astype(np.int64)
    arrays = [np.asarray(h, dtype=np.float64) for h in (true_raw, cross_raw, norm_raw)]
    shape = arrays[0].shape
    if len(shape) != 4 or shape[0] != counts.size or any(h.shape != shape for h in arrays):
        raise ValueError("All histograms must have matching shape (E, B, B, B).")

    q = _inclusion_probabilities(counts, norm_factor_cross, pair=False)
    r = _inclusion_probabilities(counts, norm_factor_norm, pair=True)
    experiments = counts.size
    # True kernels count unordered triples (6 permutations). Mixed kernels
    # count an unordered pair (2 permutations); experiment ordering is
    # accounted for by (E-1) and (E-1)*(E-2). The caller averages over E.
    true_samples = 6 * arrays[0]
    cross_samples = 2 * arrays[1] / ((experiments - 1) * q[:, None, None, None])
    norm_samples = 2 * arrays[2] / (
        (experiments - 1) * (experiments - 2) * r[:, None, None, None]
    )
    return true_samples, cross_samples, norm_samples


def relative_correlation(connected, independent):
    """Return C3/D binwise, with NaN wherever D is zero or negative.

    ``independent`` is the ensemble-mean corrected norm histogram (D),
    not the combined background 3*B - 2*D. Sparse bins may be noisy.
    """
    connected = np.asarray(connected, dtype=np.float64)
    independent = np.asarray(independent, dtype=np.float64)
    if connected.shape != independent.shape:
        raise ValueError("Connected and independent histograms must have the same shape.")
    return np.divide(
        connected, independent, out=np.full_like(connected, np.nan), where=independent > 0
    )


def compare_experiments(
    data, bins_per_dim, *, norm_factor_cross=0.001, norm_factor_norm=0.001, plot=True,
):
    """Compute C3 with corrected counting weights, preserving the original API.

    Return (connected_histogram, true_zero, base_zero). The last two arrays
    contain corrected center-bin contributions per experiment; the mean
    of their difference equals the connected histogram's center bin.
    Results are ordered-triplet yields, not probability distributions.
    For small datasets, use larger sampling fractions (e.g. 1.0).
    Set plot=False to avoid creating a figure or writing heatmap_plot.png.
    """
    if not isinstance(bins_per_dim, (int, np.integer)) or bins_per_dim <= 0:
        raise ValueError("bins_per_dim must be a positive integer.")
    multiplicities = []
    for experiment in data:
        points = np.asarray(experiment)
        if points.ndim != 2 or points.shape[0] != 3:
            raise ValueError("Each experiment must have shape (3, number_of_atoms).")
        multiplicities.append(points.shape[1])
    if len(multiplicities) < 3:
        raise ValueError("At least 3 experiments are required for distinct-shot mixing.")
    counts = np.asarray(multiplicities, dtype=np.int64)
    # Validate before calling the expensive kernels.
    _inclusion_probabilities(counts, norm_factor_cross, pair=False)
    _inclusion_probabilities(counts, norm_factor_norm, pair=True)

    result = compute_triplets.compute_triplets_numba(data, bins_per_dim)
    cross_raw = compute_baseline.compute_triplets_numba_cross(
        data, bins_per_dim, result['power'], result['max_mod'],
        norm_factor_cross=norm_factor_cross,
    )['data']
    norm_raw = compute_baseline.compute_triplets_numba_norm(
        data, bins_per_dim, result['power'], result['max_mod'],
        norm_factor_norm=norm_factor_norm,
    )['data']
    true_samples, cross_samples, norm_samples = normalize_histograms(
        result['data'], cross_raw, norm_raw, counts,
        norm_factor_cross=norm_factor_cross, norm_factor_norm=norm_factor_norm,
    )
    background = 3 * cross_samples - 2 * norm_samples
    connected = true_samples.mean(axis=0) - background.mean(axis=0)
    center = bins_per_dim // 2
    true_zero = true_samples[:, center, center, center]
    base_zero = background[:, center, center, center]

    print(f"Mean true ordered-triplet yield in center bin: {true_zero.mean():.6g}")
    print(f"Mean background in center bin: {base_zero.mean():.6g}")
    print(f"Connected center-bin yield: {connected[center, center, center]:.6g}")
    if plot:
        plot_heatmaps(connected, title="Connected three-particle histogram (C3)")
    return connected, true_zero, base_zero
