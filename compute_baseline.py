#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17∶52∶22 2026

@author: leonardo & Maxime
"""

import numpy as np
from itertools import combinations
from numba import njit, prange
import tqdm
from multiprocessing import Pool
import math



# -- Normalisation : cross term
@njit(parallel=True)
def fill_histogram_numba_cross(
    i,
    flat_points,
    offsets,
    bins_per_dim,
    min_val,
    max_val,
    norm_factor_cross
):
    # experiment i slice
    si = offsets[i]
    ei = offsets[i + 1]
    points_i = flat_points[si:ei] - min_val
    ni = ei - si

    # number of atoms excluding shot i
    n_total = len(flat_points)
    n_excl = n_total - ni

    # ---- keep a fraction of atoms from shots \neq i ----
    sample_size = int(round(n_excl * norm_factor_cross))
    chosen_idxs = np.random.choice(n_excl, size=sample_size, replace=False)

    Nshot = len(offsets) - 1

    sum_min = min_val * 3
    sum_max = max_val * 3
    range_width = sum_max - sum_min

    hist = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), dtype=np.uint64)

    shift = 0
    if int(math.log2(bins_per_dim)) == math.log2(bins_per_dim):
        prec = math.ceil(math.log2(range_width))
        shift = prec - int(math.log2(bins_per_dim))
    
    # We iterate manually to avoid itertools overhead in Numba
    # We iterate over pairs (of atoms) for better parallelisation
    M = ni * (ni - 1) // 2
    for p in prange(M): # parallel
        # recover (ai, aj) from p
        ai = int((2*ni - 1 - np.sqrt((2*ni - 1)**2 - 8*p)) // 2)
        aj = p - ai*(2*ni - ai - 1)//2 + ai + 1
        
        xi, yi, zi = points_i[ai]
        xj, yj, zj = points_i[aj]
        
        for uk in chosen_idxs:    
            # ---- virtual concatenation mapping ----
            ak = uk + (uk >= si) * ni
            
            xk, yk, zk = flat_points[ak]            
            xk -= min_val
            yk -= min_val
            zk -= min_val
            
        # for pk in points_all_but_i:
        #     xk, yk, zk = pk
            
            sx = xi + xj + xk
            sy = yi + yj + yk
            sz = zi + zj + zk

            if shift:
                ix = sx >> shift
                iy = sy >> shift
                iz = sz >> shift
            else:
                ix = (sx * bins_per_dim) // range_width
                iy = (sy * bins_per_dim) // range_width
                iz = (sz * bins_per_dim) // range_width

            if ix >= bins_per_dim: ix = bins_per_dim - 1
            if iy >= bins_per_dim: iy = bins_per_dim - 1
            if iz >= bins_per_dim: iz = bins_per_dim - 1

            if ix >= 0 and iy >= 0 and iz >= 0:
                hist[ix, iy, iz] += 1

    return hist
    
def compute_triplets_numba_cross(data, bins_per_dim=64, norm_factor_cross=0.001):
    print('pre-parsing data...')
    power, max_mod = check_precision(data)
    # master_hist = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), 
    #                        dtype=np.uint64)
    master_hist = np.zeros((len(data), bins_per_dim, bins_per_dim, bins_per_dim), 
                           dtype=np.uint64)
    max_mod = int(max_mod*10**power)
    print('computing all triplets...')

    # flatten data and keep shot indices in separate array
    flat_points = []
    offsets = [0]    
    for exp in data:
        arr = np.asarray(exp, dtype=np.float64) * 10**power
        arr = arr.astype(np.int64).T
        flat_points.append(arr)
        offsets.append(offsets[-1] + arr.shape[0])
    
    flat_points = np.vstack(flat_points)   # (N_total, 3)
    offsets = np.array(offsets, dtype=np.int64)

    for i, exp in tqdm.tqdm(enumerate(data), total=len(data)):
        # Call the JIT-compiled kernel
        
        exp_hist = fill_histogram_numba_cross(i, flat_points, offsets,
                                              bins_per_dim, -max_mod, max_mod,
                                              norm_factor_cross)
        # master_hist += exp_hist
        master_hist[i] = exp_hist
    rvalue = {'data':master_hist, 'max_mod':max_mod, 
              'bins_per_dim':bins_per_dim}
    return rvalue



# -- Normalisation : uncorrelated term
@njit(parallel=True)
def fill_histogram_numba_norm(
    i,
    flat_points,
    offsets,
    shot_index,
    bins_per_dim,
    min_val,
    max_val,
    norm_factor_norm
):

    si = offsets[i]
    ei = offsets[i + 1]

    # points in shot i
    points_i = flat_points[si:ei] - min_val
    ni = ei - si

    # number of atoms excluding shot i
    n_total = len(flat_points)
    n_excl = n_total - ni

    # ---- keep a fraction of atoms from shots \neq i ----
    sample_size = int(round(n_excl * norm_factor_norm))
    chosen_idxs = np.random.choice(n_excl, size=sample_size, replace=False)

    sum_min = min_val * 3
    sum_max = max_val * 3
    range_width = sum_max - sum_min

    hist = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), dtype=np.uint64)

    # optional fast bit binning
    shift = 0
    if int(math.log2(bins_per_dim)) == math.log2(bins_per_dim):
        prec = math.ceil(math.log2(range_width))
        shift = prec - int(math.log2(bins_per_dim))

    # triangular enumeration on sampled atoms
    m = sample_size
    M = m * (m - 1) // 2
    for p in prange(M):
        # invert triangular index
        aj_s = int((2*m - 1 - math.sqrt((2*m - 1)**2 - 8*p)) // 2)
        ak_s = p - aj_s*(2*m - aj_s - 1)//2 + aj_s + 1

        # map to flattened array index
        uj = chosen_idxs[aj_s]
        uk = chosen_idxs[ak_s]

        # ---- virtual concatenation mapping ----
        aj = uj + (uj >= si) * ni
        ak = uk + (uk >= si) * ni

        # remove same shot atoms
        if shot_index[aj] == shot_index[ak]:
            continue

        xj, yj, zj = flat_points[aj]
        xk, yk, zk = flat_points[ak]

        xj -= min_val
        yj -= min_val
        zj -= min_val
        xk -= min_val
        yk -= min_val
        zk -= min_val

        for pi in points_i:
            xi, yi, zi = pi

            sx = xi + xj + xk
            sy = yi + yj + yk
            sz = zi + zj + zk

            if shift:
                ix = sx >> shift
                iy = sy >> shift
                iz = sz >> shift
            else:
                ix = (sx * bins_per_dim) // range_width
                iy = (sy * bins_per_dim) // range_width
                iz = (sz * bins_per_dim) // range_width

            if ix >= bins_per_dim: ix = bins_per_dim - 1
            if iy >= bins_per_dim: iy = bins_per_dim - 1
            if iz >= bins_per_dim: iz = bins_per_dim - 1

            if ix >= 0 and iy >= 0 and iz >= 0:
                hist[ix, iy, iz] += 1

    return hist


def compute_triplets_numba_norm(data, bins_per_dim=64, norm_factor_norm=0.001):
    print('pre-parsing data...')
    power, max_mod = check_precision(data)
    # master_hist = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), 
    #                        dtype=np.uint64)
    master_hist = np.zeros((len(data), bins_per_dim, bins_per_dim, bins_per_dim), 
                           dtype=np.uint64)
    max_mod = int(max_mod*10**power)
    print('computing all triplets...')

    # flatten data and keep shot indices in separate array
    flat_points = []
    offsets = [0]    
    for exp in data:
        arr = np.asarray(exp, dtype=np.float64) * 10**power
        arr = arr.astype(np.int64).T
        flat_points.append(arr)
        offsets.append(offsets[-1] + arr.shape[0])
    
    flat_points = np.vstack(flat_points)   # (N_total, 3)
    offsets = np.array(offsets, dtype=np.int64)
    shot_index = np.empty(len(flat_points), dtype=np.int32)    
    for s in range(len(offsets)-1):
        shot_index[offsets[s]:offsets[s+1]] = s

    for i, exp in tqdm.tqdm(enumerate(data), total=len(data)):
        # Call the JIT-compiled kernel
        
        exp_hist = fill_histogram_numba_norm(i, flat_points, offsets, shot_index,
                                             bins_per_dim, -max_mod, max_mod,
                                             norm_factor_norm)
        # master_hist += exp_hist
        master_hist[i] = exp_hist
    rvalue = {'data':master_hist, 'max_mod':max_mod, 
              'bins_per_dim':bins_per_dim}
    return rvalue