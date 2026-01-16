#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:28:37 2026

@author: leonardo
"""

import numpy as np
from itertools import combinations
from numba import njit, prange
import tqdm


def check_precision(data):
    """ Analyse the points and extract bounds for the precision of the 
    numbers """
    max_digits = 0
    max_mod = 0
    def to_string(n):
        return len(str(abs(n)).split('.')[-1])
        
    for x,y,z in tqdm.tqdm(data):
        max_digits = max(max_digits, 
                         max([to_string(k) for k in x+y+z])) # the digits of 
                                                             # each sample 
                                                            
        max_mod = max(max_mod, max([abs(k) for k in x+y+z]))  # max modulo
        
    print(f'largest modulo: {max_mod}, largest string length: {max_digits}')
    return max_digits, max_mod                


@njit(parallel=True)
def fill_histogram_numba(points, bins_per_dim, min_val, max_val):
    """
    Numba-accelerated kernel for triplet summation and binning.
    Uses parallel range (prange) to utilize all CPU cores.
    """
    n = len(points)
    # Sum of 3 points range
    sum_min = min_val * 3
    sum_max = max_val * 3
    range_width = sum_max - sum_min
    
    # Initialize local histogram for this experiment
    hist = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), dtype=np.uint64)
    
    # We iterate manually to avoid itertools overhead in Numba
    for i in prange(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Calculate sums
                sx = points[i, 0] + points[j, 0] + points[k, 0]
                sy = points[i, 1] + points[j, 1] + points[k, 1]
                sz = points[i, 2] + points[j, 2] + points[k, 2]
                
                # Map to bin indices
                ix = int(((sx - sum_min) / range_width) * bins_per_dim)
                iy = int(((sy - sum_min) / range_width) * bins_per_dim)
                iz = int(((sz - sum_min) / range_width) * bins_per_dim)
                
                # Boundary check (handle edge case where sum == sum_max)
                if ix >= bins_per_dim: ix = bins_per_dim - 1
                if iy >= bins_per_dim: iy = bins_per_dim - 1
                if iz >= bins_per_dim: iz = bins_per_dim - 1
                
                if ix >= 0 and iy >= 0 and iz >= 0:
                    # Numba handles thread-safe increments in parallel loops
                    hist[ix, iy, iz] += 1
    return hist

def compute_triplets_numba(data, bins_per_dim=100):
    print('pre-parsing data...')
    power, max_mod = check_precision(data)
    master_hist = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), 
                           dtype=np.uint64)
    print('computing all triplets...')
    for i, exp in tqdm.tqdm(enumerate(data), total=len(data)):
        # Convert to (N, 3) int64 array 
        points = np.array(exp, dtype=np.float64)*10**power
        points = np.rint(points).T
        
        # Call the JIT-compiled kernel
        exp_hist = fill_histogram_numba(points, bins_per_dim, -max_mod, max_mod)
        master_hist += exp_hist
        
    return master_hist