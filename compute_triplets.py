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
from multiprocessing import Pool
import math

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
    Uses parallel range (prange) to use all CPU cores.
    """
    n = len(points)
    # Sum of 3 points ranges
    sum_min = min_val * 3
    sum_max = max_val * 3
    range_width = sum_max - sum_min
    bin_width = int(range_width/bins_per_dim)
    # Initialize local histogram for this experiment
    hist = np.zeros((bins_per_dim, bins_per_dim, bins_per_dim), dtype=np.uint64)
    shift = 0
    if int(math.log2(bins_per_dim)) == math.log2(bins_per_dim):
        prec = math.ceil(math.log2(range_width))
        shift = prec - int(math.log2(bins_per_dim))
    points = points - min_val
    # We iterate manually to avoid itertools overhead in Numba
    for i in prange(n): # this runs parallel processes
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Calculate sums
                sx = points[i, 0] + points[j, 0] + points[k, 0]
                sy = points[i, 1] + points[j, 1] + points[k, 1]
                sz = points[i, 2] + points[j, 2] + points[k, 2]
                
                # Here we exploit the fact that binning is like right shifting
                # ex: 1110101 if we bin with 4 bins goes to 11, as only the 
                # most siginificant bits are preserved.
                # note however that the bins are differently ranged:
                # assuming that range_width has at most N bits, then the 
                # largest bin starts from 2**(N+1), while using the 
                # division method it starts exactly at range_width
                
                if shift:
                    ix = sx >> shift
                    iy = sy >> shift
                    iz = sz >> shift
                else:
                    ix = int((sx  / range_width) * bins_per_dim)
                    iy = int((sy  / range_width) * bins_per_dim)
                    iz = int((sz  / range_width) * bins_per_dim)
                    
                    # this seems to be slower
                    #ix = (sx - sum_min) // bin_width
                    #iy = (sy - sum_min) // bin_width
                    #iz = (sz - sum_min) // bin_width
                
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
    max_mod = int(max_mod*10**power)
    print('computing all triplets...')
    for i, exp in tqdm.tqdm(enumerate(data), total=len(data)):
        # Convert to (N, 3) int64 array 
        points = np.array(exp, dtype=np.float64)*10**power
        points = points.astype(np.int64).T
        # Call the JIT-compiled kernel

        exp_hist = fill_histogram_numba(points, bins_per_dim, -max_mod, 
                                        max_mod)
        master_hist += exp_hist
    rvalue = {'data':master_hist, 'max_mod':max_mod, 
              'bins_per_dim':bins_per_dim}
    return rvalue


### this is a variant that doesn't use numba and is much slower. The issue
### is that it does compute in advance all the indexes, while instead they 
### should be reused. 

def fill_histogram(exp_data, bins_per_dim, min_val, max_val, power):
    """
    Returns a 1D flattened bincount array.
    """

    points = np.array(exp_data, dtype=np.float64)*10**power
    points = np.rint(points).T
    
    n = len(points)
    
    sum_min, sum_max = min_val * 3, max_val * 3
    
    # Vectorized Triplet Sums, these are all indices of the triplets
    idx = np.array(list(combinations(range(n), 3)), dtype=np.int32)
    triplet_sums = points[idx[:, 0]] + points[idx[:, 1]] + points[idx[:, 2]]
    
    # Normalize sums to [0, 1] then scale to [0, bins_per_dim - 1]
    indices = ((triplet_sums - sum_min) / (sum_max - sum_min) * bins_per_dim).astype(np.int32)
    
    # Clip to ensure no index out of bounds due to float precision
    np.clip(indices, 0, bins_per_dim - 1, out=indices)
    
    # Flattened Indexing to use bincount, faster than histograms
    # flat_idx = ix * B^2 + iy * B + iz
    flat_idx = (indices[:, 0] * (bins_per_dim**2) + 
                indices[:, 1] * bins_per_dim + 
                indices[:, 2])
    
    return np.bincount(flat_idx, minlength=bins_per_dim**3).astype(np.uint64)


def compute_triplets(data, bins_per_dim=100):
    """ this variant is memory intentive and slower than the numba one """
    print('pre-parsing data...')
    power, max_mod = check_precision(data)
    
    total_bins = bins_per_dim**3
    master_hist = np.zeros(total_bins, dtype=np.uint64)
    
    print('computing all triplets...')
    
    with Pool(1) as pool:
        args = [(exp, bins_per_dim, int(-max_mod), int(max_mod), power) for exp in data]
        
        for result in pool.starmap(fill_histogram, tqdm.tqdm(args, 
                                                             total=len(args))):
            master_hist += result
            
    return master_hist.reshape((bins_per_dim, bins_per_dim, bins_per_dim))