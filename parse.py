#! /usr/bin/env python3

import pickle
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import itertools
import argparse
import unittest
import math
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
import compute_triplets 
import numba


class TestFunctions(unittest.TestCase):

    def test_angle_0(self, first=[1,0,0], second=[1,0,0]):
        self.assertAlmostEqual(angle_between(first, second), 0)    

    def test_angle_180(self, first=[1,0,0], second=[-1,0,0]):
        self.assertAlmostEqual(angle_between(first, second), math.pi)
    
    def test_angle_90(self, first=[1,0,0], second=[0,1,0]):
        self.assertAlmostEqual(angle_between(first, second), math.pi/2)

    def test_angle_90l(self):
        self.assertAlmostEqual(angle_between([0.7,0,0],[0,1,0]), math.pi/2)
        
    def test_angles_all(self):
        vectors = [[1,0,0], [0,1,0], [0,0,1]]         
        for (first, second) in itertools.combinations(vectors, 2):
            self.test_angle_90(first, second)
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def data_summary(data):
    mins = [10,10,10]
    maxes = [-10,-10,-10]
    all_x = []
    all_y = []
    all_z = []
    for x,y,z in tqdm.tqdm(data):
        """ each line is an experiment, each experiment is made of 3 lists
        of coordinates. Each entry in the list refers to an atom """
        mins[0] = min(mins[0], min(x)) 
        mins[1] = min(mins[1], min(y)) 
        mins[2] = min(mins[2], min(z)) 
        maxes[0] = max(maxes[0], max(x)) 
        maxes[1] = max(maxes[1], max(y)) 
        maxes[2] = max(maxes[2], max(z)) 
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)
    all_c = np.array([all_x, all_y, all_z])
    c_mod = np.linalg.norm(all_c, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.hist(all_x, bins=100, label='x', histtype="step")
    ax.hist(all_y, bins=100, label='y', histtype="step")
    ax.hist(all_z, bins=100, label='z', histtype="step")
    ax.set_title('Coordinates')
    ax.set_ylabel('frequency')
    plt.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title('Modulo')
    ax2.hist(c_mod, bins=100, histtype='step')
    plt.show()

def plot_3d_triplet_hist(hist_bundle):
    hist = hist_bundle['data']
    min_val = -hist_bundle['max_mod']
    max_val = hist_bundle['max_mod']
    bins_per_dim = hist_bundle['bins_per_dim']
    sum_min = min_val * 3
    sum_max = max_val * 3
    
    # the center of each bin, we could limit to non-zero bins
    bin_centers = np.linspace(sum_min, sum_max, bins_per_dim)
    
    # Create 3D grids of coordinates
    X, Y, Z = np.meshgrid(bin_centers, bin_centers, bin_centers, 
                          indexing='ij') # meshgrid has two indexing modes
    
    # Mask the data: we only want to plot bins where hist > 0
    mask = hist > 0
    x_coords = X[mask]
    y_coords = Y[mask]
    z_coords = Z[mask]
    counts = hist[mask]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use color and size to represent the intensity of the bin
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                         c=counts, 
                         cmap='viridis', 
                         s=np.log1p(counts) * 10,
                         alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Distribution of Triplet Sums')
    fig.colorbar(scatter, label='Counts')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='data file', required=True)
    parser.add_argument('-r', help='limit the number of experiments', 
                        default=0, type=int)
    parser.add_argument('-p', help='limit the number of parallel numba threads', 
                        default=0, type=int)
    
    parser.add_argument('--command', help='what to do', choices=['summary',
                                                        'all_triplets',
                                                        'all_triplets_numba'])
    parser.add_argument('-d', help='dump results in a pickle file', default='')
    parser.add_argument('-b', help='number of bins per dimension', 
                        type=int, default=100)

    parser.add_argument('-s', help='show the 3D histogram', 
                        default=False, action='store_true')
    return parser.parse_args()
    
def main():
    args = parse_args()
    with open(args.f) as f:
        data = json.load(f)['data']
        """ each line is an experiment, each experiment is made of 3 lists
        of coordinates. Each entry in the list refers to an atom """
    if args.r:
        data  = data[:args.r]
    if args.command == 'summary':
        data_summary(data)
    elif args.command == 'all_triplets':
        hist = compute_triplets.compute_triplets(data, bins_per_dim=args.b)
    elif args.command == 'all_triplets_numba':
        hist = compute_triplets.compute_triplets_numba(data, 
                                                       args.b,
                                                       args.p)
    else:
        print('unknown command')
    
    if args.d:
        with open(args.d, 'bw') as f:
            pickle.dump(hist, f)
    if args.s:
        plot_3d_triplet_hist(hist)


if __name__ == '__main__':
    main()
