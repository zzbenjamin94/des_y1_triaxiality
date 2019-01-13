
# coding: utf-8

# ## Backend process of statistical processes for halo shapes
# ### Zhuowen Zhang, created Aug. 15, 2018

# In[7]:

# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt

from glob import glob
import numpy as np
import healpy as hp
import struct

import sys
sys.path.append('/home/zzbenjamin94/Desktop/Astronomy/Research/DES_Galaxy_Cluster')
from tools.setup.setup import home_dir
from tools.kmeans_radec import kmeans, kmeans_sample
import pyfits
datadir = home_dir()+'output/buzzard/halo_shape/'

from scipy.stats import t

            
    
'''
Uses the kmeans method to split the sky into different patches and run jackknife resampling on the 
user-inputted statistic to find the mean and standard error of said statistic. 

Parameters
-----------
ncen: Initial guess of halo centers to split the sky into. Determines number of centers.
halos_coord: Array of [N,2] in RA, DEC
stats: Array of [num_stats]
bins_ind: List of arrays that contains indices of "stats" parameters in each bin. Output of halo_bin_stat(). 

Output:
----------
kmeans_bin_mean: [num_bin] array of mean in each bin
kmeans_bin_SE: [num_bin] array of standard error in each bin
'''
def kmeans_stats(ncen, halos_coord, stat, bins_ind):
    #Sanity checks
    assert all([len(halos_coord), len(stat)]), "Number of halos not equal number of statistics."
    
    #output arrays
    num_bins = len(bins_ind)
    stat_bin_SE = np.zeros(num_bins)
    stat_bin_mean = np.zeros(num_bins)
    
    #kmeans centers and coordinates 
    km = kmeans_sample(halos_coord, ncen, verbose=0)
    km_centers = km.get_centers()
    km_labels= km.find_nearest(halos_coord)
    km_ncen = km.get_ncen()
    

    for i in range(num_bins):
        stat_jk_list= []
        for j in range(km_ncen):
            jackknife_cut = np.where(km_labels != j)[0]
            jackknife_cut = np.intersect1d(jackknife_cut, bins_ind[i])
            stat_bin_cut = stat[jackknife_cut]            
            stat_jk_list.append(stat_bin_cut)
            
        stat_jk_list = np.asarray(stat_jk_list)
    
        #Jackknife estimator of SE and mean
        stat_rdu_arr = np.array([np.mean(stat_jk_list[x]) for x in range(len(stat_jk_list))])
        stat_jk_mean = np.mean(stat_rdu_arr)
        stat_jk_var = np.var(stat_rdu_arr, ddof=1)
        stat_jk_std = np.std(stat_rdu_arr, ddof=1)
        stat_jk_SE = (km_ncen-1)/np.sqrt(km_ncen)*stat_jk_std
        stat_bin_mean[i] = stat_jk_mean; stat_bin_SE[i] = stat_jk_SE
        
    return stat_bin_SE, stat_bin_mean


'''
Compares the mean deviation between two distributinos that follow a student-t distribution. The two distributions 
should have an equal number of samples. 

Input
------------
n_samp: Number of samples for both distributions to compare
SE_x: standard error of the first distribution
mean_x: mean of first distribution
SE_y: standard error of the second distribution
mean_y: mean of second distribution

Returns
------------
sigma_diff: Sigma level of the difference between the two distributions
'''
def student_t_test(n_samp, SE_x, mean_x, SE_y, mean_y):
    SE_pool = np.sqrt(SE_x**2. + SE_y**2.)
    dof_pool = SE_pool**2./(SE_x**2./(n_samp*(n_samp-1.)) + SE_y**2./(n_samp*(n_samp-1.)))
    sigma_diff = (mean_x - mean_y)/SE_pool
    return sigma_diff



