
# coding: utf-8

# # Halo Mass Function Templates
# ## Zhuowen Zhang
# ## Created April 26, 2018

# In[1]:

import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt

from setup.setup import data_home_dir, home_dir, bigdata_home_dir
homedir = home_dir()
from repo.halo_shape.read_shape_param import read_shape_param
from repo.halo_shape.read_shape_param import halo_bin_stat
from mpl_toolkits.mplot3d import Axes3D
import astropy.io.fits as fits
from plot_utils import plot_pretty
plot_pretty()
import pickle

buzzdir = home_dir() + 'output/buzzard/'
halorun_dir = bigdata_home_dir() + 'Buzzard_Sims/v1.6/halo_run/'

#Basic units, MKS
ang2rad = np.pi/180
kpc2m = 3.086e19
eV2j = 1.602e-19
yr2sec = 3.154e7
c = 2.99e8 #m/s
h  = 6.626e-34 #in eV
e = 4.8e-10 #Coulumbs
G = 6.674*10**(-11) #MKS
M_sun = 1.99e30 #kg, mass of sun


#### Import halo file and extract information 
#Random catalog
ctlg_names= ['halos_ID','halos_RA', 'halos_DEC', 'halos_z', 'halos_X', 'halos_Y', 'halos_Z', 'halos_M', 'halos_Rvir']
ctlg_format = ['d', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g']

rand_ctlg = np.loadtxt(homedir+'output/buzzard/halo_rand_allz.dat', \
        dtype={'names': ctlg_names, 'formats': ctlg_format}, skiprows=1) 
rand_lnM = np.log(rand_ctlg['halos_M'])


###Change the catalog as needed. 
filename = homedir+'output/buzzard/hrun_matched_m5e13cut_allz.npy'
redM_halos = np.load(filename)
#halos_ID_shapecut, q, s, cos_i =  read_shape_param(halo_shape, convcut=True)
#shapecut = [ind for (ind, val) in enumerate(halo_shape['halos_ID']) if val in halos_ID_shapecut]
#halo_shape = halo_shape[shapecut]
halos_M = redM_halos['M200B']
richness = redM_halos['LAMBDA_CHISQ']

lnM_catalog = np.log(halos_M)
#Bin by richness
lmda_max = np.max(richness)
lmda_bins = [[20,30],[30,50],[50,lmda_max]] #upper limit must match lower limit of next bin
num_lmda_bins = len(lmda_bins)
lmda_bins_ind = halo_bin_stat(richness, lmda_bins)
lnM_lmda_binned = np.zeros_like(lmda_bins_ind)
lnM_lmda_binned = np.array([np.log(halos_M[lmda_bins_ind[x]]) for x in range(len(lmda_bins_ind))])


'''
From user inputted halo masses generates probability of input halo mass from a tabulated halo mass function. Masses below or above the minimum/max mass range will be given a density equal to the min/max bin.

Code useful for generated tabulated pdf (or in this case the halo mass function) that can be called by a MCMC code. 

Input
----------------
lnM: Array of ln of halo masses of the catalog
lnM_min: minimum of log mass 
lnM_max: maximum of log mass
num_bins: numbers of bins for the halo mass function. 10-20 for 7500 random halos appropriate with Sturgess' Law

Return
----------------
lnM_density: probabilitistic density of finding each halo in lnM in each mass bin. 
lnM_bin_cen: Array of size num_bin containing the centers of each bin
'''
def input_hmf(lnM, lnM_min = 13*np.log(10), lnM_max=15*np.log(10), num_bins=20):
    lnM = np.array(lnM) #convert floats into array
    if len(np.shape(lnM))==0:
        lnM = lnM[np.newaxis]    
        
    #lnM_bin_cen = np.linspace(lnM_min, lnM_max, num_bins)
    #lnM_bin_cen += (lnM_bin_cen[1] - lnM_bin_cen[0])
    
    lnM_bin_edge = np.linspace(lnM_min, lnM_max, num_bins+1)   #.tolist()
    lnM_bin_cen = lnM_bin_edge[:-1] + (lnM_bin_edge[1] - lnM_bin_edge[0])/2.
    
    lnM_density = np.histogram(lnM, lnM_bin_edge, density=True)[0]    
    
    lnM_bin_num = np.array(np.searchsorted(lnM_bin_edge, lnM) - 1)
    lnM_bin_num[np.where(lnM_bin_num == -1)[0]] += 1 #for mass below smallest
    lnM_bin_num[np.where(lnM_bin_num == num_bins)[0]] -= 1 #for mass above highest 

    return lnM_density[lnM_bin_num], lnM_bin_cen



'''
From user inputted halo masses generates probability of input halo mass from a tabulated halo mass function. 
The HMF is taken from the halorun halos, with option to bin them by richness.  
Masses below or above the minimum/max mass range will be given a density equal to the min/max bin.
Code useful for generated tabulated pdf (or in this case the halo mass function) that can be called by a MCMC code. 

Input
----------------
lnM: Array of ln of halo masses 
lnM_min: minimum of log mass 
lnM_max: maximum of log mass
num_bins: numbers of bins for the halo mass function. 10-20 for 7500 random halos appropriate with Sturgess' Law
lmda_bin: Bin number in lmda. Lmda bins defined above. 

Return
----------------
lnM_density: density in each bin. Should add up to 1
lnM_bin_cen: Array of size num_bin containing the centers of each bin
'''
def catalog_hmf(lnM, lnM_min = 13*np.log(10), lnM_max=15*np.log(10), num_bins=20, lmda_bin=None):
    global lnM_catalog
    global lnM_lmda_binned
    
    lnM = np.array(lnM) #convert floats into array
    if len(np.shape(lnM))==0:
        lnM = lnM[np.newaxis]    
        
    if lmda_bin == None:
        hmf_lnM = lnM_catalog
    else:
        hmf_lnM = lnM_lmda_binned[lmda_bin]
    
    lnM_bin_edge = np.linspace(lnM_min, lnM_max, num_bins+1)   #.tolist()
    lnM_bin_cen = lnM_bin_edge[:-1] + (lnM_bin_edge[1] - lnM_bin_edge[0])/2.
    lnM_density = np.histogram(hmf_lnM, lnM_bin_edge, density=True)[0]    
    
    lnM_bin_num = np.array(np.searchsorted(lnM_bin_edge, lnM) - 1)
    lnM_bin_num[np.where(lnM_bin_num == -1)[0]] += 1 #for mass below smallest
    lnM_bin_num[np.where(lnM_bin_num == num_bins)[0]] -= 1 #for mass above highest 

    return lnM_density[lnM_bin_num], lnM_bin_cen


# ## Testing the code
# 

# In[63]:

if __name__=="__main__":
    print "Yes"
#plot the halo mass function






