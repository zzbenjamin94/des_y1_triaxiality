
# coding: utf-8

# ## Extract shape parameters from halo_shape.npy

# In[1]:

import numpy as np
import sys
sys.path.append('/home/zzbenjamin94/Desktop/Astronomy/Research/DES_Galaxy_Cluster')
from tools.setup.setup import tools_home_dir, home_dir
import pyfits
datadir = home_dir() +'output/buzzard/halo_shape/' 
toolsdir = tools_home_dir()



"""
halo shape files are in the format:

shape_file = np.zeros(halos_num, dtype={'names':('halos_ID', 'richness', 'M200b', 'Rvir', 'redshift',\
            'axes_len', 'axes_dir', 'halos_dir', 'halos_RA', 'halos_DEC', 'converge'),\
            'formats':('i', 'f', 'f','f','f','(3,)f','(3,3)f','(3,)f', 'f', 'f', 'i')}) 
            
            for redMaPPer matched halos. No 'richness' column for randomly selected halos. 
------------------------------------------------------
"""

"""
Bins the halos by lmda in bins determined in this module. 
Upper limit of lower bin must match lower limit of previous bin

Parameter
----------
halo_stat: a parameter of the halos
stat_bins: list of richness bins in format [[low1, high1],[low2, high2], [low3, high3]] 
(e.g. [[20,30],[30,50],[50,300]]), where high value in previous bin 
should match low value in the next (e.g. high2 == low3))

Returns
----------
stat_bins_ind: List of len cosi_num_bin, each element in the list an array of the indices of halos in the bin.
"""

#lmda_bins = [[20,30],[30,50],[50,300]] #upper limit must match lower limit of next bin    
def halo_bin_stat(halo_stat, stat_bins):
    stat_bins_ind = []
    for i, stat_bin in enumerate(stat_bins):
        stat_bin_min = stat_bin[0]; stat_bin_max = stat_bin[1]
        stat_pos = np.where((halo_stat >= stat_bin_min) & (halo_stat < stat_bin_max))[0]
        stat_bins_ind.append(stat_pos)
       
    stat_bins_ind = np.asarray(stat_bins_ind)
        
    return stat_bins_ind

    
"""
Takes filename string, formatted as .npy according to format:
shape_file = np.zeros(halos_num, dtype={'names':('halos_ID', 'richness', 'M200b', 'Rvir', 'redshift',\
            'axes_len', 'axes_dir', 'halos_dir', 'halos_RA', 'halos_DEC', 'converge'),\
            'formats':('i', 'f', 'f','f','f','(3,)f','(3,3)f','(3,)f', 'f', 'f', 'i')}) 
            
            for redMaPPer matched halos. No 'richness' column for randomly selected halos. 
            
and outputs relevant shape parameters such as axis direction and ratio having filtered through
the shape convergence criteria and mass cuts. 
            
Input
---------
halos_shape record array filename string

Output
---------
halos_ID: Buzzard ID of halos
q: minor-major axis ratio
s: intermediate-major axis ratio
cos_i: absolute value of the cosine of major axis with LOS. 
halos_ID: Buzzard ID of halos

#!!TODO: How do you optionally output richness of the halos?
"""

def read_shape_param(halos_shape, convcut=True, verbose = False):    
    #input an structured array
    assert type(halos_shape)==np.ndarray, "Must input a nd.array" 
    
    #Convergence and mass cuts
    if convcut:
        conv_cut = np.where(halos_shape['converge']==True)
        halos_shape = halos_shape[conv_cut]
    halos_num = len(halos_shape)
    
    if verbose == True:
        print "Created from {} record array".format(halos_shape)
        print 'Number of halos after convergence cut is ', halos_num
      
    #Relevant quantities to extract and plot
    ########################################
    halos_ID = halos_shape['halos_ID']
    
    #Axis len and dir
    axes_len = halos_shape['axes_len']
    axes_dir = halos_shape['axes_dir']
    halos_dir = halos_shape['halos_dir']
    q = axes_len[:,2]/axes_len[:,0]
    s = axes_len[:,1]/axes_len[:,0]

    #Orientation PDF
    major_dir = axes_dir[:,2,:]

    #absolute value of cosine of axis between major axis and LOS
    #For redMapper selected halos
    cos_i = np.zeros(halos_num) #cos(i) in lingo of Osato 2017
    for i in range(halos_num):
        halos_dir_mag = np.linalg.norm(halos_dir[i])
        major_mag = np.linalg.norm(major_dir[i]);
        cos_i[i] = np.abs(np.dot(major_dir[i],halos_dir[i])/(halos_dir_mag * major_mag))
    
    return halos_ID, q, s, cos_i
    

# ### Testing

# In[48]:

if __name__=="__main__":    
    try:
        filename = 'halo_shape_allz.npy'
        halos_shape_noadapt = np.load(datadir+filename)
    except IOError:
        print "Error: File {0} cannot be opened".format(filename)
    else:
        print "Created from {} record array".format(filename)
        print "Number of halos is {}".format(len(halos_shape_noadapt))

    
    q, s, cos_i = read_shape_param(halos_shape_noadapt)
    print "q is ", q[0:10]
    print "s is ", s[0:10]
    print "cos_i is ", cos_i[0:10]
    
    



