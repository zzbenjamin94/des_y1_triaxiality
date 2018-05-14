
# coding: utf-8

# # Halo Mass Function Templates
# ## Zhuowen Zhang
# ## Created April 26, 2018

# ## Use P(M) from randomly selected halos in Buzzard 

# In[1]:

import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt

from setup.setup import data_home_dir
from setup.setup import code_home_dir
datadir = data_home_dir()
codedir = code_home_dir()
from mpl_toolkits.mplot3d import Axes3D
from setup.setup import data_home_dir
from setup.setup import code_home_dir
import pyfits
from plot_utils import plot_pretty
plot_pretty()

# In[2]:

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


# #### Import halo file and extract information 

# In[60]:

halos_shape = np.load(datadir+'halos_shape_rand.npy')

#check properties of halos with shapes not converged
ind_not_conv = np.where(halos_shape['converge'] == False)
#print 'Redshift of unconverged halos are', halos_shape[ind_not_conv]['redshift']
#print 'Mass of unconverged halos are', halos_shape[ind_not_conv]['Mvir']

#print 'Positions not converge:', np.where(halos_shape['converge'] != 1)
conv_cut = np.where(halos_shape['converge']==True)
halos_shape = halos_shape[conv_cut]
halos_num = len(halos_shape)
#print 'After cut positions not converged:', np.where(halos_shape['converge'] != 1)
#print 'Number of halos after cuts is ', halos_num
global ln_halos_M

#Clean up code here. Make code more portable so that users can call it. 
num_bins = 20
ln_halos_M = np.log(halos_shape['M200b'])
ln_M_max = np.max(ln_halos_M)
ln_M_min = 13*np.log(10) #corresponding to cutoff at M > 1e13 M_sun
lnM_bin_edge = np.linspace(ln_M_min, ln_M_max, num_bins+1).tolist()
lnM_density = np.histogram(ln_halos_M, lnM_bin_edge, density=True)[0]


# #### Python routine for mass function
# Finds the probability density.
# Input: log10 mass of halo
# Output: probability density

# In[61]:

'''
Finds from the random halos in Buzzard simulations (with z cutoff of z < 0.34) the 
approximate halo mass function. Outputs the density in each mass range, with the density uniform
in a mass range

Input:
lnM: Array of ln of halo masses 
num_bins: numbers of bins for the halo mass function. 10-20 for 7500 random halos appropriate with Sturgess' Law

Output:
lnM_density: Probability density for halo mass
'''

def P_lnM_Buzzard(lnM, num_bins=20):
    lnM = np.array(lnM) #convert floats into array
    if len(np.shape(lnM))==0:
        lnM = lnM[np.newaxis]    
        
    #ln_M_max = np.max(ln_halos_M)
    #ln_M_min = 13*np.log(10) #corresponding to cutoff at M > 1e13 M_sun
    #lnM_bin_edge = np.linspace(ln_M_min, ln_M_max, num_bins+1).tolist()

    #Find probability in each mass bin
    #lnM_density = np.histogram(ln_halos_M, lnM_bin_edge, density=True)[0]
    
    #Find the mass bin and the corresponding density.
    #For mass below the mass cutoff select smallest mass bin, 
    #for mass exceeding highest mass bin select highest mass bin
    lnM_bin_num = np.array(np.searchsorted(lnM_bin_edge, lnM) - 1)
    #print 'lnM_bin_num is ', lnM_bin_num
    #print 'np.where(lnM_bin_num == -1)', np.where(lnM_bin_num == -1)
    lnM_bin_num[np.where(lnM_bin_num == -1)[0]] += 1 #for mass below smallest
    lnM_bin_num[np.where(lnM_bin_num == num_bins)[0]] -= 1 #for mass above highest 

    #print "lnM_density", lnM_density
    #print "lnM_bin_edge", lnM_bin_edge
    return lnM_density[lnM_bin_num]


# ## Testing the code
# 

# In[63]:

if __name__=="__main__":
#plot the halo mass function
    lnM_fake = np.linspace(13*np.log(10),15.5*np.log(10),100) 
    P_lnM_fake = P_lnM_Buzzard(lnM_fake,100)
    plt.plot(lnM_fake, P_lnM_fake)
    plt.xlabel(r'$\ln\Big(M_{200b}~(M_\odot)\Big)$')
    plt.ylabel(r'$\frac{dn}{d(\ln{M})}$')
    plt.show()


# In[ ]:



