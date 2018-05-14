
# coding: utf-8

# # Halo Mass Function Templates
# ## Zhuowen Zhang
# ## Created April 26, 2018

# ## Use P(M) from randomly selected halos in Buzzard 

# In[1]:

import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt
from code.plot_utils import plot_pretty
plot_pretty()
#get_ipython().magic(u'matplotlib inline')

from mpl_toolkits.mplot3d import Axes3D
from code.halo_shape_calc import quad_moment
from code.lightcone_query_ra_dec import query_file, read_radial_bin
from code.setup.setup import data_home_dir
import pyfits
datadir = data_home_dir()


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

# In[3]:

halos_shape = np.load(datadir+'halos_shape_rand.npy')

#check properties of halos with shapes not converged
ind_not_conv = np.where(halos_shape['converge'] == False)
#print 'Redshift of unconverged halos are', halos_shape[ind_not_conv]['redshift']
#print 'Mass of unconverged halos are', halos_shape[ind_not_conv]['Mvir']

print 'Positions not converge:', np.where(halos_shape['converge'] != 1)
conv_cut = np.where(halos_shape['converge']==True)
halos_shape = halos_shape[conv_cut]
halos_num = len(halos_shape)
print 'After cut positions not converged:', np.where(halos_shape['converge'] != 1)
print 'Number of halos after cuts is ', halos_num
global ln_halos_M
ln_halos_M = np.log10(halos_shape['Mvir'])


# #### Python routine for mass function
# Finds the probability density.
# Input: log10 mass of halo
# Output: probability density

# In[40]:

'''
Finds from the random halos in Buzzard simulations (with z cutoff of z < 0.34) the 
approximate halo mass function. Outputs the density in each mass range, with the density uniform
in a mass range
'''

def P_lnM_Buzzard(logM, num_bins):
    ln_M_max = np.max(ln_halos_M)
    ln_M_min = 13 #corresponding to cutoff at M > 1e13 M_sun
    lnM_bin_edge = np.linspace(ln_M_min, ln_M_max, num_bins+1).tolist()

    #Find probability in each mass bin
    lnM_density = np.histogram(ln_halos_M, lnM_bin_edge, density=True)[0]
    
    #Find the mass bin and the corresponding density.
    #For mass below the mass cutoff select smallest mass bin, 
    #for mass exceeding highest mass bin select highest mass bin
    lnM_bin_num = np.searchsorted(lnM_bin_edge, logM) - 1
    lnM_bin_num[np.where(lnM_bin_num == -1)] += 1 #for mass below smallest
    lnM_bin_num[np.where(lnM_bin_num == num_bins)] -= 1 #for mass above highest 

    #print "lnM_density", lnM_density
    #print "lnM_bin_edge", lnM_bin_edge
    return lnM_density[lnM_bin_num]


# ## Testing the code
# 

# In[54]:

if __name__=="__main__":
#plot the halo mass function
    lnM_fake = np.linspace(13,15.5,100) 
    P_lnM_fake = P_lnM_Buzzard(lnM_fake,100)
    plt.plot(lnM_fake, P_lnM_fake)
    plt.xlabel(r'$\log_{10}\Big(M_{200b}~(M_\odot)\Big)$')
    plt.ylabel(r'$\frac{dn}{d(\log{M})}$')
    plt.show()


# In[ ]:



