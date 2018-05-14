
# coding: utf-8

# # Template Fitting
# ## Zhuowen Zhang
# ### First Created April 16, 2018

# In[4]:

import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from code.plot_utils import plot_pretty
plot_pretty()

from mpl_toolkits.mplot3d import Axes3D
from code.halo_shape_calc import quad_moment
from code.lightcone_query_ra_dec import query_file, read_radial_bin
from code.setup.setup import data_home_dir
from code.setup.setup import code_home_dir
datadir = data_home_dir()
codedir = code_home_dir()


import astropy.io.fits as pyfits
import ConfigParser
import healpy as hp
import treecorr
import os


# In[6]:

#Import halo files
halos_shape = np.load(datadir+'halos_shape_allz.npy')
#halos_shape = np.zeros(halos_num, dtype={'names':('halos_ID', 'richness', 'Mvir', 'Rvir', 'redshift', 'axes_len', \
#       'axes_dir', 'halos_dir', 'converge'),'formats':('i', 'f', 'f','f','f','(3,)f','(3,3)f','(3,)f','i')})

#Apply convergence cut
#print 'Positions not converge:', np.where(halos_shape['converge'] != 1)
conv_cut = np.where(halos_shape['converge']==True)
halos_shape = halos_shape[conv_cut]
halos_num = len(halos_shape)

#quantities to extract
halos_RA = halos_shape['halos_RA']; halos_DEC = halos_shape['halos_DEC']
halos_coord = np.array([halos_RA, halos_DEC]).T
axes_len = halos_shape['axes_len']
axes_dir = halos_shape['axes_dir']
q = axes_len[:,2]/axes_len[:,0]
s = axes_len[:,1]/axes_len[:,0]
richness = halos_shape['richness']
lmda_max = np.max(richness)
halos_Mvir = halos_shape['M200b']

#Find angle
#Orientation PDF
halos_dir = halos_shape['halos_dir']
axes_dir = halos_shape['axes_dir']
major_dir = axes_dir[:,2,:]

#absolute value of cosine of axis between major axis and LOS
cos_i = np.zeros(halos_num) #cos(i) in lingo of Osato 2017
for i in range(halos_num):
    halos_dir_mag = np.linalg.norm(halos_dir[i])
    major_mag = np.linalg.norm(major_dir[i]);
    cos_i[i] = np.abs(np.dot(major_dir[i],halos_dir[i])/(halos_dir_mag * major_mag))
#convert cosine to angle in degrees
angle_los_halo = np.arccos(cos_i)*180/np.pi


#MCMC Template Building
import numpy as np
from pymc import *
from pymc import DiscreteUniform, Normal, uniform_like, TruncatedNormal
from pymc import Metropolis
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.special import erf
from chainconsumer import ChainConsumer
from scipy.stats import sem

#Template Plotting

from code.plot_utils import plot_2d_dist
cosi_bins = [[0.0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]] #upper limit must match lower limit of next bin
mcmc_filestr = range(len(cosi_bins)) #refer to the 5 cosine bins
cosi_bins_ind = []
for i, cosi_bin in enumerate(cosi_bins):
    cosi_bin_min = cosi_bin[0]; cosi_bin_max = cosi_bin[1]
    cosi_pos = np.where((cos_i >= cosi_bin_min) & (cos_i < cosi_bin_max))
    cosi_bins_ind.append(cosi_pos)
    
for i in mcmc_filestr:
    cosi_bin_min = cosi_bins[i][0]; cosi_bin_max = cosi_bins[i][1]
    mcmc_folder = datadir + 'p_lmda_cosi_allz_'+str(mcmc_filestr[i])
        
    As=np.genfromtxt(mcmc_folder+'/Chain_0/A.txt')
    Bs=np.genfromtxt(mcmc_folder+'/Chain_0/B.txt')
    sig0s=np.genfromtxt(mcmc_folder+'/Chain_0/sigma0.txt')
    print np.mean(As), sem(As)
    print np.mean(Bs), sem(Bs)
    print np.mean(sig0s), sem(sig0s)

    # plot the parameter constraints
    c = ChainConsumer()
    data=np.vstack( (As, Bs, sig0s) ).T
    c.add_chain(data, parameters=[r"$A$", r"$B$", r"$\sigma_0$"], \
                name=r'Template for $cos(i)\in[%.1f, %.1f)$'%(cosi_bin_min, cosi_bin_max))
    c.configure(statistics="max_central",rainbow=True, linestyles=[":"], shade=[True], shade_alpha=[0.5])
    c.plotter.plot(display=True, figsize="column")
    plt.show()


   
#For all/fake data
#mcmc_folder = datadir + 'p_lmda_fake'
mcmc_folder = datadir + 'p_lmda_cosi_'+'allcosi_allz'
        
As=np.genfromtxt(mcmc_folder+'/Chain_0/A.txt')
Bs=np.genfromtxt(mcmc_folder+'/Chain_0/B.txt')
sig0s=np.genfromtxt(mcmc_folder+'/Chain_0/sigma0.txt')
print np.mean(As), np.std(As)
print np.mean(Bs), np.std(Bs)
print np.mean(sig0s), np.std(sig0s)

# plot the parameter constraints
c = ChainConsumer()
data=np.vstack( (As, Bs, sig0s) ).T
c.add_chain(data, parameters=[r"$A$", r"$B$", r"$\sigma_0$"], \
            name=r'Template for all data')
c.configure(statistics="max_central", rainbow=True, linestyles=[":"], shade=[True], shade_alpha=[0.5])    
c.plotter.plot(display=True, figsize="column", truth=[10,1,0.5])
#c.plotter.plot(filename="p_lmda_fake_chains.png", figsize="column", truth=[10,1,0.5])
#plt.show()

 
#If c.plotter.plot doesn't work manually use Andrey's plot_2d_hist to get your plots. 
#plot_2d_dist(As,Bs,xlim=[5,15],ylim=[0.8,1.2], clevs=[0.68,0.93,0.99],\
#             nxbins=50,nybins=50, cmin=1.e-4, cmax=1.0, xlabel='x',ylabel='y')


