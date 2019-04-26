
# coding: utf-8

# # Template Fitting
# ## Zhuowen Zhang
# ### First Created April 16, 2018

# In[4]:
import numpy as np
import sys
sys.path.append('/home/zzbenjamin94/Desktop/Astronomy/Research/DES_Galaxy_Cluster')

# import pyplot and set some parameters to make plots prettier
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from tools.plot_utils import plot_pretty
plot_pretty()

from mpl_toolkits.mplot3d import Axes3D
from tools.setup.setup import home_dir

import astropy.io.fits as pyfits
import ConfigParser
import healpy as hp
import treecorr
import os

datadir = home_dir()+'output/lmda_cosi_chains/miscentering/'

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
cosi_bins = [[0.0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]] 
#upper limit must match lower limit of next bin

mcmc_filestr = range(len(cosi_bins)) #refer to the 5 cosine bins
cosi_bins_ind = []
'''
for i, cosi_bin in enumerate(cosi_bins):
    cosi_bin_min = cosi_bin[0]; cosi_bin_max = cosi_bin[1]
    cosi_pos = np.where((cos_i >= cosi_bin_min) & (cos_i < cosi_bin_max))
    cosi_bins_ind.append(cosi_pos)
'''    

#Contour plot for the models. Model1
for i in mcmc_filestr:
	cosi_bin_min = cosi_bins[i][0]; cosi_bin_max = cosi_bins[i][1]
	mcmc_folder = datadir + 'p_lmda_cosi_{}'.format(i)+'_model1'
	As=np.genfromtxt(mcmc_folder+'/Chain_0/A.txt'.format(i))
	Bs=np.genfromtxt(mcmc_folder+'/Chain_0/B.txt')
	sig0s=np.genfromtxt(mcmc_folder+'/Chain_0/sigma0.txt')
	print "mean and stdev for A{}".format(i), np.mean(As), sem(As)
	print "mean and stdev for B{}".format(i), np.mean(Bs), sem(Bs)
	print "mean and stdev for sig0{}".format(i), np.mean(sig0s), sem(sig0s)

	# plot the parameter constraints
	c = ChainConsumer()
	data=np.vstack( (As, Bs, sig0s) ).T
	#data = As
	c.add_chain(data, parameters=[r"$A_{}$".format(i), r"$B$", r"$\sigma_0$"], \
			name=r'Template for $cos(i)\in[%.1f, %.1f)$'%(cosi_bin_min, cosi_bin_max))
	c.configure(statistics="max_central",rainbow=True, linestyles=[":"], shade=[True], shade_alpha=[0.5])
	c.plotter.plot(display=True, figsize="column")
plt.show()


'''  
#For all/fake data
#mcmc_folder = datadir + 'p_lmda_fake'
mcmc_folder = datadir + 'p_lmda_cosi_all_model2'

num_files = 5
As = [None] * num_files
for i in range(num_files):       
	As[i]=np.genfromtxt(mcmc_folder+'/Chain_0/A_{}.txt'.format(i))
	print np.mean(As[i]), np.std(As[i])

Bs=np.genfromtxt(mcmc_folder+'/Chain_0/B.txt')
sig0s=np.genfromtxt(mcmc_folder+'/Chain_0/sigma0.txt')
#print np.mean(As), np.std(As)
print np.mean(Bs), np.std(Bs)
print np.mean(sig0s), np.std(sig0s)


# plot the parameter constraints
for i in range(num_files):
	c = ChainConsumer()
	data=np.vstack( (As[i], Bs, sig0s) ).T
	c.add_chain(data, parameters=[r"$A_{}$".format(i), r"$B$", r"$\sigma_0$"], \
            name=r'Posterior for richness-mass fit')
	c.configure(statistics="max_central", rainbow=False, linestyles=[":"], shade=[True], \
	shade_alpha=[1], usetex=True, contour_labels='sigma', summary=False)    
	c.plotter.plot(display=True, figsize="column")
#c.plotter.plot(filename="p_lmda_fake_chains.png", figsize="column", truth=[10,1,0.5])
#plt.show()
'''
 
#If c.plotter.plot doesn't work manually use Andrey's plot_2d_hist to get your plots. 
#plot_2d_dist(As,Bs,xlim=[5,15],ylim=[0.8,1.2], clevs=[0.68,0.93,0.99],\
#             nxbins=50,nybins=50, cmin=1.e-4, cmax=1.0, xlabel='x',ylabel='y')


