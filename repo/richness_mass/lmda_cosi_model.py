
# coding: utf-8

# # PYMC models for richness-mass relation
# ### Zhuowen Zhang
# ### Created Aug 27, 2018
# ### Nov. 20, 2018: For hrun, with lambda cut > 5, need to renormalize. Adjusted for hrun of lmda>5. Make cut flexible in future. 

# In[4]:

import numpy as np
import sys
# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt
sys.path.append('/home/zzbenjamin94/Desktop/Astronomy/Research/DES_Galaxy_Cluster')
from tools.setup.setup import tools_home_dir, home_dir
homedir = home_dir()
toolsdir = tools_home_dir()
shapedir = home_dir()+'output/buzzard/halo_shape/'
#tpltdir = home_dir() + 'output/lmda_cosi_chains/'

from tools.plot_utils import plot_pretty
from tools.halo_mass_template import redMaPPer_hmf, hrun_hmf
plot_pretty()
get_ipython().magic(u'matplotlib inline')

from mpl_toolkits.mplot3d import Axes3D
import pyfits

import astropy.io.fits as pyfits
import ConfigParser
import healpy as hp
import treecorr
import os

from pymc import *
from pymc import DiscreteUniform, Normal, uniform_like, TruncatedNormal
from pymc import Metropolis
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.special import erf
from chainconsumer import ChainConsumer

#Define mass range and hmf for tabulated hmf
lnM_min = 13.5*np.log(10); lnM_max = 15.5*np.log(10)
lnM = np.linspace(lnM_min, lnM_max, 1000)
delta_lnM = lnM[1]-lnM[0]
#Halo mass function
#lnM_density, lnM_bin_cen = redMaPPer_hmf(lnM, lnM_min=lnM_min, lnM_max = lnM_max)
lnM_density, lnM_bin_cen = hrun_hmf(lnM, lnM_min=lnM_min, lnM_max = lnM_max, num_bins = 200)



# ## Models for richness-mass from the pymc package
# 

# Assume a linear relation in richness-mass in the log-log plane, and with a log-normal scatter. Free parameters are A the intercept, B the slope, and sig0 the instrinsic scatter.
# 
# The truncated Gaussian at $\lambda = 20$ is normalized with the mass function from the random halo catalog in the Buzzard simulations. 
# 
# The linear relation between richness and mass is
# $$
# \mu_{\log(\lambda)} = \log{A} + B*(\log{M} - \log{10^{14}})
# $$
# 
# with instrinsic scatter
# $$
# \sigma^2(\log{\lambda}) = \sigma_0^2 + \frac{\exp{\mu}-1}{2\exp{\mu}}.
# $$
# 
# The posterior probability is 
# $$
# P(A,B,\sigma|\lambda, M) = \frac{1}{N(A,B,\sigma, M)}P(\lambda, M | A, B, \sigma) P (A, B, \sigma),
# $$
# 
# with normalization factor
# $$
# N(A, B, \sigma, M) = \int dM \int_{20}^{+\infty} d \lambda P(\lambda| M, A, B, \sigma) P(M)
# = \int dM(0.5 - 0.5 \mathrm{erf}\big(\frac{\mathrm{ln} 20 - \mu{(\mathrm{ln} \lambda)}}{\sqrt{2(\sigma_0^2+ \frac{\mathrm{exp} ( \mu{(\mathrm{ln} \lambda)})-1}{\mathrm{exp} ( 2\mu{(\mathrm{ln} \lambda)})})}} \big) P(M).
# $$
# 
# $P(M)$ is sampled from the randomly selected halo (see code tools.halo_mass_template) and integration is done by tabulating P(M) into 100 bins. 
# 
# 

# ### Model 1
# Find global A, B, and sigma0. This essentially becomes a three parameter model if you input the data iteratively from different bins. It then finds the A, B, sigma0 for each bin. 

# In[1]:

def make_model(lnls, lnms):
    A=Uniform('A', lower=0.1, upper=100 )
    B=Uniform('B', lower=0.0001, upper=10 )
    sig0=Uniform('sigma0', lower=0.001, upper=10.0 )   
    
    #Use table to approximate integration. Speed things up
    def norm_5_tab(A,B,sig0):         
        mu_lnl=np.log(A)+B*(lnM-14.0*np.log(10.0)) 
        var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))

        norm = np.sum((0.5-0.5*erf((np.log(5) - mu_lnl)/np.sqrt(2.0*var_lnl)))*lnM_density*delta_lnM)
        return norm

    @pymc.stochastic(observed=True, plot=False)
    def log_prob(value=0, A=A, B=B, sig0=sig0):       
        mu_lnl=np.log(A)+B*(lnms-14.0*np.log(10.0)) 
        var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))

        norm = norm_5_tab(A,B,sig0)
        log_prs=-0.5*(lnls-mu_lnl)**2/var_lnl -0.5*np.log(var_lnl) - np.log(norm) 

        tot_logprob=np.sum(log_prs)
        return tot_logprob
    return locals()


# ### Model 2
# This model varies A across different cosine bins and finds a global fit for B, Sigma0

# In[2]:

def make_model2(lnls, lnms):
    #Extract number of bins
    assert len(lnls)==len(lnms), "Number of bins different for lnls and lnms"
    num_bins = len(lnls)
    
    #Vary A across bins but keep B, Sigma the same across bins
    A = [None]*num_bins
    A_0 =Uniform('A_{}'.format(0), lower=0.1, upper=100)
    A_1 =Uniform('A_{}'.format(1), lower=0.1, upper=100)
    A_2 =Uniform('A_{}'.format(2), lower=0.1, upper=100)
    A_3 =Uniform('A_{}'.format(3), lower=0.1, upper=100)
    A_4 =Uniform('A_{}'.format(4), lower=0.1, upper=100)
    A[0]=A_0; A[1]=A_1; A[2]=A_2; A[3]=A_3; A[4]=A_4
    
    #The any() function overwritten by PYMC. Does not work
    assert None not in A, "Number of A_i do not match size of A" #Equiv to not any(x is None for x in A)
    
    B=Uniform('B', lower=0.0001, upper=10 )
    sig0=Uniform('sigma0', lower=0.001, upper=10.0 )   
    
    #Use table to approximate integration. Speed things up
    def norm_5_tab(A_i,B,sig0):       
        mu_lnl=np.log(A_i)+B*(lnM-14.0*np.log(10.0)) 
        var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))
        norm = np.sum((0.5-0.5*erf((np.log(5) - mu_lnl)/np.sqrt(2.0*var_lnl)))*lnM_density*delta_lnM)
        return norm

    @pymc.stochastic(observed=True, plot=False)
    def log_prob(value=0, A=A, B=B, sig0=sig0):      
        tot_logprob = 0
        for i in range(num_bins):
            mu_lnl=np.log(A[i])+B*(lnms[i]-14.0*np.log(10.0)) #changed May 1, 2018
            var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))
            norm = norm_5_tab(A[i],B,sig0)
            log_prs=-0.5*(lnls[i]-mu_lnl)**2/var_lnl -0.5*np.log(var_lnl) - np.log(norm) 
            tot_logprob += np.sum(log_prs)
            
        return tot_logprob
    
    return A_0, A_1, A_2, A_3, A_4, B, sig0


# ### Model 3
# This models varies A only, and inputs the best fit B and sig0 found from the maximum posterior from make_model2
# 

# In[3]:

def make_model3(lnls, lnms, B, sig0):
    A=Uniform('A', lower=0.1, upper=100 )
   
    #Use table to approximate integration. Speed things up
    def norm_5_tab(A):       
        mu_lnl=np.log(A)+B*(lnM-14.0*np.log(10.0)) 
        var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))
        norm = np.sum((0.5-0.5*erf((np.log(5) - mu_lnl)/np.sqrt(2.0*var_lnl)))*lnM_density*delta_lnM)
        return norm

    @pymc.stochastic(observed=True, plot=False)
    def log_prob(value=0, A=A, B=B, sig0=sig0):
        mu_lnl=np.log(A)+B*(lnms-14.0*np.log(10.0)) 
        var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))

        norm = norm_5_tab(A)
        log_prs=-0.5*(lnls-mu_lnl)**2/var_lnl -0.5*np.log(var_lnl) - np.log(norm) 

        tot_logprob=np.sum(log_prs)
        return tot_logprob
    return locals()



