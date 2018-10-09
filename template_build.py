
# coding: utf-8

# # Template Fitting
# ## Zhuowen Zhang
# ### First Created April 16, 2018

# In[4]:

import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

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
'''
#Import halo files
halos_shape = np.load(datadir+'halos_shape.npy')
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
'''

# In[ ]:

import numpy as np
from pymc import *
from pymc import DiscreteUniform, Normal, uniform_like, TruncatedNormal
from pymc import Metropolis
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.special import erf
from chainconsumer import ChainConsumer

def make_model(lnls, lnms, lnms_all):
    A=Uniform('A', lower=0.1, upper=100 )
    B=Uniform('B', lower=0.0001, upper=10 )
    sig0=Uniform('sigma0', lower=0.001, upper=10.0 )

    @pymc.stochastic(observed=True, plot=False)
    def log_prob(value=0, A=A, B=B, sig0=sig0):
        mu_lnl=np.log(A)+B*(lnms-14.0*np.log(10.0))
        var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))

        mu_lnl_all=np.log(A)+B*(lnms_all-14.0*np.log(10.0))
        var_lnl_all=sig0**2+(np.exp(mu_lnl_all)-1)/(np.exp(2*mu_lnl_all))
        norm_20=np.sum(0.5-0.5*erf( (np.log(20) - mu_lnl_all)/np.sqrt(2.0*var_lnl_all) ))

        log_prs=-0.5*(lnls-mu_lnl)**2/var_lnl -0.5*np.log(var_lnl) - np.log(norm_20) 

        tot_logprob=np.sum(log_prs)
        return tot_logprob
    return locals()

if __name__ == "__main__":
 
    # generate some fake data 
    lnM=randn(1000)
    lnM=lnM[np.where(lnM > 0)]
    lnM=lnM*3+30
    A=10;B=1;sig0=0.5
    mu_lnl=np.log(A)+B*(lnM-14.0*np.log(10.0))
    var_lnl=sig0**2+(np.exp(mu_lnl)-1)/(np.exp(2*mu_lnl))
    lnl=randn(len(lnM))*np.sqrt(var_lnl)+mu_lnl
    # apply a lambda cut
    ind=np.where(lnl >  np.log(20))
    lnl_cut=lnl[ind]
    lnM_cut=lnM[ind]
         
    mcmc_file='mcmc_output_test1'
    M=pymc.Model(make_model(lnl_cut, lnM_cut, lnM))
    mc=MCMC(M, db='txt', dbname=mcmc_file)
    num=200
    n_iter=num*5000
    n_burn=num*3000
    n_thin=num
    mc.sample(iter=n_iter, burn=n_burn, thin=n_thin)


# In[33]:

from code.plot_utils import plot_2d_dist
if __name__ == "__main__":   
    As=np.genfromtxt(mcmc_file+'/Chain_0/A.txt')
    Bs=np.genfromtxt(mcmc_file+'/Chain_0/B.txt')
    sig0s=np.genfromtxt(mcmc_file+'/Chain_0/sigma0.txt')
    print np.mean(As), np.std(As)
    print np.mean(Bs), np.std(Bs)
    print np.mean(sig0s), np.std(sig0s)

    # plot the parameter constraints
    c = ChainConsumer()
    data=np.vstack( (As, Bs, sig0s) ).T
    c.add_chain(data, parameters=[r"$A$", r"$B$", r"$\sigma_0$"], name='fake data')
    c.configure(colors=['k'], linestyles=[":"], shade=[True], shade_alpha=[0.5])
    c.plotter.plot(filename="test1.png", figsize=(5,5), truth=[A, B, sig0])
    plt.show()
    
    #If c.plotter.plot doesn't work manually use Andrey's plot_2d_hist to get your plots. 
    #plot_2d_dist(As,Bs,xlim=[5,15],ylim=[0.8,1.2], clevs=[0.68,0.93,0.99],\
    #             nxbins=50,nybins=50, cmin=1.e-4, cmax=1.0, xlabel='x',ylabel='y')

    # make another plot to show the fake data and the constraints
    plt.loglog(np.exp(lnM_cut), np.exp(lnl_cut), 'k.');plt.xlabel(r'$M$');plt.ylabel(r'$\lambda$')
    lnM=np.arange(31, 40, 0.1)
    mu_lnl_arr=np.zeros([len(lnM), len(As)])
    sig_lnl_arr=np.zeros([len(lnM), len(As)])
    for ii in range(len(As)):
        A=As[ii];B=Bs[ii];sig0=sig0s[ii]
        mu_lnl_arr[:, ii]=np.log(A)+B*(lnM-14.0*np.log(10.0))
        sig_lnl_arr[:, ii]=np.sqrt(sig0**2+(np.exp(mu_lnl_arr[:, ii])-1)/(np.exp(2*mu_lnl_arr[:, ii])))
    mu_lnl_mea=np.zeros([len(lnM)])
    sig_lnl_mea=np.zeros([len(lnM)])
    mu_lnl_std=np.zeros([len(lnM)])
    sig_lnl_std=np.zeros([len(lnM)])
    for jj in range(len(lnM)):
        mu_lnl_mea[jj]=np.mean(mu_lnl_arr[jj, :])
        mu_lnl_std[jj]=np.std(mu_lnl_arr[jj, :])
        sig_lnl_mea[jj]=np.mean(sig_lnl_arr[jj, :])
        sig_lnl_std[jj]=np.std(sig_lnl_arr[jj, :])
    plt.plot(np.exp(lnM), np.exp(mu_lnl_mea), 'r')     
    plt.fill_between(np.exp(lnM), np.exp(mu_lnl_mea-mu_lnl_std), np.exp(mu_lnl_mea+mu_lnl_std), facecolor='r', alpha=0.1)     
    plt.plot(np.exp(lnM), np.exp(mu_lnl_mea-sig_lnl_mea), 'r--')     
    plt.fill_between(np.exp(lnM), np.exp(mu_lnl_mea-sig_lnl_mea+sig_lnl_std), np.exp(mu_lnl_mea-sig_lnl_mea-sig_lnl_std), facecolor='r', alpha=0.1)     
    plt.plot(np.exp(lnM), np.exp(mu_lnl_mea+sig_lnl_mea), 'r--')    
    plt.fill_between(np.exp(lnM), np.exp(mu_lnl_mea+sig_lnl_mea+sig_lnl_std), np.exp(mu_lnl_mea+sig_lnl_mea-sig_lnl_std), facecolor='r', alpha=0.1)     
    plt.show() 


# In[ ]:



