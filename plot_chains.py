
# coding: utf-8

# # Template Fitting
# ## Zhuowen Zhang
# ### First Created April 16, 2018

# In[4]:

import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
from chainconsumer import ChainConsumer


if __name__ == "__main__":   
    mcmc_dir='chains/'+'final_chains/'
    As=np.genfromtxt(mcmc_dir+'A.txt')
    Bs=np.genfromtxt(mcmc_dir+'B.txt')
    sig0s=np.genfromtxt(mcmc_dir+'sigma0.txt')
    lnA = np.log(As)
    print(np.mean(As), np.std(As))
    print(np.mean(Bs), np.std(Bs))
    print(np.mean(sig0s), np.std(sig0s))

    # plot the parameter constraints
    c = ChainConsumer()
    data=np.vstack( (lnA, Bs, sig0s) ).T
    c.add_chain(data, parameters=[r"$\ln{A}$", r"$B$", r"$\sigma_0$"], name='All bins')
    c.configure(colors=['k'], linestyles=[":"], shade=[True], shade_alpha=[0.5])
    c.plotter.plot(filename="posterior_hrun_allbins.png", figsize=(5,5))
    plt.show()
   


