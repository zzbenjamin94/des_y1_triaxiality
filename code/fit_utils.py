import numpy as np

def MCMCsample(x, nparams=2, nwalkers=100, nRval=100, modelpdf = None, params=[]):
    """
    MCMC sample implementing Goodman & Weare (2010) algorithm
    inner loop is vectorized
    
    written by Andrey Kravtsov (2014)
    """
    
    try:
        import acor
    except:
        raise Exception("acor package is not installed!")
        
    # parameters used to draw random number with the GW10 proposal distribution
    ap = 2.0; api = 1.0/ap; asqri=1.0/np.sqrt(ap); afact=(ap-1.0)

    # initialize some auxiliary arrays and variables 
    chain = []
    Rval = []

    naccept = 0; ntry = 0; nchain = 0
    mw = np.zeros((nwalkers,nparams)); sw = np.zeros((nwalkers,nparams))
    m = np.zeros(nparams)
    Wgr = np.zeros(nparams); Bgr = np.zeros(nparams); Rgr=np.zeros(nparams)
    
    mutx = []; taux = []
    for i in range(nparams): 
        mutx.append([]); #taux.append([])
        Rval.append([])

    gxo = np.zeros((2,nwalkers/2))
    gxo[0,:] = modelpdf(x[0,:,:], params); gxo[1,:] = modelpdf(x[1,:,:], params)
    
    converged = False;
    while not converged:
        # for parallelization (not implemented here but the MPI code can be found in code examples)
        # the walkers are split into two complementary sub-groups (see GW10)
        for kd in range(2):
            k = abs(kd-1)

            # vectorized inner loop of walkers stretch move in the Goodman & Weare sampling algorithm
            xchunk = x[k,:,:]
            jcompl = np.random.randint(0,nwalkers/2,nwalkers/2)
            xcompl = x[kd,jcompl,:]
            gxold  = gxo[k,:]
            zf= np.random.rand(nwalkers/2)   # the next few steps implement Goodman & Weare sampling algorithm
            zf = zf * afact; zr = (1.0+zf)*(1.0+zf)*api
            zrtile = np.transpose(np.tile(zr,(nparams,1))) # duplicate zr for nparams
            xtry  = xcompl + zrtile*(xchunk-xcompl)
            gxtry = modelpdf(xtry, params); gx    = gxold 
            ilow = np.where(gx<-100.); gx[ilow] = -100. # guard against underflow in regions of very low p
            gr   = gxtry - gx
            iacc = np.where(gr>0.)
            xchunk[iacc] = xtry[iacc]
            gxold[iacc] = gxtry[iacc]
            aprob = (nparams-1)*np.log(zr) + (gxtry - gx)
            u = np.random.uniform(0.0,1.0,np.shape(xchunk)[0])        
            iprob = np.where(aprob>np.log(u))
            xchunk[iprob] = xtry[iprob]
            gxold[iprob] = gxtry[iprob]
            naccept += len(iprob[0])

            x[k,:,:] = xchunk
            gxo[k,:] = gxold        
    
            for i in range(nwalkers/2):
                chain.append(np.array(x[k,i,:]))

            for i in range(nwalkers/2):
                mw[k*nwalkers/2+i,:] += x[k,i,:]
                sw[k*nwalkers/2+i,:] += x[k,i,:]**2
                ntry += 1

        nchain += 1
        
        # compute means for the auto-correlation time estimate
        for i in range(nparams):
            mutx[i].append(np.sum(x[:,:,i])/(nwalkers))

        #if nchain > 50*nRval: return chain, Rval

        # compute Gelman-Rubin indicator for all parameters
        if ( nchain >= nwalkers/2 and nchain%nRval == 0):
            # calculate Gelman & Rubin convergence indicator
            mwc = mw/(nchain-1.0)
            swc = sw/(nchain-1.0)-np.power(mwc,2)

            for i in range(nparams):
                # within chain variance
                Wgr[i] = np.sum(swc[:,i])/nwalkers
                # mean of the means over Nwalkers
                m[i] = np.sum(mwc[:,i])/nwalkers
                # between chain variance
                Bgr[i] = nchain*np.sum(np.power(mwc[:,i]-m[i],2))/(nwalkers-1.0)
                # Gelman-Rubin R factor
                Rgr[i] = (1.0 - 1.0/nchain + Bgr[i]/Wgr[i]/nchain)*(nwalkers+1.0)/nwalkers - (nchain-1.0)/(nchain*nwalkers)
                #tacorx = acor.acor(mutx[i])[0]; taux[i].append(np.max(tacorx))
                Rval[i].append(Rgr[i]-1.0)
            print "nchain=",nchain
            print "R values for parameters:", Rgr
            #print "tcorr =", np.max(tacorx)
            if np.max(np.abs(Rgr-1.0)) < 0.01: converged = True
        
    print "MCMC sampler generated ",ntry," samples using", nwalkers," walkers"
    print "with step acceptance ratio of", 1.0*naccept/ntry
    
    xh = zip(*chain)[0]; yh = zip(*chain)[1]
    
    # chop of burn-in period, and thin samples on auto-correlation time following Sokal's (1996) recommendations
    #nthin = int(tacorx)
    #nburn = int(20*nwalkers*nthin)
    #xh = xh[nburn::nthin]; yh = yh[nburn::nthin]
    
    chain = zip(xh,yh)

    return chain, Rval#, taux

def MCMCsample_init(nparams=2,nwalkers=100,x0=None,step=None):
    """
    distribute initial positions of walkers in an isotropic Gaussian around the initial point
    """
    np.random.seed(156)
    
    # in this implementation the walkers are split into 2 subgroups and thus nwalkers must be divisible by 2
    if nwalkers%2:
        raise ValueError("MCMCsample_init: nwalkers must be divisible by 2!")
         
    x = np.zeros([2,nwalkers/2,nparams])

    for i in range(nparams):
        x[:,:,i] = np.reshape(np.random.normal(x0[i],step[i],nwalkers),(2,nwalkers/2))
    return x
        
def gaussian2d(x, params=[1.0, 1.0, 0.95]):
    "2d Gaussian with non-zero correlation coefficient r"
    #print "gaussian2d:", np.shape(params), params
    sig1=params[0]; sig2=params[1]; r=params[2]
    r2 = r*r
    prob = -0.5*((x[:,0]/sig1)**2+(x[:,1]/sig2)**2-2.0*r*x[:,0]*x[:,1]/(sig1*sig2))/(1.0-r2) - np.log((2.*np.pi*sig1*sig2)/np.sqrt(1.0-r2))
    return prob
    
def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def plot_chain2d(xh,yh,x0,y0,Rxval, Ryval, nRval=None, modelpdf=None, params=[]):
    from matplotlib import pylab as plt
    from matplotlib.colors import LogNorm
    import scipy.optimize as opt
            
    x = np.arange(np.min(xh)-1.0,np.max(xh)+1,0.05); 
    y = np.arange(np.min(yh)-1.0,np.max(yh)+1,0.05)

    X, Y = np.meshgrid(x,y)
    xs = np.array(zip(X,Y))
    Z = modelpdf(xs, params=params)

    plt.rc('font', family='sans-serif', size=16)
    fig=plt.figure(figsize=(15,15))
    plt.subplot(221)
    plt.title('Gelman-Rubin indicator')
    #plt.plot(chain)
    plt.yscale('log')
    iterd = np.linspace(0,len(xh)-1)
    xp = nRval*np.arange(np.size(Rxval))
    #print "x, Rxval:", np.size(xp), np.size(Rxval)
    plt.plot(xp, Rxval)
    plt.plot(xp, Ryval)


    ax = plt.subplot(222)
    plt.hist2d(xh,yh, bins=100, norm=LogNorm(), normed=1)
    plt.colorbar()

    dlnL2= np.array([2.30, 9.21]) # contours enclosing 68.27 and 99% of the probability density
    xc = np.zeros([1,2])
    Lmax = modelpdf(xc, params=params)
    lvls = Lmax/np.exp(0.5*dlnL2)
    cs=plt.contour(X,Y,Z, linewidths=(1.0,2.0), colors='black', norm = LogNorm(), levels = lvls, legend='target pdf' )

    xmin = -4.0; xmax = 4.0; nbins = 80; dx = (xmax-xmin)/nbins; dx2 = dx*dx
    H, xbins, ybins = np.histogram2d(xh, yh,  bins=(np.linspace(xmin, xmax, nbins), np.linspace(xmin, xmax, nbins)))

    H = np.rot90(H); H = np.flipud(H); Hmask = np.ma.masked_where(H==0,H)
    H = H/np.sum(H)    

    clevs = [0.99,0.6827]

    lvls =[]
    for cld in clevs:  
        sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )  
        lvls.append(sig)

    plt.contour(H, linewidths=(3.0,2.0), colors='magenta', levels = lvls, norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
 
    plt.title('MCMC samples vs target distribution')

    labels = ['68.27%', '99%']
    for i in range(len(labels)):
        cs.collections[i].set_label(labels[i])

    plt.legend(loc='upper left')
    plt.ylabel('y')
    plt.xlabel('x')
    
    plt.show()

if __name__ == '__main__':
    
    import py_compile
    import os
    from setup import code_home_dir
    
    py_compile.compile(os.path.join(code_home_dir(), 'read_data.py'))
    # set average initial position of the walkers
    x0 = 1.+np.zeros(2); step = 0.1*np.ones_like(x0);
    
    # initialize walkers
    xwalk = MCMCsample_init(2, 200, x0, step)
    
    # run the sampler
    nRval = 100 # record GR R and acor's tau each nRval'th step
    params = [1.0, 1.0, 0.95]
    chain, Rval = MCMCsample(xwalk, nparams=2, nwalkers=200, nRval=nRval, modelpdf=gaussian2d, params=params)
    x0 *= nRval
    # plot distribution of parameter values in the parameter space
    plot_chain2d(zip(*chain)[0],zip(*chain)[1],x0[:][0],x0[:][1],
                Rval[0],Rval[1], nRval=nRval, 
                modelpdf=gaussian2d, params=params)
 
