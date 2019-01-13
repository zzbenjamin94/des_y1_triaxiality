#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
#import fitsio 
#from astropy.cosmology import FlatLambdaCDM
#cosmo = FlatLambdaCDM(H0=100, Om0=0.286)

from lightcone_query_ra_dec import query_file, read_radial_bin
from setup.setup import data_home_dir
import pyfits
#Read http://pythonhosted.org/pyfits/

datadir = data_home_dir()
### PARTICLES ####
#basepath = '/project/projectdirs/des/jderose/BCC/Chinchilla/Herd/Chinchilla-0/Lb1050/output/pixlc'

def read_halo_ptcl(ra_cen, dec_cen, red_cen, x_cen, y_cen, z_cen, Rmax):
    #basepath = '/project/projectdirs/des/jderose/BCC/Chinchilla/Herd/Chinchilla-0/Lb1050/output/pixlc/'
    basepath = datadir
    
    chi_cen = np.sqrt(x_cen**2 + y_cen**2 + z_cen**2)

    ang = (Rmax/chi_cen) * (180./np.pi) # angle to search particles (in degree)
    #print('ang', ang)

    x = [] # bad practice
    y = [] # bad practice!
    z = [] # bad practice!!
    filename_exists = []

    for ra_pm in [-ang,0,ang]: # some particle might be in another patch
        for dec_pm in [-ang,0,ang]:
            for chi_pm in [-Rmax,0,Rmax]:
                filename = query_file(basepath, ra=ra_cen+ra_pm, dec=dec_cen+dec_pm, r=chi_cen+chi_pm)
                if filename in filename_exists: 
                    #print('used')
                    pass
                else:
                    filename_exists.append(filename)
                    hdr, idx, pos = read_radial_bin(filename, read_pos=True)
                    npart = len(pos)//3
                    pos=pos.reshape(npart, 3)
                    dist = np.sqrt((pos[:,0] - x_cen)**2 + (pos[:,1] - y_cen)**2 + (pos[:,2] - z_cen)**2)
                    mask = (dist <= Rmax)
                    #plt.scatter(pos[mask,0], pos[mask,1], s=1)
                    x.extend(pos[mask,0].tolist()) # not append! 
                    y.extend(pos[mask,1].tolist())
                    z.extend(pos[mask,2].tolist())

    return np.array(x), np.array(y), np.array(z)
    

if __name__ == "__main__":
    ra_cen = 64.3599024
    dec_cen = 16.68787569
    red_cen = 0.23191588
    x_cen = 273.26001
    y_cen = 569.31482
    z_cen = 189.31299
    Rvir = 2.549908
    x, y, z = read_halo_ptcl(ra_cen, dec_cen, red_cen, x_cen, y_cen, z_cen, Rmax=Rvir)
    print('np.shape(x)', np.shape(x))
    #plt.figure(figsize=(7,7))
    plt.scatter(x,y,s=1)
    plt.savefig('../../plots/particles/particles_test2.png')
    outfile = open('particles_test2.dat','w')
    for i in range(len(x)):
        outfile.write('%12g %12g %12g \n'%(x[i],y[i],z[i]))
    outfile.close()
