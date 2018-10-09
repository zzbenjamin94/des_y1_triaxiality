import healpy as hp
import numpy as np
import fitsio

#file containing the rotation matrix 
rotfile = sys.argv[1]

#halofile
halofile = sys.argv[2]

with open(rotfile, 'r') as fp:
    rmat = pickle.load(fp)

h    = fitsio.read(halofile)

#rotate the positions
vec  = h[['PX','PY','PZ']].view((h['PX'].dtype,3))
rvec = np.dot(rmat, vec.T).T

#convert to angular coords
theta, phi = hp.vec2ang(rvec)

ra         = phi * 180. / np.pi
dec        = 90 - theta * 180 / np.pi

