#!/usr/bin/env python
from glob import glob
import numpy as np
import healpy as hp
import struct
# copied from Joe DeRose
# https://gist.github.com/j-dr/f39fd36a2b529af1412d9a398807977a


def read_radial_bin(filename, read_pos=False, read_vel=False, read_ids=False):
    """
    Read in a radial/hpix cell

    filename -- The name of the file to read, or a file object. If file
                object, will not be closed upon function return. Instead
                the pointer will be left at the location of the last
                data read.
    read_xxx -- Whether or not to read xxx

    positions and velocities are flattted
    use: reshape(npart, 3)
    """

    hdrfmt = 'QIIffQfdddd'
    idxfmt = np.dtype('i8')
    to_read = np.array([read_pos, read_vel, read_ids])
    fmt = [np.dtype(np.float32), np.dtype(np.float32), np.dtype(np.uint64)]
    item_per_row = [3,3,1]
    data = []

    opened = False
    if not hasattr(filename, 'read'):
        opened = True
        fp = open(filename, 'rb')
    else:
        fp = filename

    #read the header
    h = list(struct.unpack(hdrfmt, \
            fp.read(struct.calcsize(hdrfmt))))

    npart = h[0]
    indexnside = h[1]
    indexnpix  = 12*indexnside**2
    data.append(h)
    #read the peano index
    idx = np.fromstring(fp.read(idxfmt.itemsize*indexnpix), idxfmt)
    data.append(idx)

    if to_read.any():
        for i, r in enumerate(to_read):
            d = np.fromstring(fp.read(int(npart*item_per_row[i]*fmt[i].itemsize)), fmt[i])
            if r:
                data.append(d)
            if not to_read[i+1:].any():break

        if opened:
            fp.close()
            
    return data
   
def query_file(basepath, ra, dec, r):
    # r: comoving distance of the object in Mpc/h
    # sqrt(px**2 + py**2 + pz**2) in the halo catalogs

    rbin = int(r//25)

    path = '{}snapshot_Lightcone_{}_0'.format(basepath, rbin)
    
    hdr, idx = read_radial_bin(path)
    nside = hdr[2]
    
    pix = hp.ang2pix(nside, (90-dec)*np.pi / 180., ra * np.pi / 180., nest=True)
    return '{}snapshot_Lightcone_{}_{}'.format(basepath, rbin, pix)
    
    
    
#if __name__ == "__main__":
#    basepath = '/project/projectdirs/des/jderose/BCC/Chinchilla/Herd/Chinchilla-0/Lb1050/output/pixlc'
#    filename = query_file(basepath, ra=20, dec=30,r=30)
#    hdr, idx, pos, vel, ids = read_radial_bin(filename, read_pos=True, read_vel=True, read_ids=True)
#    npart = len(ids)
#    pos=pos.reshape(npart, 3)
#    print(pos)

  