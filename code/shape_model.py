
# coding: utf-8

# ### Object Oriented Programming for Shape Calculation

# In[1]:

# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt

from glob import glob
import numpy as np
import healpy as hp
import struct

import code.setup.setup as setup
import pyfits
datadir = setup.data_home_dir()

#Need to set basepath to the dark matter halo particle files. 
#basepath = datadir

"""
read_radial_bin() and query_file() are global functions called by the halo_shape object to read halo particle files.
"""

def read_radial_bin(filename, read_pos=False, read_vel=False, read_ids=False):
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
    h = list(struct.unpack(hdrfmt, fp.read(struct.calcsize(hdrfmt))))

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



#halo_shape object. 

class halo_shape(object):
    
    def __init__(self, ra_cen=None, dec_cen=None, red_cen=None, x_cen=None, y_cen=None, z_cen=None, Rmax=None, verbose = False):
        
        assert ra_cen is not None, "ra_cen is None"
        assert dec_cen is not None, "dec_cen is None" 
        assert red_cen is not None, "red_cen is None" #redshift
        assert x_cen is not None, "x_cen is None" #x, y, z in Mpc/h
        assert y_cen is not None, "y_cen is None"
        assert z_cen is not None, "z_cen is None" 
        assert Rmax is not None, "Rmax is None" #Virial radius in Mpc/h
        
        self.ra_cen = ra_cen
        self.dec_cen = dec_cen
        self.red_cen = red_cen
        self.x_cen = x_cen
        self.y_cen = y_cen
        self.z_cen = z_cen
        self.Rmax = Rmax
        
        #Some statement if verbose is true
        if verbose:
            print "Verbose"
            
        #Other non-instantiated parameters of the halo_shape object
        self.axis_ratio = np.array([1,1,1])
        self.axis_dir = np.identity(3)
        self.converge = False
        self.ptcl_num = 0
        
        #Choose between high_z and low_z folders
        if (red_cen >= 0.34) & (red_cen < 0.90):
            self.basepath = setup.buzzard_particles_highz_dir()
        elif (red_cen < 0.34):
            self.basepath = setup.buzzard_particles_lowz_dir()
        else:
            raise Exception('redshift z={} outside of Buzzard range'.format(red_cen))
        return 
     
    """
    Iteratively solves the axis ratios and direction until convergence criterion for 
    envelope and shape of halo particles inside the envelope is met. 
    
    #Convergence criterion for envelope same as for particles inside envelope. 
    """
    def evolve(self):
        ptcl_coord = self.read_halo_ptcl()
        self.ptcl_num = len(ptcl_coord[0])
	#too few particles proabably won't converge and may generate error. 
        if self.ptcl_num  < 100:
	    #print "Initial number of particles is ", len(ptcl_coord[0]) 
            return 
        
        #Converge criterion for the envelope
        a = self.axis_ratio[0]; b = self.axis_ratio[1]; c = self.axis_ratio[2]
        q_prev = c/a; s_prev = b/a
        
        conv_err = 1e-6
        conv_iter_max = 100
        env_converge=False
        ptcl_converge=True
        conv_iter=0
        while (ptcl_converge) & (not env_converge) & (conv_iter < conv_iter_max):
            #Performs quad_moment only if there are enough particles
	    if self.ptcl_num > 100:            
                self.ptcl_num, ptcl_converge, self.axis_ratio, self.axis_dir = self.quad_moment(ptcl_coord)
            #print self.axis_ratio, self.axis_dir            
                a = self.axis_ratio[0]; b = self.axis_ratio[1]; c = self.axis_ratio[2]
                q_cur = c/a; s_cur = b/a

                conv_s = np.abs(1 - s_cur/s_prev); conv_q = np.abs(1 - q_cur/q_prev)
                env_converge = (conv_s < conv_err) & (conv_q < conv_err)
    
                q_prev = q_cur; s_prev = s_cur
                conv_iter += 1

	    else:
		self.converge = False
		return


        
        self.converge = ptcl_converge & env_converge & (conv_iter < conv_iter_max)
        return   
        
    """
    Returns the number of DM particles inside halo
    """
    def get_ptcl_num(self):
        return self.ptcl_num
        
    """
    Returns whether shape of halo converge. Need for the envelope and particle inside envelope both to converge
    """
    def get_converge(self):
        return self.converge
    
    """
    Returns array of axis ratio a,b,c -- major, intermediate, minor axis
    """
    def get_axis_ratio(self):
        return self.axis_ratio
    
    """
    Returns axis direction lx, ly, lz -- minor, intermeidate, major axis (different order from a,b,c)
    """
    def get_axis_dir(self):
        return self.axis_dir
    
        
    """
    Reads out the particle positions of those that belong within a given distance to the halo center.
    Inputs halo position and reads halo particle files adjacent to the halo. Calls read_radial_bin() and 
    query_file() to read halo particles. 
    
    Inputs:
    ra_cen, de_cen: RA, DEC of halo center
    red_cen: redshift 
    x_cen, y_cen, z_cen: Unrotated position of halo in Mpc/h
    Rmax: max radius to enclose particles in Mpc/h. 
    
    Returns:
    (3,n) list of particle positions
    """
    def read_halo_ptcl(self):
        r_search = 2*self.Rmax #Search within twice virial radius, large enough to include all potential particles. 
        chi_cen = np.sqrt(self.x_cen**2 + self.y_cen**2 + self.z_cen**2)
        ang = (r_search/chi_cen) * (180./np.pi) # angle to search particles (in degree)

        x = [] # bad practice
        y = [] # bad practice!
        z = [] # bad practice!!
        filename_exists = []

        #Change this part to vary the radius of the envelope. 
        for ra_pm in [-ang,0,ang]: # some particle might be in another patch
            for dec_pm in [-ang,0,ang]:
                for chi_pm in [-r_search,0,r_search]:
                    
                    filename = query_file(self.basepath, ra=self.ra_cen+ra_pm, dec=self.dec_cen+dec_pm, r=chi_cen+chi_pm)
                    if filename in filename_exists: 
                        #print('used')
                        pass
                    else:
                        filename_exists.append(filename)
                        hdr, idx, pos = read_radial_bin(filename, read_pos=True)
                        
                        #pos relative to center less than r_search
                        npart = len(pos)//3
                        pos=pos.reshape(npart, 3) 
                        dist_x = pos[:,0]-self.x_cen; dist_y = pos[:,1]-self.y_cen; dist_z = pos[:,2]-self.z_cen;
                        dist = np.sqrt(dist_x**2. + dist_y**2. + dist_z**2.)
                        mask = (dist <= r_search)
                        x.extend(pos[mask,0].tolist()) # not append! 
                        y.extend(pos[mask,1].tolist())
                        z.extend(pos[mask,2].tolist())

        return [np.array(x), np.array(y), np.array(z)]    
    
  
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

    """
    Follow convention of Ken Osato: Use reduced quadropole moment to find axis ratio of ellipsoidal cluster
    1. Project onto principle axes spitted out by quadropole tensor
    2. Do not remove particles. Particles chosen for those inside Rvir
    3. Use Reduced tensor
    4. q, s refer to ratio of minor to major, and intermediate to major axis

    Returns:
    converge -- Boolean
    [a,b,c] -- normalized major, intermediate, minor axes lengths (only ratio matters in reduced tensor)
    [lx, ly, lz] -- direction of minor, intermediate, major
    """
    def quad_moment(self, ptcl_coord):
        
        #Selects particle with elliptical radius within Rmax. The axis direction and ratio are from 
        #the object itself outside the function, different from the axis direction and ratio generated
        #from the quad_moment() function. 
        
        #[a,b,c] -- normalized major, intermediate, minor axes lengths (only ratio matters in reduced tensor)
        #[lx, ly, lz] -- direction of minor, intermediate, major
        a = self.axis_ratio[0]; b = self.axis_ratio[1]; c = self.axis_ratio[2]
        q = c/a; s = b/a
        lx = self.axis_dir[0]; ly=self.axis_dir[1]; lz=self.axis_dir[2]
        
        ptcl_coord_x = ptcl_coord[0]; ptcl_coord_y = ptcl_coord[1]; ptcl_coord_z = ptcl_coord[2]
        rx = ptcl_coord_x - self.x_cen
        ry = ptcl_coord_y - self.y_cen
        rz = ptcl_coord_z - self.z_cen
        r_carte = np.array([rx, ry, rz]).T
        rx_proj = np.dot(r_carte, lx) 
        ry_proj = np.dot(r_carte, ly)
        rz_proj = np.dot(r_carte, lz)   
        R_range = np.sqrt((rx_proj/q)**2. + (ry_proj/s)**2. + rz_proj**2.) #Elliptical radius

        #Choose particles inside elliptical Rvir
        ptcl_range = np.where(R_range < self.Rmax)
        rx = rx[ptcl_range]; ry = ry[ptcl_range]; rz = rz[ptcl_range]

        num_mem_ptcl = len(rx)
	if num_mem_ptcl < 100:
	    return 0, False, [1,1,1], np.eye(3)
        #print "Number of particles inside virial radius is ", num_mem_ptcl

        
        
        #Part below same as original (before making into class object). The axis ratio a,b,c
        #and directions lz,ly,lx are different from the one called from the object. 
        #The ratio and direction if converged will become that of the object. 
        
        #Building quadrupole tensor. 
        Rp = np.sqrt(rx**2. + ry**2. + rz**2.)
        r = np.matrix([rx,ry,rz])
        r_rdu = r/Rp
        M_rdu = r_rdu*r_rdu.T #Initial quadrupole tensor before iteration

        #Finding eigvec, eigval
        M_eigval, M_eigvec = np.linalg.eig(M_rdu)
        sort_eigval = np.argsort(M_eigval)[::-1]
        a, b, c = np.sqrt(M_eigval[sort_eigval]/num_mem_ptcl) #a, b, c major, intermediate, minor
        lx, ly, lz = M_eigvec.T[sort_eigval][::-1] #lx, ly, lz minor, intermediate, major (order reversed from a, b, c)
        lx = np.array(lx)[0]; ly = np.array(ly)[0]; lz = np.array(lz)[0]

        #Sanity check
        """
        print "r_rdu", r_rdu
        check_eig = M_rdu.dot(lx) - num_mem_ptcl*c**2.*lx
        print "M_rdu.dot(lx) ", np.dot(np.array(M_rdu), lx)
        print "check_eig ", check_eig
        print "lx is ", lx
        print "M_eigvec.T[sort_eigval], ", M_eigvec.T[sort_eigval]
        print "M_eigvec[:,0] ", M_eigvec[:,0]
        print "M_eigvec[sort_eigval] ", M_eigvec[sort_eigval]
        print "M_eigvec", M_eigvec
        print "sort_eigval ", sort_eigval
        """

        #Initial conditions
        q_prev = 1.; s_prev = 1.
        converge = False
        conv_iter = 0

        P_tot = np.eye(3) #the multiplicative product of all projections done over each iteration
        while (not converge) & (conv_iter < 100):
            #Change of basis
            P_axis = np.matrix([lx,ly,lz])
            P_tot = P_axis*P_tot
            r_proj = P_axis*r
            rx = np.array(r_proj[0,:])[0]; ry = np.array(r_proj[1,:])[0]; rz = np.array(r_proj[2,:])[0]

            #New iteration
            q_cur = c/a; s_cur = b/a #Osato conventaion
            Rp = np.sqrt((rx/q_cur)**2. + (ry/s_cur)**2. + rz**2.)
            r = np.matrix([rx, ry, rz])
            r_rdu = r/Rp
            M_rdu = r_rdu*r_rdu.T
            M_eigval, M_eigvec = np.linalg.eig(M_rdu)
            sort_eigval = np.argsort(M_eigval)[::-1]
            a, b, c = np.sqrt(M_eigval[sort_eigval]/num_mem_ptcl)
            lx, ly, lz = M_eigvec.T[sort_eigval][::-1]
            lx = np.array(lx)[0]; ly = np.array(ly)[0]; lz = np.array(lz)[0]

	    #if unit-lengths are too small may generate error. Exit function as not converged
	    if any([x < 0.01 for x in [a,b,c]]):
		print "Error: axis_len too small. a,b,c={},{},{}".format(a,b,c)
	    	return 0, False, [1,1,1], np.eye(3)

	    #test converge
            conv_err = 1e-6
            conv_s = np.abs(1 - s_cur/s_prev); conv_q = np.abs(1 - q_cur/q_prev)
            converge = (conv_s < conv_err) & (conv_q < conv_err)
            conv_iter += 1
            q_prev = q_cur; s_prev = s_cur

        #find lx, ly, lz in original basis
        P_inv = np.linalg.inv(P_tot)
        l_new_basis = np.matrix([lx,ly,lz]).T
        l_orig_basis = np.transpose(P_inv*l_new_basis)
        lx_orig = np.array(l_orig_basis[0])[0]
        ly_orig = np.array(l_orig_basis[1])[0]
        lz_orig = np.array(l_orig_basis[2])[0]


        return num_mem_ptcl, converge, [a,b,c], np.array([lx_orig, ly_orig, lz_orig])



