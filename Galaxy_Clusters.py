
# coding: utf-8

# # Project with Yuanyuan on Orientational Bias in Cluster Mass
# ### Created Jan, 2018
# ### Zhuowen Zhang

# In[1]:

import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib.pyplot as plt
from code.plot_utils import plot_pretty
plot_pretty()

# setup notebook for inline figures
get_ipython().magic(u'matplotlib inline')


# In[123]:

from code.setup.setup import data_home_dir
import pyfits
#Read http://pythonhosted.org/pyfits/

datadir = data_home_dir()

#RedMapper Galaxy cluster data
redM_data = datadir+'buzzard-1_1.6_y3_run_redmapper_v6.4.20_lgt20_vl02_catalog.fit'
redM_list = pyfits.open(redM_data)
redM_data = redM_list[1].data

print "Number of clusters is ", len(redM_data)



# ### Extracting Halo and Cluster Files
# From buzzard and Chinchilla in Slac_Stanford simulation repository

# In[124]:

#Reading halo files
halo_filenum = [0,1,2,3,4,5,6,7,17,19,21,22,23,26,27]
halo_iter = np.shape(halo_filenum)[0]

#halos_data, the rec_array that combines all halo files, reads from first halo file, and then stacks to it.
halo_data = datadir+'Chinchilla-1_halos_lensed.'+str(halo_filenum[0])+'.fits'
halo_list = pyfits.open(halo_data)
cols = halo_list[1].data.columns
halos_data = pyfits.FITS_rec.from_columns(cols) #stores as record array

#check that halos_data match halo_data
print 'number of rows in halo_data/halos_data (initial) ', len(halos_data) 

#check that files_num has rows totaling the number of rows combined from each halo file
tot_row = halos_data.shape[0]

#Iterate through rest of halo files. Each iteration creates a new and updated halos_file record array
#with new rows added that correspond to data from each halo file.
for i in xrange(1,halo_iter):
    halo_data = datadir+'Chinchilla-1_halos_lensed.'+str(halo_filenum[i])+'.fits'
    halo_list = pyfits.open(halo_data)
    halo_data = halo_list[1].data

    nrows1 = halos_data.shape[0]
    nrows2 = halo_data.shape[0]
    nrows = nrows1 + nrows2
    tot_row += nrows2
    
    #halos_data.columns contains both name and data info of each columns. Extra rows are 0 or null.
    halos_data = pyfits.FITS_rec.from_columns(halos_data.columns, nrows=nrows) 

    #Fill in the extra rows with data from new halo file
    for colname in halo_data.columns.names:
        halos_data[colname][nrows1:] = halo_data[colname]


# In[125]:

#Check that halos_file is combining the files properly.
#For more info on record arrays, pyFITS:
#https://pythonhosted.org/pyfits/usage/table.html#table-data-as-a-record-array
#http://pythonhosted.org/pyfits/api/tables.html#pyfits.FITS_rec

#Check 1: Field values match that in the last halo file read
halo_data = datadir+'Chinchilla-1_halos_lensed.'+str(halo_filenum[-1])+'.fits' #last file
halo_list = pyfits.open(halo_data)
halo_data = halo_list[1].data
print 'Values for column name \'Z\' for halos_data and last halo file read'
print halos_data['Z'][-10::]
print halo_data['Z'][-10::]

#Check 2: The length of halos_file equals that of all halo_file rows combined.
if tot_row == len(halos_data):
    print "Number of rows match. It is ", len(halos_data)
    
#Check 3: Read the columns names and type of halos_data to see if makes sense. They do by inspection
#print halos_data.columns
#halo_data.columns
#halos_data['ID'][5000:5100]



# ### Halo and cluster matching algorithm
# 

# In[188]:

#Distance calculations for proximity matching
from colossus.cosmology import cosmology

#Q? Parameters need to change?
# define a vector of cosmological parameters of Via Lactea II cosmology:    
h = 0.73
my_cosmo = {'flat': True, 'H0': h*100, 'Om0': 0.238, 'Ob0': 0.045714, 'sigma8': 0.74, 'ns': 0.951}
# set my_cosmo to be the current cosmology
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)   

#Set to transverse or line of sight comoving distance? Can't tell. 
#comoving_r = cosmo.comovingDistance(zmin, zmax, transverse=True)


# In[126]:

#Cut off halos with mass less than 10^13 M_sun
mass_cutoff = halos_data['M200C'] >1e13
halos_data = halos_data[mass_cutoff]
print "After mass cut of M > 1e13 M_sun number of halos is ", len(halos_data)


# In[255]:

#Parameters for for matching
redM_RA = np.copy(redM_data['RA']); redM_DEC = np.copy(redM_data['DEC']); redM_z = np.copy(redM_data['Z_LAMBDA']) 
redM_lmda = np.copy(redM_data['LAMBDA_CHISQ']); halos_RA = np.copy(halos_data['RA']) 
halos_DEC = np.copy(halos_data['DEC']); halos_z = np.copy(halos_data['Z']); halos_M = np.copy(halos_data['M200C'])

#Convert RA, DEC to radians
redM_RA *= np.pi/180.; redM_DEC *= np.pi/180.; halos_RA *= np.pi/180.; halos_DEC *= np.pi/180.;
#print "redM_RA in Rad and degrees respectively"
#print redM_DEC, redM_data['DEC'][0:100]
#print np.where(halos_DEC <0 )

#Convert from z to comoving distance in fiducial cosmology
redM_comvr = cosmo.comovingDistance(0.0,redM_z)*h
halos_comvr = cosmo.comovingDistance(0.0, halos_z)*h


# In[234]:

#Debug this code. Something is wrong with it. 

#Cluster matching algorithm: Go down list of halos, 
#from most to least massive, and find potential cluster matches

#Find sorted indices for halos_M
halos_sort=np.asarray(sorted(range(len(halos_M)),key=lambda x:halos_M[x],reverse=True))
print "Number of halos is ", halos_sort.shape[0]

cl_match_ind = np.array([]) #indices of (potentially) matched clusters 
for i in [1]: #range(halos_sort.shape[0]):
    cur_halo_RA = halos_RA[halos_sort[i]]; cur_halo_DEC = halos_DEC[halos_sort[i]]
    cur_halo_comvr = halos_comvr[halos_sort[i]]; cur_halo_z = halos_z[halos_sort[i]]
       
    #Angle difference from RA, DEC using spherical law of cosines
    d_lmda = redM_RA - cur_halo_RA
    d_Sigma = np.arccos(np.sin(redM_DEC)*np.sin(cur_halo_DEC)+np.cos(redM_DEC)*np.cos(cur_halo_DEC)*np.cos(d_lmda))
    #print 'd_Sigma is ', d_Sigma[0:100]
    
    #comoving distance between halo and clusters from law of cosines
    #d_cmvr = np.sqrt(cur_halo_comvr**2.+redM_comvr**2. - 2.*redM_comvr*cur_halo_comvr*np.cos(d_Sigma))
    
    #Find clusters within redshift bin of +/- 0.1 
    zbin = 0.1
    zbin_cl_ind = np.asarray(np.where(np.abs(cur_halo_z-redM_z)<zbin))
    #Assume clusters that fall into this redshift range have the same redshift as the halo. 
    #Find the distance to the halo according to this assumption, and find ones local to the halo.
    
    #comoving distance between halo and clusters from law of cosines for clusters in redshift range
    d_cmvr = np.sqrt(2*cur_halo_comvr**2.*(1.-np.cos(d_Sigma[zbin_cl_ind])))
    
    #find local clusters
    max_d_cmvr = 1.0 #set to 1Mpc
    loc_cl_ind = zbin_cl_ind[np.where(d_cmvr<max_d_cmvr)]
    
    #if local clusters present find richest one
    if np.size(loc_cl_ind) > 0:
        cur_cl_match_ind = loc_cl_ind[np.argmax(redM_lmda[loc_cl_ind])]
    else:
        cur_cl_match_ind = -1 #-1 for no match
        
    cl_match_ind = np.concatenate((cl_match_ind, [cur_cl_match_ind]))
    
    if i%1e4 == 0:
        print "Matching halo number ", i
        print 'cl_matches are ', np.where(cl_match_ind > -1)
        print 'd_cmvr ', d_cmvr[0:50]
        print 'zbin_cl_ind is ', zbin_cl_ind
        


# In[ ]:




# In[208]:

z = np.linspace(0,1,1000)
comovingD = cosmo.comovingDistance(0.,z)
#plt.plot(z,comovingD)
#plt.show()
comovingD1 = comovingD[0:998].copy()
comovingD2 = comovingD[1:999].copy()
d_comovingD = comovingD2 - comovingD1
plt.plot(z[1:999], d_comovingD)
plt.show()


# In[ ]:



