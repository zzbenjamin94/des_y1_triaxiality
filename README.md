# DES Y3 Cluster Triaxiality
#### Zhuowen Zhang 
#### README file updated Sept. 14, 2018

## Datasets
###### * darksky particle catalog
        *public catalog for DM particles at z=0 (what is the redshift range?? )snapshot 
        *high mass resolution DM particle catalog for convergent surface density profiles at low redshift
        *What is url?
        
        
###### *Buzzard DM Halo
        *Buzzard simulations of DM halo using ROCKSTAR halo finder on lightcone to z < 0.90
        *unique FITS files do not have overlapping halos from different simulation boxes in 0.25 < z < 0.35
        *halos need to be rotated using desy3_irot.pkl 
        *NERSC link: /global/homes/j/jderose/des/jderose/catalog/halos/semihemisphere/buzzard/flock/buzzard-y3/Chinchilla-0_halos_unique.*.fits
        
###### *redMaPPer cluster finder
        *redmaPPer cluster finder from same Buzzard simulation as DM Halo
        *lambda cutoff at l>20
        *http://www.slac.stanford.edu/~jderose/bcc/catalog/redmapper/y3/buzzard/flock/buzzard-0/a/buzzard-0_1.6_y3_run_redmapper_v6.4.20_lgt20_vl02_catalog.fit
        
###### *Buzzard DM particle 
        *matching DM particles to Buzzard DM Halo simulation
        *mass resolution of at z < 0.34; < M < at 0.34 < z < 0.90 
        *high z ptcls at NERSC (neeed to ask Joe to put on your scratch space, keep for 3 months) /project/projectdirs/des/jderose/BCC/Chinchilla/Herd/Chinchilla-0/Lb1050/output/pixlc
        *low z ptcls at NERSC: /project/projectdirs/des/jderose/BCC/Chinchilla/Herd/Chinchilla-0/Lb1050/output/pixlc
        
## Tools and Configuration
* __setup__: need to append the DES_Galaxy_Cluster folder path to sys.path to call relative imports of .py files in the subfolders. 
* Need to change directories to your own in tools.setup.setup.py
 
## Cosmology and Parameter Choices

## Main Modules

## Cosmology Dependent Runs

## Non-stardard Packages to Install
*CLASS
*Cosmosis
*cluster_toolkit
*pymc
*chainconsumer
*Astropy ? 

        
        
