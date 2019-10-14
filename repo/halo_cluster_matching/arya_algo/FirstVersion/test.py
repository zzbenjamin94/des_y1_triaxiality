import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd 
from membership_matching_algorithm import membership_matching, read_matched_clusters

table = Table.read('./data/G3CMockGal.fits')
GAMA_members = table.to_pandas()
GAMA_members['Pmem'] = 1.0
GAMA_members = GAMA_members[GAMA_members['GroupID'] > 0.0]

table = Table.read('./data/G3CMockFoFGroup.fits')
GAMA_clusters = table.to_pandas()

GAMA_clusters = GAMA_clusters[GAMA_clusters['Nfof'] > 20]
GAMA_cluster_members = GAMA_clusters.join(GAMA_members.set_index('GroupID'), on='GroupID', rsuffix='_c')


galaxies = GAMA_cluster_members[['HaloID', 'GroupID', 'Pmem']]
membership_matching(galaxies, 'GroupID', 'HaloID', 'Pmem')
matched_clusters = read_matched_clusters()


