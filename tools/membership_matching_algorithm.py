import pandas as pd
import numpy as np
import os

from setup.setup import tools_home_dir, home_dir, bigdata_home_dir
homedir = home_dir()

def membership_matching(members, measured_cluster_id_label, true_cluster_id_label, membership_probability_label):
    """ a membership matching algorithm
    it saves the result of matched output in the following folder "./data/clusters/"
    
    members: a pandas DataFrame with the following columns
        measured_cluster_id_label, 
        true_cluster_id_label, 
        membership_probability_label,
        
    return:
        None
    """
    
    """
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    
    if not os.path.exists('./data/matched/'):
        os.makedirs('./data/matched/')
    """
    
    clusters = members.groupby(measured_cluster_id_label)
    cluster_key = clusters.groups.keys()
    
    for cluster_key in clusters.groups.keys():
        
        cluster_i =  clusters.get_group(cluster_key)
        n_members = np.sum(cluster_i[membership_probability_label])

        true_cluster = cluster_i.groupby(true_cluster_id_label).agg({membership_probability_label:'sum'})
        true_cluster['Strength'] = true_cluster[membership_probability_label] / n_members
        true_cluster[measured_cluster_id_label] = cluster_key
        true_cluster.sort_values('Strength', ascending=False, inplace=True)
        true_cluster['Rank'] = np.arange(1, len(true_cluster)+1)
        
        true_cluster.to_csv(homedir+'output/buzzard/matched_Farahi16/'+'%i.csv'%cluster_key)


def read_matched_clusters():
    
    cluster_files = [f for f in os.listdir(homedir+'output/buzzard/matched_Farahi16/') if
                     os.path.isfile(os.path.join(homedir+'output/buzzard/matched_Farahi16/', f))]
    
    matched_clusters = [] 

    for file_name in cluster_files:
        matched_clusters += [pd.read_csv(homedir+'output/buzzard/matched_Farahi16/'+file_name)]
    
    matched_clusters = pd.concat(matched_clusters)
    matched_clusters.to_csv(homedir+'output/buzzard/matched_Farahi16/'+'matched_cluster_catalog.csv', index=False)
    
    return matched_clusters

