import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterNode, fcluster
from scipy.spatial.distance import pdist, squareform
import gmaps
import gmaps.datasets

def limit_area(df_merged, state=None, minlat=-90, maxlat=90, minlong=-180, maxlong=180):
    
    df = df_merged.copy()
    
    #Filter by state
    if state:
        df = df[df['USPS_O']==state]

    #Filter by lat/long
    df = df[(df['INTPTLONG_O']<maxlong) & (df['INTPTLONG_O']>minlong)]
    df = df[(df['INTPTLONG_D']<maxlong) & (df['INTPTLONG_D']>minlong)]
    df = df[(df['INTPTLAT_O']<maxlat) & (df['INTPTLAT_O']>minlat)]
    df = df[(df['INTPTLAT_D']<maxlat) & (df['INTPTLAT_D']>minlat)]
    
    return df


def create_distance_matrix(df_input, method='absolute'):
    #Method = 'absolute' or 'perc_origin'
    
    #Pivot input data
    df = df_input.pivot(index='OFIPS', columns='DFIPS', values='FLOW')
    
    #Fill NAs
    df = df.fillna(0)
    
    #Remove IDs that don't appear in both origin and destination
    set_D = set(df_input['DFIPS'].unique())
    set_O = set(df_input['OFIPS'].unique())
    drop_ids = list((set_D | set_O) - (set_D & set_O))
    df = df.drop(index = drop_ids, columns=drop_ids, errors='ignore')
    
    #Calculate distances
    if method == 'absolute':
        df = df.values.max()/(1+df)
    elif method == 'perc_origin':
        df_max = df.sum(axis=1)
        df = df.divide(df_max, axis=0)
        df = 1/(1+df)
    else:
        RaiseError('Distance method not defined!')
    
    return df


if __name__ == '__main__':
    df_commute = pd.read_csv('data/commute_data.csv')
    df_tracts = pd.read_csv('data/census_tracts_2010.csv')
    df_tracts = df_tracts.rename(columns=lambda x: x.strip())
    
    # Merge on OFIPS = GEOID (federal representation of census tract ID)
    df_merged = df_commute.merge(df_tracts, how='inner', left_on='OFIPS', right_on='GEOID')
    df_merged = df_merged.merge(df_tracts, how='inner', left_on='DFIPS', right_on='GEOID', suffixes = ('_O','_D'))
    
    df_bayarea = limit_area(df_merged, minlat=35.959793, maxlat=38.419866, minlong=-123.355416, maxlong=-120.609292)
    
    df_distance_matrix = create_distance_matrix(df_bayarea, method='absolute')
    
    linkage_method = 'complete'
    Z = linkage(df_distance_matrix, linkage_method)
    
    k = 500
    df_results = pd.DataFrame(index = df_distance_matrix.columns, data=fcluster(Z,k,criterion='maxclust'))
    
    df_results = df_results.rename(columns={0: 'cluster_id'})
    
    df_results = df_results.merge(df_tracts, left_index=True, right_on='GEOID')
    
    plt.figure(figsize=(20,20))
    plt.scatter(df_results['INTPTLONG'],df_results['INTPTLAT'], c=df_results['cluster_id'].values, cmap='Paired')
    plt.ylim(36,38.5)
    plt.xlim(-123,-120.5)
    plt.show()