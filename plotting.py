import numpy as np
import pandas as pd
from model import *
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterNode, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import seaborn as sns
import gmaps
import gmaps.datasets

def hc_level_alpha(df_results, Z_df, a=2):
    """
    Input: Df of hierarchical clustering results, linkage matrix, and alpha multiplier
    Output: returns df with hc_level and alpha added
    """
    df_results_2 = df_results.copy().reset_index()
    step_0 = df_results_2.merge(Z_df, how='left', left_index=True, right_on=0).set_index(0)
    step_1 = df_results_2.merge(Z_df, how='left', left_index=True, right_on=1).set_index(1)
    df_results_2['hc_level']=step_0[3].fillna(step_1[3])
    df_results_2['alpha'] = a/df_results_2['hc_level']
    
    return df_results_2

def plot_bayarea(df_merged, df_tracts, dist_method='absolute', link_method='complete', k=20, alpha=2, cmap=cm.Dark2, cnum=8):
    """ Runs hierarchical clustering and plotting for bay area case """
    # Limits view to bay area
    df_bayarea = limit_area(df_merged,minlat=36.5,maxlat=38.5,minlong=-123,maxlong=-121)
    
    # Performs clustering based on kwargs
    df_results, Z_df = commute_hc(df_bayarea, df_tracts, dist_method=dist_method, link_method=link_method,k=k)

    # Adds alpha to each data point based on level of clustering
    df_results_alpha = hc_level_alpha(df_results, Z_df, a=alpha)
    
    # Gets colors for clusters by rgba based on input cmap
    df_results_colored = add_colors(df_results_alpha, cmap=cmap, cnum=cnum)
    color_array = get_rgba(df_results_colored)

    # Scatterplots colored results
    plot_hc(df_results_colored, color_array)

def commute_hc(df_merged, df_tracts, dist_method='absolute', link_method='complete', k=3):
    """
    Performs hierarchical clustering on commute data, returning results and linkage matrix Z
    Input: 
    """
    df_distance_matrix = create_distance_matrix(df_merged, method=dist_method)
    Z = linkage(df_distance_matrix, link_method)
    Z_df = pd.DataFrame(Z).astype('int')
    
    fc = fcluster(Z,k,criterion='maxclust')
    df_results = pd.DataFrame(index=df_distance_matrix.columns, data=fc)
    df_results = df_results.rename(columns={0: 'cluster_id'})
    df_results = df_results.merge(df_tracts, left_index=True, right_on='GEOID')

    return df_results, Z_df

def add_colors(df_results, cmap=cm.Dark2, cnum=8):
    """ 
    Input: Cluster results df and target cmap
    Output: Dataframe with colors assigned to clusters in RGB format
    """
    cluster_order = list(df_results.groupby('cluster_id').count().sort_values('GEOID', ascending=False).index)

    colors = []
    for i, val in enumerate(cluster_order):
        (r, g, b, a) = cmap(i%cnum)
        colors.append([val,r, g, b])
    colors = pd.DataFrame(colors, columns=['cluster_id','r','g','b'])

    df_results_2 = df_results.merge(colors, on='cluster_id')

    return df_results_2

def get_rgba(df_results):
    """ 
    Input: Results df including r, g, b, and alpha values
    Output: Array of colors (rgb + alpha) for scatterplotting 
    """
    color_array = []
    for i, row in df_results.iterrows():
        r, g, b, a = row['r'], row['g'], row['b'], row['alpha']
        color_array.append([r, g, b, a])

    return color_array

def plot_hc(df_results, color_array, figsize=(20,20)):
    """
    Input: Results of hierarchical clustering to be plotted, and respective rgba color mapping and figsize
    Output: None
    """
    plt.figure(figsize=figsize)
    plt.scatter(df_results['INTPTLONG'],df_results['INTPTLAT'], c=color_array)
    plt.show()
