import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import gmaps
from matplotlib import cm
import math
from mpl_toolkits.basemap import Basemap

def get_commute_dfs():
    """ Read commute data, and return formatted df_merged and df_tracts """
    df_commute = pd.read_csv('data/commute_data.csv')
    df_tracts = pd.read_csv('data/census_tracts_2010.csv')
    df_tracts = df_tracts.rename(columns=lambda x: x.strip())

    # Merge on OFIPS = GEOID (federal representation of census tract ID)
    df_merged = df_commute.merge(df_tracts, how='inner', left_on='OFIPS', right_on='GEOID')
    df_merged = df_merged.merge(df_tracts, how='inner', left_on='DFIPS', right_on='GEOID', suffixes = ('_O','_D'))

    return df_merged, df_tracts

def limit_area(df_merged, region_name):

    #Get lat longs
    minlat, maxlat, minlong, maxlong = get_lat_long(region_name)

    df = df_merged.copy()

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

def plot_silhouette_score(df_distance_matrix, Z, region_name, max_k=30):
    scores = []
    for k in range(2, max_k):
        f_cluster = fcluster(Z,k,criterion='maxclust')
        score = silhouette_score(df_distance_matrix , f_cluster, metric='precomputed')
        scores.append([k, score])
    scores = pd.DataFrame(scores, columns=['k', 'score'])
    best_k = scores.sort_values('score', ascending=False)['k'].iloc[0]

    plt.figure(figsize=(12,8))
    plt.xlabel('k')
    plt.ylabel('score')
    xint = range(min(scores['k']), math.ceil(max(scores['k']))+1)
    plt.xticks(xint)
    plt.plot(scores['k'], scores['score'])
    plt.savefig(region_name + ' k-scores.png', bbox_inches='tight')

    return best_k

def plot_map(df_distance_matrix, df_tracts, Z, k, region_name, criterion='maxclust', a=2, cmap=cm.tab20, cnum=20, marker_size=50):
    #Gets results for plotting based on given linkage matrix and target k
    fc = fcluster(Z,k,criterion=criterion)
    df_results = pd.DataFrame(index=df_distance_matrix.columns, data=fc)
    df_results = df_results.rename(columns={0: 'cluster_id'})
    df_results = df_results.merge(df_tracts, left_index=True, right_on='GEOID')

    #Adds hc_level and alpha
    Z_df = pd.DataFrame(Z).astype('int')
    df_results = df_results.reset_index()
    step_0 = df_results.merge(Z_df, how='left', left_index=True, right_on=0).set_index(0)
    step_1 = df_results.merge(Z_df, how='left', left_index=True, right_on=1).set_index(1)
    df_results['hc_level']=step_0[3].fillna(step_1[3])
    df_results['alpha'] = a/df_results['hc_level']

    #Assign colors in RBG format
    cluster_order = list(df_results.groupby('cluster_id').count().sort_values('GEOID', ascending=False).index)

    colors = []
    for i, val in enumerate(cluster_order):
        (r, g, b, a) = cmap(i%cnum)
        colors.append([val, r, g, b])
    colors = pd.DataFrame(colors, columns=['cluster_id','r','g','b'])

    df_results = df_results.merge(colors, on='cluster_id')

    #Get color array for pyplot
    color_array_pyplot = get_rgba_pyplot(df_results)

    #Get lat longs
    minlat, maxlat, minlong, maxlong = get_lat_long(region_name)

    #Calculate scale
    s = marker_size/(maxlong - minlong)

    # Make the figure
    plt.figure(figsize=(14, 14))

    # Initialize the basemap
    m = Basemap(llcrnrlat = minlat,
                llcrnrlon = minlong,
                urcrnrlat = maxlat,
                urcrnrlon = maxlong)

    # Get the area of interest imagery
    m.arcgisimage(service='World_Street_Map', xpixels = 2500, alpha = 0.6)
    m.scatter(df_results['INTPTLONG'],df_results['INTPTLAT'], c=color_array_pyplot, s=s)
    plt.savefig(region_name + '_k=' + str(k) + '.png', bbox_inches='tight')


def get_rgba_gmaps(df_results):
    """
    Input: Results df including r, g, b, and alpha values
    Output: Array of colors (rgb + alpha) for scatterplotting
    """
    color_array = []
    for i, row in df_results.iterrows():
        r, g, b, a = row['r'], row['g'], row['b'], row['alpha']
        color_array.append((int(r*256), int(g*256), int(b*256), a))

    return color_array


def get_rgba_pyplot(df_results):
    """
    Input: Results df including r, g, b, and alpha values
    Output: Array of colors (rgb + alpha) for scatterplotting
    """
    color_array = []
    for i, row in df_results.iterrows():
        r, g, b, a = row['r'], row['g'], row['b'], row['alpha']
        color_array.append((r, g, b, a))

    return np.array(color_array)

def get_lat_long(region_name):
    #Define regions
    regions = {'Bay Area': (37, 38.5, -123, -121.5),
               'Southern California': (32.5, 34.75, -119.5, -116.75),
               'New York': (40.3, 41.3, -74.5, -73.5),
               'North East': (38.5, 43, -77.75, -70.25),
               'Texas Triangle': (29, 33.5, -99, -95)
              }
    try:
        minlat = regions[region_name][0]
        maxlat = regions[region_name][1]
        minlong = regions[region_name][2]
        maxlong = regions[region_name][3]
    except:
        raise Exception('region_name not found!')

    return minlat, maxlat, minlong, maxlong


if __name__ == '__main__':
    #Pick a region
    region_name = 'Southern California'

    #Load data
    df_merged, df_tracts = get_commute_dfs()

    #Filter to region
    df_filtered = limit_area(df_merged, region_name)

    #Calculate distance matrix
    df_distance_matrix = create_distance_matrix(df_filtered, method='absolute')

    #Caluclate linkage
    linkage_method = 'complete'
    Z = linkage(df_distance_matrix, linkage_method)

    #Plot silhoutte score
    best_k = plot_silhouette_score(df_distance_matrix, Z, region_name, max_k=30)

    #Pick a k (use best_k by default)
    k = best_k

    #Plot map
    plot_map(df_distance_matrix, df_tracts, Z, k, region_name, criterion='maxclust', a=2, cmap=cm.tab20, cnum=20, marker_size=50)
