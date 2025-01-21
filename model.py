import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.path import Path
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import pickle
from matplotlib import cm
import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
from mpl_toolkits.basemap import Basemap


class CommuteArea:
    def __init__(self, region_name):
        self.region_name = region_name
        self.border = self.get_borders()

        df = pd.DataFrame(data=self.border, columns=['long', 'lat'])

        self.min_long, self.min_lat = df.min().values
        self.max_long, self.max_lat = df.max().values


    def get_borders(self):
        # List of points on map in (lon, lat) format
        a = [-126.697218, 42]
        b = [-123.608066, 42]
        c = [-123.516729, 40.520434]
        d = [-121.619774, 38.031762]
        e = [-121.627565, 37.483266]
        f = [-120.483437, 36.085636]
        g = [-121.682342, 35.436459]
        h = [-122.683520, 42]
        i = [-117.920691, 35.313449]
        j = [-119.290395, 34.634641]
        k = [-116.616020, 36.011754]
        l = [-114.133531, 32.553095]
        m = [-119.632172, 32.383832]
        n = [-120, 42]

        # Polygons defining borders of regions
        borders = {
            'California': [a, n, l, m],
            'Southern California': [j, i, k, l, m, g, f],
            'Central Valley': [b, h, i, j, e, d, c],
            'Northern California': [a, b, c, d, e, f, g]}

        return np.array(borders[self.region_name])


    def load_data(self):
        """ Read commute data, and return formatted df_merged and df_tracts """
        print('Loading commute data...')
        self.df_commute = pd.read_csv('data/commute_data.csv')
        self.df_tracts = pd.read_csv('data/census_tracts_2010.csv').rename(columns=lambda x: x.strip())


    def merge_data(self, cached=False):
        print('Merging commute data...')
        # Merge on OFIPS = GEOID (federal representation of census tract ID)
        self.df_merged = self.df_commute.merge(self.df_tracts, how='inner', left_on='OFIPS', right_on='GEOID')
        self.df_merged = self.df_merged.merge(self.df_tracts, how='inner', left_on='DFIPS', right_on='GEOID', suffixes = ('_O','_D'))


    def limit_area(self):
        print('Filtering commute data...')
        p = Path(self.border)

        #Find all origin pairs in border
        x_origin, y_origin = self.df_merged['INTPTLONG_O'].values, self.df_merged['INTPTLAT_O'].values
        x_origin, y_origin = x_origin.flatten(), y_origin.flatten()
        points_origin = np.vstack((x_origin,y_origin)).T 

        self.df_merged['origin_mask'] = p.contains_points(points_origin)

        #Find all destination pairs in border
        x_destination, y_destination = self.df_merged['INTPTLONG_D'].values, self.df_merged['INTPTLAT_D'].values
        x_destination, y_destination = x_destination.flatten(), y_destination.flatten()
        points_destination = np.vstack((x_destination, y_destination)).T 

        self.df_merged['destination_mask'] = p.contains_points(points_destination)

        # Filter out records where either origin or destination are not in border
        self.df_merged = self.df_merged.loc[self.df_merged['origin_mask'] & self.df_merged['destination_mask']]
        self.df_merged.drop(['origin_mask', 'destination_mask'], axis=1, inplace=True)


    def create_distance_matrix(self, distance_method='absolute', linkage_method = 'complete'):
        print('Creating distance matrix...')

        #Pivot input data
        self.df_distance_matrix = self.df_merged.pivot(index='OFIPS', columns='DFIPS', values='FLOW')

        #Fill NAs
        self.df_distance_matrix = self.df_distance_matrix.fillna(0)

        #Remove IDs that don't appear in both origin and destination
        set_D = set(self.df_merged['DFIPS'].unique())
        set_O = set(self.df_merged['OFIPS'].unique())
        drop_ids = list((set_D | set_O) - (set_D & set_O))
        self.df_distance_matrix.drop(index = drop_ids, columns=drop_ids, errors='ignore', inplace=True)

        #Calculate distances
        if distance_method == 'absolute':
            self.df_distance_matrix = self.df_distance_matrix.values.max()/(1+self.df_distance_matrix)
        elif distance_method == 'perc_origin':
            df_max = self.df_distance_matrix.sum(axis=1)
            self.df_distance_matrix = self.df_distance_matrix.divide(df_max, axis=0)
            self.df_distance_matrix = 1/(1+self.df_distance_matrix)
        else:
            raise Exception('Distance method not defined!')


    def calculate_linkage(self, linkage_method='complete', cached=False):
        
        print('Calculating linkage...')
        self.Z = linkage(self.df_distance_matrix, linkage_method).astype('float')  


    def plot_silhouette_score(self, max_k=30):
        scores = []
        for k in range(2, max_k):
            f_cluster = fcluster(self.Z,k,criterion='maxclust')
            score = silhouette_score(self.df_distance_matrix , f_cluster, metric='precomputed')
            scores.append([k, score])
        scores = pd.DataFrame(scores, columns=['k', 'score'])
        scores.to_csv(self.region_name + '/' + self.region_name + '_k-scores.csv')
        self.best_k = scores.sort_values('score', ascending=False)['k'].iloc[0]
        print('Best clustering found at k=' + str(self.best_k))

        plt.figure(figsize=(12,8))
        plt.xlabel('k')
        plt.ylabel('score')
        xint = range(int(min(scores['k'])), int(math.ceil(max(scores['k'])))+1)
        plt.xticks(xint)
        plt.plot(scores['k'], scores['score'])
        plt.savefig(self.region_name + '/' + self.region_name + '_k-scores.png', bbox_inches='tight', dpi=600)


    def plot_map(self, k, criterion='maxclust', a=2, cmap=cm.tab20, cnum=20, marker_size=50):
        print('Plotting map for k=' + str(k) + '...')

        #Gets results for plotting based on given linkage matrix and target k
        fc = fcluster(self.Z, k, criterion=criterion)
        df_results = pd.DataFrame(index=self.df_distance_matrix.columns, data=fc)
        df_results = df_results.rename(columns={0: 'cluster_id'})
        df_results = df_results.merge(self.df_tracts, left_index=True, right_on='GEOID')

        #Adds hc_level and alpha
        Z_df = pd.DataFrame(self.Z).astype('int')
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
        color_array_pyplot = self.get_rgba_pyplot(df_results)

        #Calculate scale
        s = marker_size/(self.max_long - self.min_long)

        # Make the figure
        plt.figure(figsize=(14, 14))

        # Initialize the basemap
        m = Basemap(resolution='h',
                    llcrnrlat = self.min_lat,
                    llcrnrlon = self.min_long,
                    urcrnrlat = self.max_lat,
                    urcrnrlon = self.max_long)

        # Get the area of interest imagery
        m.arcgisimage(service='World_Street_Map', xpixels = 2000, alpha = 0.6)

        m.scatter(df_results['INTPTLONG'], df_results['INTPTLAT'], c=color_array_pyplot, s=s)
        plt.savefig(self.region_name + '/' + self.region_name + '_k=' + str(k) + '.png', bbox_inches='tight', dpi=600)


    def get_rgba_pyplot(self, df_results):
        """
        Input: Results df including r, g, b, and alpha values
        Output: Array of colors (rgb + alpha) for scatterplotting
        """
        color_array = []
        for i, row in df_results.iterrows():
            r, g, b, a = row['r'], row['g'], row['b'], row['alpha']
            color_array.append((r, g, b, a))

        return np.array(color_array)


def main():
    commute_area = CommuteArea('Southern California')

    # Load data
    commute_area.load_data()

    # Merge data
    commute_area.merge_data()

    # Filter to region
    commute_area.limit_area()

    # Calculate distance matrix
    commute_area.create_distance_matrix()

    # Calculate linkage
    commute_area.calculate_linkage()

    # Plot silhoutte score
    commute_area.plot_silhouette_score()

    # Plot graph for best value of k
    # commute_area.plot_map(commute_area.best_k)

    # Alternativly, plot graphs for all values of k in a given range
    for k in range(2, 21):
        commute_area.plot_map(k)

if __name__ == '__main__':
    main()