import numpy as np
from sklearn.cluster import KMeans, DBSCAN

str_clustering: str = 'clutering'
str_params: str = 'params'
str_display_name: str = 'display_name'
str_n_clusters: str = 'n_clusters'
str_random_state: str = 'random_state'
str_n_init: str = 'n_init'

def calculate_clusters(points: np.ndarray, k = 3):
    clustering_methods = [
        {str_clustering: KMeans, str_display_name: 'KMeans',
         str_params: {str_n_clusters: k, str_random_state: 0, str_n_init: 10}},
        {str_clustering: DBSCAN, str_display_name: 'DBSCAN',
         str_params: {'eps': 0.3, 'min_samples': 10}}
    ]

    clustering_method = clustering_methods[1]

    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Get cluster labels and centroids
    labels = dbscan.labels_

    if len(labels) > 0:
        centroids = set(labels)

        if -1 in centroids:
            centroids.remove(-1)
            #centroids = dbscan.cluster_centers_

        if len(centroids) > 0:
            list_clusters = [None] * len(centroids)

            for index in range(len(points)):
                coords = points[index]

                if index < len(labels):
                    label = labels[index]

                    tup = list_clusters[label]

                    if not tup:
                        list_clusters[label] = [None,None]

                    tup = list_clusters[label]

                list_coords = tup[0]

                if not list_coords:
                    tup[0] = []

                list_coords = tup[0]

                list_coords.append(coords)

            for cluster in list_clusters:
                list_coords = cluster[0]

                # Compute Convex Hull
                cluster[1] = np.asarray(calculate_convex_hull(list_coords), dtype=float)

    return list_clusters
