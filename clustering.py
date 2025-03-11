import math

import numpy as np
from sklearn.cluster import KMeans, DBSCAN

from convex_hull import calculate_convex_hull

str_method: str = 'method'
str_params: str = 'params'
str_display_name: str = 'display_name'
str_n_clusters: str = 'n_clusters'
str_random_state: str = 'random_state'
str_n_init: str = 'n_init'

def calculate_epsilon(points: np.ndarray):
    array_length: int = points.shape[0]
    array_width: int = points.shape[1]

    square_of_radius: float = 0

    for point in points:
        r = 0

        for i in range(array_width):
            c = point[i]

            r += c*c

        if r > square_of_radius:
            square_of_radius = r

    diameter: float = math.sqrt(square_of_radius) * 2

    epsilon = diameter / array_length

    return epsilon

def calculate_clusters(points: np.ndarray, k = 3):
    clustering_options = [
        {str_method: KMeans, str_display_name: 'KMeans',
         str_params: {str_n_clusters: k, str_random_state: 0, str_n_init: 10}},
        {str_method: DBSCAN, str_display_name: 'DBSCAN',
         str_params: {'eps': 0.3, 'min_samples': 10}}
    ]

    calculate_epsilon(points)

    clustering = clustering_options[0]

    clustering_method = clustering[str_method]
    clustering_params = clustering[str_params]
    clustering_display_name = clustering[str_display_name]

    clustering_object = clustering_method(**clustering_params)

    clustering_object.fit(points)

    # Get cluster labels and centroids
    labels = clustering_object.labels_

    list_clusters = []

    if len(labels) > 0:
        clusters = set(labels)

        if -1 in clusters:
            clusters.remove(-1)
            #centroids = dbscan.cluster_centers_

        if len(clusters) > 0:
            list_clusters = [None] * len(clusters)

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
