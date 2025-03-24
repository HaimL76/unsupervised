import math

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from convex_hull import calculate_convex_hull

str_method: str = 'method'
str_params: str = 'params'
str_display_name: str = 'display_name'
str_n_clusters: str = 'n_clusters'
str_n_components: str = 'n_components'
str_random_state: str = 'random_state'
str_n_init: str = 'n_init'
str_labels: str = 'labels_'

def calculate_epsilon(points: np.ndarray):
    array_length: int = points.shape[0]
    array_width: int = points.shape[1]

    min_coords = [math.nan] * array_width
    max_coords = [math.nan] * array_width
    len_coords = [math.nan] * array_width

    for point in points:
        r = 0

        for i in range(array_width):
            c = point[i]

            min_coord: float = min_coords[i]

            if min_coord is math.nan or c < min_coord:
                min_coords[i] = c

            max_coord: float = max_coords[i]

            if max_coord is math.nan or c > max_coord:
                max_coords[i] = c

    for i in range(array_width):
        len_coords[i] = math.fabs(max_coords[i] - min_coords[i])

    area: float = 1

    for i in range(array_width):
        area *= len_coords[i]

    point_area: float = area / array_length

    return math.pow(point_area, 1/array_width)

def calculate_clusters(points: np.ndarray, k_min: int = 3, k_max: int = 3):
    clustering_options = [
        {str_method: KMeans, str_display_name: 'KMeans',
         str_params: {str_n_clusters: k_min, str_random_state: 0, str_n_init: 10}},
        {str_method: DBSCAN, str_display_name: 'DBSCAN',
         str_params: {'eps': 0.3, 'min_samples': 10}},
        {str_method: GaussianMixture, str_display_name: 'GaussianMixture',
         str_params: {str_n_components: k_min}}
    ]

    clustering = clustering_options[2]

    clustering_method = clustering[str_method]
    clustering_params = clustering[str_params]
    clustering_display_name = clustering[str_display_name]

    if clustering_method == DBSCAN and 'eps' in clustering_params:
        epsilon = calculate_epsilon(points)

        clustering_params['eps'] = epsilon

    labels = None

    for index in range(k_min, k_max):
        if str_n_clusters in clustering_params:
            clustering_params[str_n_clusters] = index

        clustering_object = clustering_method(**clustering_params)

        clustering_object.fit(points)

        if hasattr(clustering_object, str_labels):
            labels = clustering_object.labels_
        else:
            labels = clustering_object.predict(points)

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
