import math

import os.path

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from convex_hull import calculate_convex_hull

from calc_statistics import calculate_statistics_for_clusters

str_method: str = 'method'
str_params: str = 'params'
str_display_name: str = 'display_name'
str_n_clusters: str = 'n_clusters'
str_n_components: str = 'n_components'
str_random_state: str = 'random_state'
str_n_init: str = 'n_init'
str_labels: str = 'labels_'
str_inertia: str = 'inertia_'


def calculate_epsilon(points: np.ndarray, k=2):
    X = points[:, 0]
    Y = points[:, 1]

    max_x = X.max()
    max_y = Y.max()
    min_x = X.min()
    min_y = Y.min()

    len_x = abs(max_x - min_x) / k
    len_y = abs(max_y - min_y) / k

    return math.sqrt(len_x * len_x + len_y * len_y)


k_min_default = 3

clustering_options = [
    {str_method: KMeans, str_display_name: 'KMeans',
     str_params: {str_n_clusters: k_min_default, str_random_state: 0, str_n_init: 10}},
    {str_method: DBSCAN, str_display_name: 'DBSCAN',
     str_params: {'eps': 0.3, 'min_samples': 10}},
    {str_method: GaussianMixture, str_display_name: 'GaussianMixture',
     str_params: {str_n_components: k_min_default}}
]


def calculate_clusters(df_original, points: np.ndarray, clustering, k_min: int = 3, k_max: int = 3,
                       reducer_display_name: str = '', opt_cluster_scores: list = [],
                       list_stats: list = [], list_stats_test: list = []):
    clustering_method = clustering[str_method]
    clustering_params = clustering[str_params]
    clustering_display_name = clustering[str_display_name]

    log_prefix = f'reducer = {reducer_display_name}, cluster = {clustering_display_name}'

    cluster_labels = None

    k_max = k_max + 1

    k_num: int = k_max - k_min

    results = [None] * k_num

    opt_k: int = 0
    highest_score: float = None
    opt_labels = None

    length_results: int = len(results)

    for index in range(length_results):
        k: int = index + k_min

        if str_n_clusters in clustering_params:
            clustering_params[str_n_clusters] = k

        if str_n_components in clustering_params:
            clustering_params[str_n_components] = k

        if 'eps' in clustering_params:
            epsilon = calculate_epsilon(points, k)

            clustering_params['eps'] = epsilon

        clustering_object = clustering_method(**clustering_params)

        clustering_object.fit(points)

        if hasattr(clustering_object, str_labels):
            cluster_labels = clustering_object.labels_
        else:
            cluster_labels = clustering_object.predict(points)

        distinct_clusters = set(cluster_labels)

        len_cluster_labels = len(distinct_clusters)

        if len_cluster_labels > 1:
            sil_score = silhouette_score(points, cluster_labels)

            print(f'{log_prefix}, num clusters = {len_cluster_labels}, silhouette score = {sil_score}')

            if highest_score is None or highest_score < sil_score:
                highest_score = sil_score
                opt_k = len_cluster_labels
                opt_labels = cluster_labels

        if hasattr(clustering_object, str_inertia):
            results[index] = cluster_labels, clustering_object.inertia_

    print(f'{log_prefix}, opt k = {opt_k}, highest score = {highest_score}')

    if opt_cluster_scores is not None:
        opt_cluster_score = {
            'reducer_display_name': reducer_display_name,
            'clustering_display_name': clustering_display_name,
            'opt_k': opt_k,
            'highest_score': highest_score,
            'opt_labels': opt_labels
        }

        opt_cluster_scores.append(opt_cluster_score)

        if not os.path.exists('output'):
            os.makedirs('output')

        list_stats: list = []
        list_stats_test: list = []

        path_components = ['classification']

        list_stats, list_stats_test = calculate_statistics_for_clusters(df_original, opt_cluster_score,
                                                                        list_stats, list_stats_test,
                                                                        path_components=path_components)

    list_clusters = []

    if opt_labels is not None and len(opt_labels) > 0:
        cluster_labels = opt_labels

    if cluster_labels is not None and len(cluster_labels) > 0:
        clusters = set(cluster_labels)

        if -1 in clusters:
            clusters.remove(-1)

        if len(clusters) > 0:
            list_clusters = [None] * len(clusters)

            for index in range(len(points)):
                coords = points[index]

                if index < len(cluster_labels):
                    label = cluster_labels[index]

                    tup = list_clusters[label]

                    if not tup:
                        list_clusters[label] = [None, None]

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

    return list_clusters, opt_cluster_scores, list_stats, list_stats_test
