import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import replace_extension, csv_to_dict, k_means

from scipy.stats import f_oneway, kruskal
import scikit_posthocs as sp

import seaborn as sns

import matplotlib.patheffects as path_effects

import shutil

from clustering import calculate_clusters, clustering_options

from calc_statistics import calculate_statistics_for_clusters

str_reducer: str = 'reducer'
str_params: str = 'params'
str_n_components = 'n_components'
str_random_state = 'random_state'
str_display_name = 'display_name'

dimension_reduction_methods = [
    {str_reducer: PCA, str_display_name: 'PCA',
     str_params: {str_n_components: 0, str_random_state: 42}},
    {str_reducer: TSNE, str_display_name: 't-SNE',
     str_params: {str_n_components: 0, str_random_state: 42}},
    {str_reducer: umap.UMAP, str_display_name: 'UMAP',
     str_params: {'n_neighbors': 15, 'min_dist': 0.1, str_n_components: 0, str_random_state: 42}}
]


def calculate(csv_file, pivot_column=None, target_column: str = None, drop_pivot_column: bool = True,
              columns_to_drop: list = None, csv_sep=',', k_min=2, k_max=22, list_of_columns: list = None):
    if os.path.exists('output'):
        shutil.rmtree('output')

    # Load data
    df_original = pd.read_csv(csv_file, sep=csv_sep)

    trans_file = replace_extension(csv_file, '.trans')

    if os.path.exists(trans_file):
        column_map: dict = csv_to_dict(trans_file)

        df_original.rename(columns=column_map, inplace=True)

    if isinstance(columns_to_drop, list) and len(columns_to_drop) > 0:
        for col in columns_to_drop:
            if col in df_original.columns:
                df_original = df_original.drop(columns=[col])

    column_names = df_original.keys()

    encoder = LabelEncoder()

    for col_name in column_names:
        col = df_original[col_name]
        df_original[col_name] = encoder.fit_transform(col)

    # If pivot_column is provided, extract labels
    if pivot_column and pivot_column in df_original.columns:
        labels = df_original[pivot_column]

        if drop_pivot_column:
            df = df_original.drop(columns=[pivot_column])
    else:
        labels = None

    column_names = df_original.keys()

    if not os.path.exists('output'):
        os.makedirs('output')

    features_file_path = os.path.join('output', 'features.txt')

    with open(features_file_path, 'w') as fwriter:
        fwriter.write('feature,mean,std,min,max\n')

        for col in df_original.columns:
            arr0 = df_original[col]
            m = arr0.mean()
            s = arr0.std()
            mn = arr0.min()
            mx = arr0.max()

            fwriter.write(f'{col},{m},{s},{mn},{mx}\n')

    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_original.select_dtypes(include=[np.number]))

    opt_cluster_scores: list = []
    most_optimal_cluster = None

    len_dimension_reduction_methods = len(dimension_reduction_methods)

    list_stats: list = []
    list_stats_test: list = []

    for reducer_index in range(len_dimension_reduction_methods):
        opt_cluster_scores, most_optimal_cluster = calculate_dimension_reduction(
            df_original, df_scaled, reducer_index, labels, pivot_column, target_column=target_column,
            opt_cluster_scores=opt_cluster_scores, most_optimal_cluster=most_optimal_cluster,
            k_min=k_min, k_max=k_max, list_stats=list_stats, list_stats_test=list_stats_test)

    if most_optimal_cluster is not None:
        path_components = ['classification', 'highest-score-cluster']

        _, _ = calculate_statistics_for_clusters(df_original, most_optimal_cluster, [], [],
                                                 path_components=path_components,
                                                 pivot_column=pivot_column, threshold=0.5,
                                                 target_column=target_column,
                                                 list_of_columns=list_of_columns)

    clusters_file_path = os.path.join('output', 'clusters.txt')

    with open(clusters_file_path, 'w') as fwriter:
        fwriter.write('reducer_display_name, clustering_display_name, opt_k, highest_score\n')

        for entry in opt_cluster_scores:
            reducer_display_name = entry['reducer_display_name']
            clustering_display_name = entry['clustering_display_name']
            opt_k = entry['opt_k']
            highest_score = entry['highest_score']

            str = f'{reducer_display_name},{clustering_display_name},{opt_k},{highest_score}\n'

            fwriter.write(str)

    stats_file_path = os.path.join('output', 'stats.txt')

    if len(list_stats) > 0:
        with open(stats_file_path, 'w') as fwriter:
            fwriter.write('reducer_display_name, clustering_display_name,col,f_stat,p_anova,h_stat,p_kruskal\n')

            for tup in list_stats:
                str_tup = [f'{obj}' for obj in tup]

                str0 = ','.join(str_tup)

                fwriter.write(f'{str0}\n')

    stats_test_file_path = os.path.join('output', 'stats-test.txt')

    if len(list_stats_test) > 0:
        with open(stats_test_file_path, 'w') as fwriter:
            fwriter.write('reducer_display_name,clustering_display_name,col,p_anova,p_kruskal\n')

            for tup in list_stats_test:
                str_tup = [f'{obj}' for obj in tup]

                str0 = ','.join(str_tup)

                fwriter.write(f'{str0}\n')


def calculate_dimension_reduction(
        df_original, df_scaled, reducer_index, labels, pivot_column=None, target_column: str = None,
        opt_cluster_scores: list = [], most_optimal_cluster=None,
        k_min=2, k_max=22, list_stats: list = [], list_stats_test: list = [],
        list_of_columns: list = None):
    num_comps = 2

    # Run dimension reduce
    reducer = dimension_reduction_methods[reducer_index]

    reducer_method = reducer[str_reducer]
    reducer_params = reducer[str_params]
    reducer_display_name = reducer[str_display_name]

    if str_n_components in reducer_params:
        reducer_params[str_n_components] = num_comps

    reducer = reducer_method(**reducer_params)

    results = reducer.fit_transform(df_scaled)

    arr = np.asarray(results, dtype=float)

    len_clustering_options = len(clustering_options)

    list_stats: list = []
    list_stats_test: list = []

    for cluster_index in range(len_clustering_options):
        clustering = clustering_options[cluster_index]

        cluster_display_name = clustering[str_display_name]

        clusters, opt_cluster_scores, list_stats, list_stats_test = calculate_clusters(
            df_original, arr, clustering, k_min=k_min, k_max=k_max,
            reducer_display_name=reducer_display_name,
            opt_cluster_scores=opt_cluster_scores,
            list_stats=list_stats, list_stats_test=list_stats_test,
            pivot_column=pivot_column, target_column=target_column,
            list_of_columns=list_of_columns)

        if isinstance(opt_cluster_scores, list) and len(opt_cluster_scores) > 0:
            new_opt_score = opt_cluster_scores[-1]

            if most_optimal_cluster is None:
                most_optimal_cluster = new_opt_score

            existing_highest_score = most_optimal_cluster['highest_score']

            new_highest_score = new_opt_score['highest_score']

            if new_highest_score is not None:
                if existing_highest_score is None:
                    existing_highest_score = new_highest_score

                if new_highest_score > existing_highest_score:
                    most_optimal_cluster = new_opt_score

        save_results_to_image(reducer_display_name, cluster_display_name, num_comps, labels, results, clusters)

    return opt_cluster_scores, most_optimal_cluster


def save_results_to_image(reducer_display_name, cluster_display_name, num_comps, labels, results, clusters):
    # Creating figure
    fig = plt.figure(figsize=(10, 7))

    plt.title(f'{reducer_display_name} simple {num_comps}D scatter plot')

    label_encoder = LabelEncoder()

    labels_encoded = labels  # default value is the original

    try:
        labels_encoded = label_encoder.fit_transform(labels)
    except Exception as e:
        _ = e

    # Creating plot
    if num_comps == 2:
        plt.scatter(results[:, 0], results[:, 1], c=labels_encoded, alpha=0.7)
    else:
        ax = plt.axes(projection="3d")
        ax.scatter3D(results[:, 0], results[:, 1], results[:, 2], c=labels_encoded, cmap='viridis', alpha=0.7)

    # show plot
    if num_comps == 2:
        plt.xlabel(f'{reducer_display_name} Component 1')
        plt.ylabel(f'{reducer_display_name} Component 2')
    else:
        ax.set_xlabel(f'{reducer_display_name} Component 1')
        ax.set_xlabel(f'{reducer_display_name} Component 2')
        ax.set_xlabel(f'{reducer_display_name} Component 3')

    len_clusters = len(clusters)

    for i in range(len_clusters):
        cluster = clusters[i]

        hull_points = cluster[1]

        if isinstance(hull_points, np.ndarray):
            shape = hull_points.shape

            if isinstance(shape, tuple) and len(shape) == 2:  # 3?
                centroid = np.mean(hull_points, axis=0)

                txt = plt.text(centroid[0], centroid[1], str(i), fontsize=22, color='lightblue',
                               alpha=0.5, ha='center', va='center')

                txt.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal()
                ])

                # Plot polygon
                plt.plot(hull_points[:, 0], hull_points[:, 1], 'b-', linewidth=2)  # , label='Polygon')

    if not os.path.exists('output'):
        os.makedirs('output')

    out_file_path = os.path.join('output', f'{reducer_display_name}-{cluster_display_name}.png')

    plt.savefig(out_file_path)
