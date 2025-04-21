import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

from clustering import calculate_clusters, clustering_options
from utils import replace_extension, csv_to_dict, k_means

from scipy.stats import f_oneway, kruskal

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


def calculate(csv_file, target_column=None, drop_target_column: bool = True, columns_to_drop: list = None):
    # Load data
    df = pd.read_csv(csv_file)

    trans_file = replace_extension(csv_file, '.trans')

    if os.path.exists(trans_file):
        column_map: dict = csv_to_dict(trans_file)

        df.rename(columns=column_map, inplace=True)

    if isinstance(columns_to_drop, list) and len(columns_to_drop) > 0:
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

    column_names = df.keys()

    encoder = LabelEncoder()

    for col_name in column_names:
        col = df[col_name]
        df[col_name] = encoder.fit_transform(col)

    # If target_column is provided, extract labels
    if target_column and target_column in df.columns:
        labels = df[target_column]

        if drop_target_column:
            df = df.drop(columns=[target_column])
    else:
        labels = None

    column_names = df.keys()

    if not os.path.exists('output'):
        os.makedirs('output')

    features_file_path = os.path.join('output', 'features.txt')

    with open(features_file_path, 'w') as fwriter:
        fwriter.write('feature,mean,std,min,max\n')

        for col in df.columns:
            arr0 = df[col]
            m = arr0.mean()
            s = arr0.std()
            mn = arr0.min()
            mx = arr0.max()

            fwriter.write(f'{col},{m},{s},{mn},{mx}\n')

    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    opt_cluster_scores: list = []

    len_dimension_reduction_methods = len(dimension_reduction_methods)

    for reducer_index in range(len_dimension_reduction_methods):
        opt_cluster_scores = calculate_dimension_reduction(df_scaled, reducer_index, labels, target_column,
                                                           opt_cluster_scores=opt_cluster_scores)

    list_stats: list = []

    for entry in opt_cluster_scores:
        cluster_labels = entry['opt_labels']
        num_clusters = entry['opt_k']
        reducer_display_name = entry['reducer_display_name']
        clustering_display_name = entry['clustering_display_name']

        if num_clusters > 0:
            for col in df.columns:
                arr = df.get(col).tolist()

                list_clusters = [[] for i in range(num_clusters)]

                for i in range(len(arr)):
                    label = cluster_labels[i]

                    if label < len(list_clusters):
                        cluster = list_clusters[label]
                        cluster.append(arr[i])

                f_stat, p_anova = f_oneway(*list_clusters)

                list_stats.append((reducer_display_name,clustering_display_name,col,f_stat,p_anova))

                    #h_stat, p_kruskal = kruskal(*list_clusters)
            ##anova_results.append((col, f_stat, p_anova))
            ##kruskal_results.append((col, h_stat, p_kruskal))

    if not os.path.exists('output'):
        os.makedirs('output')

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


def calculate_dimension_reduction(df_scaled, reducer_index, labels, target_column=None,
                                  opt_cluster_scores: list = []):
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

    for cluster_index in range(len_clustering_options):
        clustering = clustering_options[cluster_index]

        cluster_display_name = clustering[str_display_name]

        clusters, opt_cluster_scores = calculate_clusters(arr, clustering, k_min=2, k_max=22,
                                                          reducer_display_name=reducer_display_name,
                                                          opt_cluster_scores=opt_cluster_scores)

        save_results_to_image(reducer_display_name, cluster_display_name, num_comps, labels, results, clusters)

    return opt_cluster_scores


def save_results_to_image(reducer_display_name, cluster_display_name, num_comps, labels, results, clusters):
    # Creating figure
    fig = plt.figure(figsize=(10, 7))

    plt.title(f'{reducer_display_name} simple {num_comps}D scatter plot')

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

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

    for cluster in clusters:
        hull_points = cluster[1]

        if isinstance(hull_points, np.ndarray):
            shape = hull_points.shape

            if isinstance(shape, tuple) and len(shape) == 2:  # 3?
                # Plot polygon
                plt.plot(hull_points[:, 0], hull_points[:, 1], 'b-', linewidth=2)  # , label='Polygon')

    if not os.path.exists('output'):
        os.makedirs('output')

    out_file_path = os.path.join('output', f'{reducer_display_name}-{cluster_display_name}.png')

    plt.savefig(out_file_path)


# Example usage:
arr_files: list = [
    r'ds\sleep_deprivation_dataset_detailed.csv',
    r'ds\Bank_Transaction_Fraud_Detection.csv',
    r'ds\sales_data.csv',
    r'ds\sleep_cycle_productivity.csv',
    r'ds\car_price_dataset.csv',
    r'ds\schizophrenia_dataset.csv'
]

file_path: str = arr_files[-1]

calculate(file_path, target_column="Diagnosis", drop_target_column=False, columns_to_drop=['Patient_ID'])
