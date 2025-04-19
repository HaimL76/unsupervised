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

str_reducer: str = 'reducer'
str_params: str = 'params'
str_n_components = 'n_components'
str_random_state = 'random_state'
str_display_name = 'display_name'

dimension_reduction_methods = [
    { str_reducer:  PCA, str_display_name: 'PCA',
      str_params: {str_n_components: 0, str_random_state: 42 }},
    { str_reducer:  TSNE, str_display_name: 't-SNE',
      str_params: {str_n_components: 0, str_random_state: 42 }},
    { str_reducer:  umap.UMAP, str_display_name: 'UMAP',
      str_params: {'n_neighbors': 15, 'min_dist': 0.1, str_n_components: 0, str_random_state: 42}}
]

def calculate(csv_file, target_column=None):
    # Load data
    df = pd.read_csv(csv_file)

    trans_file = replace_extension(csv_file, '.trans')

    if os.path.exists(trans_file):
        column_map: dict = csv_to_dict(trans_file)

        df.rename(columns=column_map, inplace=True)

    target_column = 'Diagnosis'

    encoder = LabelEncoder()

    column_names = df.keys()

    for col_name in column_names:
        col = df[col_name]
        df[col_name] = encoder.fit_transform(col)

    # If target_column is provided, extract labels
    if target_column and target_column in df.columns:
        labels = df[target_column]
        df = df.drop(columns=[target_column])
    else:
        labels = None

    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    for reducer_index in range(len(dimension_reduction_methods)):
        print(f'reducer index = {reducer_index}')

        calculate_dimension_reduction(df_scaled, reducer_index, labels, target_column)

def calculate_dimension_reduction(df_scaled, reducer_index, labels, target_column=None):
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

    for cluster_index in range(len(clustering_options)):
        print(f'reducer index = {reducer_index}, cluster index = {cluster_index}')

        clustering = clustering_options[cluster_index]

        clusters = calculate_clusters(arr, clustering, k_min = 3, k_max=30)

        cluster_display_name = clustering[str_display_name]

        save_results_to_image(reducer_display_name, cluster_display_name, num_comps, labels, results, clusters)

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

    #if labels is not None:
     #   label_encoder = LabelEncoder()
      #  labels_encoded = label_encoder.fit_transform(labels)
       # scatter = plt.scatter(results[:, 0], results[:, 1], c=labels_encoded, cmap='viridis', alpha=0.7)
        #plt.colorbar(scatter, label='Categories')
    #else:
     #   plt.scatter(results[:, 0], results[:, 1], alpha=0.7)

    for cluster in clusters:
        hull_points = cluster[1]

        if isinstance(hull_points, np.ndarray):
            shape = hull_points.shape

            if isinstance(shape, tuple) and len(shape) == 2:#3?
                # Plot polygon
                plt.plot(hull_points[:, 0], hull_points[:, 1], 'b-', linewidth=2)  # , label='Polygon')
                ##plt.fill(hull_points[:, 0], hull_points[:, 1], color='skyblue', alpha=0.4)  # Optional fill

    ##dir_path = os.path.join('ds', 'output')
    ##plt.show()
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

calculate(file_path, target_column="category")
