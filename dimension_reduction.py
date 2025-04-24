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
from clustering import calculate_clusters, clustering_options
from utils import replace_extension, csv_to_dict, k_means

from scipy.stats import f_oneway, kruskal
import scikit_posthocs as sp

import seaborn as sns

import matplotlib.patheffects as path_effects

p_value_threshold = 0.05

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


def calculate_statistics_for_clusters(df, entry, list_stats: list, list_stats_test: list,
                                      path_components: list = ['classification']):
    cluster_labels = entry['opt_labels']
    num_clusters = entry['opt_k']
    reducer_display_name = entry['reducer_display_name']
    clustering_display_name = entry['clustering_display_name']

    if num_clusters > 0:
        is_valid_path_components = isinstance(path_components, list) and len(path_components) > 0

        if not is_valid_path_components:
            path_components = ['classification']

        class_folder = 'output'

        for component in path_components:
            if component is not None:
                class_folder = os.path.join(class_folder, component)

                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

        file_name = f'{reducer_display_name}-{clustering_display_name}'

        file_name_heat = f'{file_name}-heatmap.png'

        class_file_heat = os.path.join(class_folder, file_name_heat)

        df['cluster'] = cluster_labels

        cluster_summary = df.groupby('cluster').mean().round(2)
        # print(cluster_summary)

        plt.figure(figsize=(12, 6))
        sns.heatmap(cluster_summary, annot=True, cmap='coolwarm')
        plt.title("Cluster Feature Means")
        plt.savefig(class_file_heat, dpi=300)

        df.drop(columns=['cluster'])

        file_name_csv = f'{file_name}.txt'

        class_file = os.path.join(class_folder, file_name_csv)

        X = df
        y = cluster_labels

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)

        importances = clf.feature_importances_
        feature_names = X.columns

        importance_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        importance_df.to_csv(class_file)

        file_name_image = f'{file_name}.png'

        class_image_file = os.path.join(class_folder, file_name_image)

        plt.figure(figsize=(10, 6))
        importance_df.plot(kind='barh', color='teal')
        plt.title("Feature Importance by Random Forest")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(class_image_file, dpi=300)

        for col in df.columns:
            arr = df.get(col).tolist()

            list_clusters = [[] for i in range(num_clusters)]

            for i in range(len(arr)):
                label = cluster_labels[i]

                if label < len(list_clusters):
                    cluster = list_clusters[label]
                    cluster.append(arr[i])

            f_stat, p_anova = f_oneway(*list_clusters)

            h_stat = None
            p_kruskal = None

            try:
                h_stat, p_kruskal = kruskal(*list_clusters)
            except Exception as e:
                _ = e

            if p_anova < p_value_threshold and p_kruskal < p_value_threshold:
                df0 = pd.DataFrame({
                    'values': arr,
                    'groups': cluster_labels
                })

                dunn_result = sp.posthoc_dunn(df0, val_col='values', group_col='groups', p_adjust='bonferroni')

                list_stats_test.append((reducer_display_name, clustering_display_name, col, p_anova, p_kruskal))

            list_stats.append(
                (reducer_display_name, clustering_display_name, col, f_stat, p_anova, h_stat, p_kruskal))

    return list_stats, list_stats_test


def calculate(csv_file, target_column=None, drop_target_column: bool = True, columns_to_drop: list = None,
              csv_sep=',', k_min=2, k_max=22):
    # Load data
    df = pd.read_csv(csv_file, sep=csv_sep)

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
    most_optimal_cluster = None

    len_dimension_reduction_methods = len(dimension_reduction_methods)

    for reducer_index in range(len_dimension_reduction_methods):
        opt_cluster_scores, most_optimal_cluster = calculate_dimension_reduction(df_scaled, reducer_index,
                                                                               labels, target_column,
                                                                               opt_cluster_scores=opt_cluster_scores,
                                                                               most_optimal_cluster=most_optimal_cluster,
                                                                               k_min=k_min, k_max=k_max)

    if not os.path.exists('output'):
        os.makedirs('output')

    list_stats: list = []
    list_stats_test: list = []

    path_components = ['classification']

    for entry in opt_cluster_scores:
        list_stats, list_stats_test = calculate_statistics_for_clusters(df, entry, list_stats,
                                                                        list_stats_test,
                                                                        path_components=path_components)

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


def calculate_dimension_reduction(df_scaled, reducer_index, labels, target_column=None,
                                  opt_cluster_scores: list = [], most_optimal_cluster=None,
                                  k_min=2, k_max=22):
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

        clusters, opt_cluster_scores = calculate_clusters(arr, clustering, k_min=k_min, k_max=k_max,
                                                          reducer_display_name=reducer_display_name,
                                                          opt_cluster_scores=opt_cluster_scores)

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


# Example usage:
arr_files: list = [
    (r'ds\sleep_deprivation_dataset_detailed.csv',),
    (r'ds\Bank_Transaction_Fraud_Detection.csv',),
    (r'ds\sales_data.csv',),
    (r'ds\sleep_cycle_productivity.csv',),
    (r'ds\car_price_dataset.csv',),
    (r'ds\heart.csv',),
    (r'ds\cardio_train.csv', ['id'], ';'),
    (r'ds\cardio_data_processed.csv', ['id'],),
    (r'ds\alzheimers_disease_data.csv', ['PatientID']),
    (r'ds\health_data.csv', ['id']),
    (r'ds\UserCarData.csv', ['Sales_ID'], ',', (2, 40), 'sold'),
    (r'ds\schizophrenia_dataset.csv', ['Patient_ID'], ',', (2, 5), 'Diagnosis')
]

file_tuple: tuple = arr_files[-1]

file_path: str = None
file_separator: str = None
list_columns_to_drop: list = None
num_k_min = 2
num_k_max = 22
str_target_column: str = None

len_file_tuple = len(file_tuple)

if len_file_tuple > 0:
    file_path = file_tuple[0]

if len_file_tuple > 1:
    list_columns_to_drop = file_tuple[1]

if len_file_tuple > 2:
    file_separator = file_tuple[2]

if len_file_tuple > 3:
    k_tup = file_tuple[3]

    if isinstance(k_tup, tuple) and len(k_tup) == 2:
        num_k_min = k_tup[0]
        num_k_max = k_tup[1]

if len_file_tuple > 4:
    str_target_column = file_tuple[4]

if file_path:
    if file_separator is None:
        file_separator = ','

    calculate(file_path, target_column=str_target_column, drop_target_column=False,
              columns_to_drop=list_columns_to_drop, csv_sep=file_separator,
              k_min=num_k_min, k_max=num_k_max)
