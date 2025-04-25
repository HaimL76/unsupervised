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

from statistics import mean as st_mean

p_value_threshold = 0.05


def calculate_statistics_for_clusters(df, entry, list_stats: list, list_stats_test: list,
                                      path_components: list = ['classification'],
                                      target_column: str = None, threshold: float = None):
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

        df = df.drop(columns=['cluster'])

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

        list_stats, list_stats_test = calculate_statistics_on_all_clusters(df, entry,
                                                                           list_stats=list_stats,
                                                                           list_stats_test=list_stats_test)

        arr_list_stats = []
        arr_list_stats_test = []

        arr_list_stats, arr_list_stats_test = calculate_statistics_on_clusters_by_target(df, entry,
                                                                                         target_column=target_column,
                                                                                         threshold=threshold,
                                                                                         list_stats=arr_list_stats,
                                                                                         list_stats_test=arr_list_stats_test)

    return list_stats, list_stats_test


def calculate_statistics_on_all_clusters(df, entry,
                                         list_stats: list = [],
                                         list_stats_test: list = []):
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


def calculate_statistics_on_clusters_by_target(df, entry, target_column, threshold,
                                               list_stats: list = [],
                                               list_stats_test: list = []):
    cluster_labels = entry['opt_labels']
    num_clusters = entry['opt_k']
    reducer_display_name = entry['reducer_display_name']
    clustering_display_name = entry['clustering_display_name']

    if num_clusters > 0 and target_column in df:
        arr = df.get(target_column).tolist()

        list_clusters = [[] for i in range(num_clusters)]

        for i in range(len(arr)):
            label = cluster_labels[i]

            if label < len(list_clusters):
                cluster = list_clusters[label]
                cluster.append(arr[i])

        for i in range(len(list_clusters)):
            cluster = list_clusters[i]

            mn = st_mean(cluster)

            if not isinstance(list_stats, list) or len(list_stats) < 2:
                list_stats = [[], []]

            index = 0

            if mn > threshold:
                index = 1

            list_stats[index].append(i)

    return list_stats, list_stats_test
