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

from scipy.stats import f_oneway, kruskal, chi2_contingency, ttest_ind
import scikit_posthocs as sp

import seaborn as sns

import matplotlib.patheffects as path_effects

import shutil

from statistics import mean as st_mean

from scipy.stats import chi2_contingency

from test_chi2 import run_chi2

p_value_threshold = 0.05


def calculate_statistics_for_clusters(df, entry, list_stats: list, list_stats_test: list,
                                      path_components: list = ['classification'],
                                      pivot_column: str = None, threshold: float = None,
                                      target_column: str = None, list_of_columns: list = None):
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
                                                                                         pivot_column=pivot_column,
                                                                                         pivot_value=1,
                                                                                         threshold=threshold,
                                                                                         list_stats=arr_list_stats,
                                                                                         list_stats_test=arr_list_stats_test,
                                                                                         target_column=target_column,
                                                                                         path_components=[
                                                                                             'classification',
                                                                                             'highest-score-cluster'],
                                                                                         list_of_columns=list_of_columns
                                                                                         )

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


def calculate_statistics_on_clusters_by_target(df, entry, pivot_column, pivot_value, threshold,
                                               list_stats: list = [],
                                               list_stats_test: list = [],
                                               target_column: str = None,
                                               path_components: list = None,
                                               list_of_columns: list = None):
    cluster_labels = entry['opt_labels']
    num_clusters = entry['opt_k']
    reducer_display_name = entry['reducer_display_name']
    clustering_display_name = entry['clustering_display_name']

    df['cluster'] = cluster_labels

    table = pd.crosstab(df['cluster'], df[target_column])

    chi2, p, dof, expected = chi2_contingency(table)

    df = df.drop(columns=['cluster'])

    if num_clusters > 0 and pivot_column in df:
        arr = df.get(pivot_column).tolist()

        list_clusters = [[] for i in range(num_clusters)]

        for i in range(len(arr)):
            label = cluster_labels[i]

            if label < len(list_clusters):
                cluster = list_clusters[label]
                cluster.append(arr[i])

        list_by_pivot = None

        clusters_of_interest: list = None

        for i in range(len(list_clusters)):
            cluster = list_clusters[i]

            mn = st_mean(cluster)

            if not isinstance(list_by_pivot, list) or len(list_by_pivot) < 2:
                list_by_pivot = [[], []]

            index = 0

            if mn > threshold:
                index = 1

            list_by_pivot[index].append(i)

        arr = df.get(target_column).tolist()

        list_cluster_indices: list = None

        for i in range(len(list_by_pivot)):
            if i == pivot_value:
                list_cluster_indices = list_by_pivot[i]

        if len(list_cluster_indices) > 1:
            dict_clusters: dist = None

            for i in range(len(arr)):
                label = cluster_labels[i]

                if label in list_cluster_indices:
                    if dict_clusters is None:
                        dict_clusters = {}

                    if label not in dict_clusters:
                        dict_clusters[label] = []

                    cluster = dict_clusters[label]

                    cluster.append(arr[i])
            _ = 0

            list0: list = list(dict_clusters.values())

            if len(list0) > 1:
                max_different_values: int = 0

                for list1 in list0:
                    list1 = list(set(list1))

                    len_list1: int = len(list1)

                    if len_list1 > max_different_values:
                        max_different_values = len_list1

                can_check_columns: bool = False

                if max_different_values == 1:
                    can_check_columns = True

                if not can_check_columns:
                    chi2, p, dof, expected = run_chi2(list0)

                    if p < 0.05:
                        can_check_columns = True

                if can_check_columns:
                    list_stat_test0: list = None

                    len_columns: int = 0

                    if isinstance(list_of_columns, list):
                        len_columns = len(list_of_columns)

                    if len_columns > 0:
                        for tup in list_of_columns:
                            if isinstance(tup, tuple) and len(tup) > 0:
                                col = tup[0]

                                if col:
                                    dict_clusters: dict = None

                                    arr = df.get(col).tolist()

                                    len_arr: int = len(arr)

                                    if len_arr > 0:
                                        for i in range(len_arr):
                                            label = cluster_labels[i]

                                            if label in list_cluster_indices:
                                                if dict_clusters is None:
                                                    dict_clusters = {}

                                                if label not in dict_clusters:
                                                    dict_clusters[label] = []

                                                cluster = dict_clusters[label]

                                                cluster.append(arr[i])

                                        list0: list = list(dict_clusters.values())

                                        if len(list0) > 1:
                                            if len(tup) > 1 and tup[1] == 1:  # categorical
                                                chi2, p, dof, expected = run_chi2(list0)
                                            else:
                                                if len(list0) == 2:
                                                    t_stat, p = ttest_ind(list0[0], list0[1])
                                                else:
                                                    f_stat, p = f_oneway(*list0)

                                            if list_stat_test0 is None:
                                                list_stat_test0 = []

                                            list_stat_test0.append({
                                                'column': col,
                                                'p': p
                                            })

                    len_stat_test0: int = 0

                    if isinstance(list_stat_test0, list):
                        len_stat_test0 = len(list_stat_test0)

                    if len_stat_test0 > 0:
                        class_folder = 'output'

                        for component in path_components:
                            if component is not None:
                                class_folder = os.path.join(class_folder, component)

                                if not os.path.exists(class_folder):
                                    os.makedirs(class_folder)

                        file_name = f'{reducer_display_name}-{clustering_display_name}'

                        file_name_stat_test = f'{file_name}-columns-stat-test.txt'

                        class_file_name_stat_test = os.path.join(class_folder, file_name_stat_test)

                        with open(class_file_name_stat_test, 'w') as fwriter:
                            fwriter.write('column,p\n')

                            for i in range(len_stat_test0):
                                stat_obj = list_stat_test0[i]

                                if isinstance(stat_obj, dict) and len(stat_obj) > 0:
                                    column = stat_obj['column'],
                                    p = stat_obj['p'],

                                    str = f'{column[0]},{p[0]}\n'

                                    fwriter.write(str)

    return list_stats, list_stats_test
