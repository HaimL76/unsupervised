import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import replace_extension, csv_to_dict, k_means, my_k_means


def plot_tsne(csv_file, target_column=None):
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

    # Run dimension reduce
    if True:
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    results = reducer.fit_transform(df_scaled)

    arr = np.asarray(results, dtype=float)

    clusters = my_k_means(arr, k = 12)

    # Plot results
    plt.figure(figsize=(8, 6))
    if labels is not None:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        scatter = plt.scatter(results[:, 0], results[:, 1], c=labels_encoded, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Categories')
    else:
        plt.scatter(results[:, 0], results[:, 1], alpha=0.7)

    length = len(results)

    for index in range(length):
        axis1 = results[index, 0]
        axis2 = results[index, 1]

        ##plt.text(tsne1, tsne2, str(index), fontsize=8, ha='right', va='bottom', color='red')

    for cluster in clusters:
        hull_points = cluster[1]
        # Plot polygon
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'b-', linewidth=2)  # , label='Polygon')
        ##plt.fill(hull_points[:, 0], hull_points[:, 1], color='skyblue', alpha=0.4)  # Optional fill

    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

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

plot_tsne(file_path, target_column="category")
