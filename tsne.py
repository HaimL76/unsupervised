import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import replace_extension, csv_to_dict


def plot_tsne(csv_file, target_column=None):
    # Load data
    df = pd.read_csv(csv_file)

    trans_file = replace_extension(csv_file, '.trans')

    if os.path.exists(trans_file):
        column_map: dict = csv_to_dict(trans_file)

        df.rename(columns=column_map, inplace=True)

    encoder = LabelEncoder()

    column_names = df.keys()

    for col_name in column_names:
        col = df[col_name]
        df[col_name] = encoder.fit_transform(col)

    target_column = 'Mood Score'
    target_column = 'Productivity Score'
    #target_column = 'Stress'

    # If target_column is provided, extract labels
    if target_column and target_column in df.columns:
        labels = df[target_column]
        df = df.drop(columns=[target_column])
    else:
        labels = None

    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df_scaled)

    # Plot results
    plt.figure(figsize=(8, 6))
    if labels is not None:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_encoded, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Categories')
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)

    length = len(tsne_results)

    for index in range(length):
        tsne1 = tsne_results[index, 0]
        tsne2 = tsne_results[index, 1]

        ##plt.text(tsne1, tsne2, str(index), fontsize=8, ha='right', va='bottom', color='red')

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
