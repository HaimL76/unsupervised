import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder


def plot_tsne(csv_file, target_column=None):
    # Load data
    df = pd.read_csv(csv_file)

    encoder = LabelEncoder()

    column_names = df.keys()

    for col_name in column_names:
        col = df[col_name]
        df[col_name] = encoder.fit_transform(col)

    exit(0)

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

    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# Example usage:
filepath: str = r"c:\ds\sleep_deprivation_dataset_detailed.csv"

##filepath = r"c:\ds\Bank_Transaction_Fraud_Detection.csv"
##filepath = r"c:\ds\Bank_Transaction_Fraud_Detection.csv"

plot_tsne(filepath, target_column="category")
