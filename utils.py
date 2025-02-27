import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from scipy.spatial import ConvexHull
from sklearn.cluster import k_means

def my_k_means(k = 3):
    # Generate sample data
    np.random.seed(0)
    X = np.random.rand(100, 2) * 10  # 100 points in 2D space

    # Apply K-Means
      # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X)

    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    list_clusters = [None] * len(centroids)

    for index in range(len(X)):
        coords = X[index]
        label = labels[index]

        tup = list_clusters[label]

        if not tup:
            list_clusters[label] = [None,None]

        tup = list_clusters[label]

        list_coords = tup[0]

        if not list_coords:
            tup[0] = []

        list_coords = tup[0]

        list_coords.append(coords)

    for cluster in list_clusters:
        list_coords = cluster[0]

        # Compute Convex Hull
        hull = ConvexHull(list_coords)

        cluster[1] = hull.points

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
    plt.legend()
    plt.show()

def csv_to_dict(filename):
    d = dict()

    # Open the CSV file
    with open(filename, mode='r', encoding='utf-8') as file:
        # Read the rest of the file line by line
        for line in file:
            l = line.strip().split(',')

            key = l[0]
            d[key] = l[1]

    return d

def replace_extension(file_path: str, new_extension: str) -> str:
    """
    Replaces the extension of the given file path with a new extension.

    :param file_path: Full file path
    :param new_extension: New extension (with or without a leading dot)
    :return: File path with the new extension
    """
    base = os.path.splitext(file_path)[0]  # Get the file path without the extension
    new_extension = new_extension if new_extension.startswith(".") else f".{new_extension}"
    return f"{base}{new_extension}"

my_k_means(10)