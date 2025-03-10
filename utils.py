import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

from scipy.spatial import ConvexHull
from sklearn.cluster import k_means

from convex_hull import calculate_convex_hull

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

#my_k_means(10)