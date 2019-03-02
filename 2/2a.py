import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report


def run_kmeans(data_set, test_set):
    k3 = KMeans(n_clusters=3).fit(data_set.values)
    k5 = KMeans(n_clusters=5).fit(data_set)
    k10 = KMeans(n_clusters=10).fit(data_set)

    print(f'---[ k{k3.n_clusters} ]---')
    print(f'cluster centers:\n{k3.cluster_centers_}')

    print(f'---[ k{k5.n_clusters} ]---')
    print(f'cluster centers:\n{k5.cluster_centers_}')

    print(f'---[ k{k10.n_clusters} ]---')
    print(f'cluster centers:\n{k10.cluster_centers_}')

X = pd.read_csv('p_adult.data')
Y = pd.read_csv('adult.test')

run_kmeans(X, Y)
