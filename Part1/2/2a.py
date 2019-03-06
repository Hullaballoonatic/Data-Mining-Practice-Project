import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report


train = pd.read_csv('data/train')

k3 = KMeans(n_clusters=3).fit(train)
k5 = KMeans(n_clusters=5).fit(train)
k10 = KMeans(n_clusters=10).fit(train)

print(f'---[ k{k3.n_clusters} ]---')
print(f'cluster centers:\n{k3.cluster_centers_}')

print(f'---[ k{k5.n_clusters} ]---')
print(f'cluster centers:\n{k5.cluster_centers_}')

print(f'---[ k{k10.n_clusters} ]---')
print(f'cluster centers:\n{k10.cluster_centers_}')
