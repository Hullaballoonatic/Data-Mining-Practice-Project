import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv('data/train')
test = pd.read_csv('data/test').tail(10)

knn3 = KNeighborsClassifier(n_neighbors=3).fit(Y, X)
# knn5 = NearestNeighbors(n_neighbors=5).fit(Y)
# knn10 = NearestNeighbors(n_neighbors=10).fit(Y)

knn3_prediction = knn3.predict(Y)

print(f"accuracy: {accuracy_score(Y, knn3_prediction)}")

print(Y)
