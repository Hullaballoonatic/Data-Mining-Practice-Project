import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv('data/train')
train_label = train.income_gt50K.map({0: False, 1: True})

test = pd.read_csv('data/test').tail(10)
test_label = test.income_gt50K.map({0: False, 1: True})

knn3 = KNeighborsClassifier(n_neighbors=3).fit(train, train_label)
knn5 = KNeighborsClassifier(n_neighbors=5).fit(train, train_label)
knn10 = KNeighborsClassifier(n_neighbors=10).fit(train, train_label)

knn3_prediction = knn3.predict(test)
knn5_prediction = knn5.predict(test)
knn10_prediction = knn10.predict(test)

print(f'knn3 accuracy: {accuracy_score(test_label.values, knn3_prediction)}')
print(f'knn5 accuracy: {accuracy_score(test_label.values, knn5_prediction)}')
print(f'knn10 accuracy: {accuracy_score(test_label.values, knn10_prediction)}')
