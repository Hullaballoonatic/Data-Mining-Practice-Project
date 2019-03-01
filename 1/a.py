import graphviz
import pandas as pd
from sklearn import tree

with open('1/features.csv') as file:
    features = [feature.strip() for feature in file.readline().split(',')]

train_df = pd.read_csv('1/training.csv', dtype='category')
train_samples = pd.get_dummies(train_df.loc[:, features[:-1]])
train_labels = pd.get_dummies(train_df.loc[:, features[-1]])

test_df = pd.read_csv('1/testing.csv', dtype='category')
test_samples = pd.get_dummies(test_df.loc[:, features[:-1]])
test_labels = pd.get_dummies(test_df.loc[:, features[-1]])

missing = set(train_samples) - set(test_samples)
for attribute in missing:
    test_samples[attribute] = pd.Series(0, dtype='uint8', index=test_samples.index)


clf = tree.DecisionTreeClassifier().fit(train_samples, train_labels)

# View the tree
# graphviz.Source(tree.export_graphviz(clf, out_file=None)).render(filename='treeclassifier', directory='1', view=True)

predictions = clf.predict(test_samples)

comparison = test_labels == predictions

numCorrect = comparison.loc[:, '<=50K'].values.tolist.count(True)

accuracy = float(numCorrect) / float(len(predictions))

print(accuracy)
