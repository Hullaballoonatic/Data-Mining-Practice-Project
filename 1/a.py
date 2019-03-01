import graphviz
import pandas as pd
from sklearn import tree

# Grab feature names
with open('1/features.csv') as file:
    features = [feature.strip() for feature in file.readline().split(',')]

# Grab training data
train_df = pd.read_csv('1/training.csv', dtype='category')
train_samples = pd.get_dummies(train_df.loc[:, features[:-1]])
train_labels = pd.get_dummies(train_df.loc[:, features[-1]])

# Create the tree
clf = tree.DecisionTreeClassifier().fit(train_samples, train_labels)

# View the tree
graphviz.Source(tree.export_graphviz(clf, out_file=None)).render(filename='treeclassifier', directory='1', view=True)

# Grab testing data
test_df = pd.read_csv('1/testing.csv', dtype='category')
test_samples = pd.get_dummies(test_df.loc[:, features[:-1]])
test_labels = pd.get_dummies(test_df.loc[:, features[-1]])

# Get predictions from the testing samples

# Compare predicted labels to test labels

# Compute accuracy
