import graphviz
import pandas as pd
from sklearn import tree, metrics

features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
col_names = features + ['label']

df = pd.read_csv('1/training.csv', header=None, names=col_names, dtype='category')
train_samples = pd.get_dummies(df[features])
train_label = df.label.map({'<=50K': False, '>50K': True})

df = pd.read_csv('1/testing.csv', header=None, names=col_names, dtype='category')
test_samples = pd.get_dummies(df[features])
test_label = df.label.map({'<=50K': False, '>50K': True})

missing = set(train_samples) - set(test_samples)
for attribute in missing:
    test_samples[attribute] = pd.Series(0, dtype='uint8', index=test_samples.index)

clf = tree.DecisionTreeClassifier().fit(train_samples, train_label)

# View the tree
graphviz.Source(tree.export_graphviz(clf, out_file=None)).render(filename='a', directory='1', view=True)

predictions = clf.predict(test_samples)

confusion = metrics.confusion_matrix(test_label, predictions)

tp = confusion[1, 1]
tn = confusion[0, 0]
fp = confusion[0, 1]
fn = confusion[1, 0]

with open('1/a.txt', 'w') as results:
    results.write(
        "Accuracy:  {0:.3f}\nTP/Recall: {1:.3f}\nFP rate:   {2:.3f}\nPrecision: {3:.3f}\nF1 Score:  {4:.3f}"
        .format(
            metrics.accuracy_score(test_label, predictions),
            metrics.recall_score(test_label, predictions),
            fp / float(tn + fp),
            metrics.precision_score(test_label, predictions),
            metrics.f1_score(test_label, predictions)
        )
    )
