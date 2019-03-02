import graphviz
import pandas as pd
from sklearn import tree, metrics, naive_bayes


features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
col_names = features + ['label']


def get_data(filename):
    df = pd.read_csv(filename, header=None, names=col_names, dtype='category')
    samples = pd.get_dummies(df[features])
    label = df.label.map({'<=50K': False, '>50K': True})
    return samples, label


train_samples, train_label = get_data('training.csv')
test_samples, test_label = get_data('testing.csv')

missing = set(train_samples) - set(test_samples)
for attribute in missing:
    test_samples[attribute] = pd.Series(0, dtype='uint8', index=test_samples.index)


def print_metrics(predictions, filename=None):
    confusion = metrics.confusion_matrix(test_label, predictions)

    tp = confusion[1, 1]
    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]

    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    recall = tp / float(tp + fn)
    precision = tp / float(tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    out_string = "Accuracy:  {0:.3f}\nTP/Recall: {1:.3f}\nFP rate:   {2:.3f}\nPrecision: {3:.3f}\nF1 Score:  {4:.3f}".format(
        accuracy,
        recall,
        tn / float(tn + fp),
        precision,
        f1
    )

    if (filename):
        with open(filename, 'w') as results:
            results.write(out_string)
    else:
        print(out_string)


tree = tree.DecisionTreeClassifier().fit(train_samples, train_label)
nb = naive_bayes.GaussianNB().fit(train_samples, train_label)

# View the tree
# graphviz.Source(tree.export_graphviz(tree, out_file=None)).render(filename='a', view=True)

print_metrics(tree.predict(test_samples), 'tree_metrics.txt')
print_metrics(nb.predict(test_samples), 'naivebayes_metrics.txt')
