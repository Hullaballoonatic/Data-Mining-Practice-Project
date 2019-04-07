from data import train_samples, train_label, test_samples, test_label, train_samples_cat, test_samples_cat
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def tree():
    clf = DecisionTreeClassifier()
    clf.fit(train_samples_cat, train_label)
    print_metrics(clf.predict(test_samples_cat), './out/1a.txt')


def naive_bayes():
    clf = GaussianNB()
    clf.fit(train_samples_cat, train_label)
    print_metrics(clf.predict(test_samples_cat), './out/1b.txt')


def kmeans(num=None):
    if num == None:
        kmeans(3)
        kmeans(5)
        kmeans(10)
    else:
        clf = KMeans(num)
        clf.fit(train_samples)
        print_centroids(clf, f'./out/2a.{num}.txt')


def kNN(num=None):
    if num == None:
        kNN(3)
        kNN(5)
        kNN(10)
    else:
        clf = KNeighborsClassifier(num)
        clf.fit(train_samples, train_label)
        print_accuracy(clf.predict(test_samples), f'./out/2b.{num}.txt')


def svm():
    clf = SVC(gamma='scale')
    clf.fit(train_samples, train_label)
    print_accuracy(clf.predict(test_samples), './out/3.txt')


def nn():
    clf = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500)
    clf.fit(train_samples, train_label)
    print_accuracy(clf.predict(test_samples), './out/4.txt')


def print_accuracy(predictions, outfile):
    accuracy = metrics.accuracy_score(test_label, predictions)
    out_str = f'accuracy: {accuracy:.3f}'
    if outfile:
        with open(outfile, 'w') as out:
            out.write(out_str)
    print(out_str + '\n\n')


def print_centroids(clf, outfile):
    out_str = '\n\n'.join([', '.join([f'{value:.3f}' for value in centroid]) for centroid in clf.cluster_centers_])
    if outfile:
        with open(outfile, 'w') as out:
            out.write(out_str)
    print(out_str + '\n\n')


def print_metrics(predictions, outfile):
    confusion = metrics.confusion_matrix(test_label, predictions)

    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    tp = confusion[1, 1]

    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    recall = tp / float(tp + fn)
    precision = tp / float(tp + fp)
    fp_rate = tn / float(tn + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    out_str = f'Accuracy:  {accuracy:.3f}\nTP/Recall: {recall:.3f}\nFP rate:   {fp_rate:.3f}\nPrecision: {precision:.3f}\nF1 Score:  {f1:.3f}'

    if (outfile):
        with open(outfile, 'w') as results:
            results.write(out_str)
    print(out_str + '\n\n')


problems = {'1a': tree, '1b': naive_bayes, '2a': kmeans, '2b': kNN, '3': svm, '4': nn}

while True:
    choice = str(input("\nEnter a problem number, or 'i' for info, or 'q' to Quit: ")).lower()
    if choice == 'i':
        print('1a: Decision Tree')
        print('1b: Naive-Bayes')
        print('2a: k-means')
        print('2b: k-nearest neighbors')
        print('3: Support Vector Machine')
        print('4: Neural Network')
    elif choice == 'q':
        break
    elif choice not in problems.keys():
        print('not an option.\noptions are ' + problems.keys().join(', '))
    else:
        problems[choice]()
