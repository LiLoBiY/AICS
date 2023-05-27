import numpy as np
from sklearn.svm import LinearSVC

from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'

data = np.loadtxt(input_file, delimiter=',')

X, y = data[:, :-1], data[:, -1]

classifier = LinearSVC(max_iter=10000)

classifier.fit(X, y)

y_pred = classifier.predict(X)

accurancy = 100.0 * (y == y_pred).sum() / X.shape[0]

print("Accurancy of SVM classifier = ", round(accurancy, 2), "%")

visualize_classifier(classifier, X, y)