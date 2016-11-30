from hist_feature_test import *
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier


# creating the classifier
#clf = svm.LinearSVC()

mlb = MultiLabelBinarizer()

y_train_lb = mlb.fit_transform(y_train)

clf = OneVsRestClassifier(SVC(probability=True))


# trainning the classifier with 9000 examples
print X_train[0]
print y_train[0]

clf.fit(X_train, y_train_lb)

# verifying the accuracy for the model
predicted = clf.predict(X_test)
print accuracy_score(predicted, y_test)
