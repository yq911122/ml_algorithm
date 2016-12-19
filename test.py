import numpy as np
from sklearn.datasets import load_digits, load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale

import importHelper
cv = importHelper.load('cvMachine')

import prep

def classification_test(clf, classes):
	X, y = get_classification_data(classes)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	# print y_pred
	# print y_test
	print sum([1 for i in range(y_test.shape[0]) if y_pred[i] == y_test[i]]) / float(y_test.shape[0])
	print cv.confusion_matrix_from_cv(clf, X, y, cv=5)


def regression_test(clf):
	X, y = get_regression_data()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)

	clf.fit(X_train, y_train)
	print np.sum((clf.predict(X_test)-y_test)**2)


def get_classification_data(classes):
	digits = load_digits(classes)
	X = scale(digits.data, with_mean=True, with_std=True)
	le = prep.encode(digits.target)
	y = le.transform(digits.target)

	return X, y


def get_regression_data():
	boston = load_boston()
	X, y = scale(boston.data, with_mean=True, with_std=True), scale(boston.target, with_mean=True, with_std=True)

	return X, y