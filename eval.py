import numpy as np
import test
import importHelper
cv = importHelper.load('cvMachine')

import logr, lr

def learn_regularization_term(clf, X, y):
	l = [0]*12
	l[1] = 0.01
	for i in range(2,len(l)):
		l[i] = l[i-1]*2

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)
	for t in l:
		clf.set_regularization(t)
		clf.fit(X_train, y_train)
		print t

		clf.costfunc.set_lambda(0)
		print clf.costfunc.sum(clf._coef, X_train, y_train)
		print clf.costfunc.sum(clf._coef, X_test, y_test)
		print '\n'
			# test.regression_test(clf)


def learning_curve(clf, X, y):
	min_size = 30
	incr_size = 20

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)
	m = X_train.shape[0]
	for i in range(min_size, m, incr_size):
		clf.fit(X_train[:i,:], y_train[:i])
		print i
		clf.costfunc.set_lambda(0)
		print clf.costfunc.sum(clf._coef, X_train[:i,:], y_train[:i])
		print clf.costfunc.sum(clf._coef, X_test, y_test)
		print '\n'

def f1_score(clf, X, y):
	matrix = cv.confusion_matrix_from_cv(clf, X, y, cv=5)
	# f1_scores = np.array([2*precision*recall/(precision+recall) for precision, recall in zip(matrix[-1,:-1], matrix[:-1, -1])])
	# classes = np.sum(matrix[:-1, :-1], axis=0)
	# print matrix
	# print f1_scores
	# return np.sum(f1_scores * classes) / np.sum(classes)
	return matrix[-1,-1]

def main():
	# clf1 = lr.linearRegression(solver='normal')
	# X, y = test.get_regression_data()
	# # learn_regularization_term(clf1, False)
	# learning_curve(clf1, False)

	for b in range(-4, 5):
		clf2 = logr.logisticRegression(bias=0.1*b)
		X, y = test.get_classification_data(2)
		print f1_score(clf2, X, y)
	# learn_regularization_term(clf2, True)
	# learning_curve(clf2, True)
	


if __name__ == '__main__':
	main()