import numpy as np
from gradDecent import batchGradDecent

sigmod = lambda x: 1 / (1 + np.exp(-x))

class log_cost(object):
	def __init__(self, l=0):
		super(log_cost, self).__init__()
		self.l = 0
	
	def sum(self, w, X, y):
		
		m = y.shape[0]
		if X.shape[1] == w.shape[0] - 1: X = np.concatenate((np.ones((m,1)), X), axis=1)
		hX = sigmod(X.dot(w))
		return -np.sum( np.dot(y.T, np.log(hX)) + np.dot((1 - y).T, np.log(1 - hX)) ) / m + \
				np.sum(self.l * np.sum(w ** 2)) / ( 2 * m )

	def part_deri(self, w, X, y):
		hX = sigmod(X.dot(w))
		m = y.shape[0]
		return ( np.dot(X.T, ( hX - y )) + self.l * w ) / m

	def set_lambda(self, l):
		self.l = l

class logisticRegression(object):
	"""docstring for logisticRegression"""

	costfunc = log_cost()

	def __init__(self, bias=0, fit_intercept =True, regularization=0):
		super(logisticRegression, self).__init__()
		self.fit_intercept = fit_intercept 
		self.multiClass = False
		self.l = regularization
		self.bias = bias
		self._coef = None

	def set_regularization(self, l):
		self.l = l
	
	def fit(self, X, y):
		m, n = X.shape[0], X.shape[1]

		if self.fit_intercept:
			X = np.concatenate((np.ones((m,1)), X), axis=1)
			n += 1
			l = np.zeros((n, ))
			l[1:] = self.l
		else:
			l = np.zeros((n, 1))
			l[:] = self.l

		logisticRegression.costfunc.set_lambda(l)		

		g = batchGradDecent(logisticRegression.costfunc)

		labels = np.unique(y)
		counts = len(labels)

		if counts > 2:
			self.multiClass = True
			self._coef = np.zeros((n, counts))
			for e, i in zip(labels, range(counts)):
				y1 = np.zeros_like(y)
				y1[y==e] = 1
				g.run(X, y1, False)
				self._coef[:,i] = g.get_params()
		else: 
			self.multiClass = False
			g.run(X, y, False)
			self._coef = g.get_params()
		return

	def predict(self, X):
		if self.fit_intercept: X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
		if self.multiClass:
			labels, n = self._coef.shape[1], X.shape[0]
			val = np.zeros((labels, n))
			for i in range(labels):
				val[i] = sigmod(np.dot(X, self._coef[:, i]))
			return np.argmax(val,axis=0)
		else:
			return (sigmod(np.dot(X, self._coef)) > 0.5 + self.bias).astype(int)


def main():
	from test import classification_test
	lr = logisticRegression(bias=0, regularization=0.01)

	classification_test(lr, 5)
	classification_test(lr, 2)

if __name__ == '__main__':
	main()