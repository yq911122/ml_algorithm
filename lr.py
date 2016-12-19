import numpy as np
from gradDecent import *


class linear_square_cost(object):
	"""docstring for linear_square_cost"""
	def __init__(self, l=0):
		super(linear_square_cost, self).__init__()		
		self.l = 0

	def sum(self, w, X, y):
		m = X.shape[0]
		if X.shape[1] == w.shape[0] - 1: X = np.concatenate((np.ones((m,1)), X), axis=1)
		hX = X.dot(w)

		return ( np.sum(( hX - y ) ** 2) + np.sum(self.l * (w ** 2)) ) / ( 2 * m )

	def part_deri(self, w, X, y):
		m = X.shape[0]
		hX = X.dot(w)
		return ( np.dot(X.T, ( hX - y )) + self.l * w ) / m
		# print a
		# return a

	def set_lambda(self, l):
		self.l = l

class linearRegression(object):
	"""docstring for linearRegression"""

	costfunc = linear_square_cost()
	solvers = ['gradDecent', 'normal']
	
	def __init__(self, fit_intercept=True, regularization=0, solver='gradDecent'):
		super(linearRegression, self).__init__()
		self.l = regularization
		self.fit_intercept = fit_intercept
		self.solver = solver
		self._coef = None

	def set_regularization(self, l):
		self.l = l

	def fit(self, X, y):
		m, n = X.shape[0], X.shape[1]
		if self.fit_intercept:
			X = np.concatenate((np.ones((m,1)), X), axis=1)
			n += 1	

		if self.solver == 'gradDecent':
			l = np.zeros((n, ))
			l[:] = self.l
			if self.fit_intercept: l[0] = 0
			linearRegression.costfunc.set_lambda(l)

			g = minibatchGradDecent(linearRegression.costfunc)

			g.run(X, y, print_cost=True)
			self._coef = g.get_params()
			return
		if self.solver == 'normal':
			reg_matrix = np.eye(n)
			if self.fit_intercept:
				reg_matrix[0,0] = 0

			inv = np.linalg.inv(X.T.dot(X) + self.l * reg_matrix)
			self._coef = inv.dot(X.T).dot(y)
			return

	def predict(self, X):
		if self.fit_intercept: X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
		return X.dot(self._coef)


def main():
	from test import regression_test
	lr = linearRegression(solver='normal')
	regression_test(lr)


if __name__ == '__main__':
	main()