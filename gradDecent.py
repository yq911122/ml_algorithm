import numpy as np
import cost

from sklearn.utils import shuffle

class gradDecent(object):
	"""docstring for gradDecent"""

	def __init__(self, cost_func, a=0.1):
		super(gradDecent, self).__init__()
		self.a = a
		self.J = None
		self.dJ = None
		if cost_func:
			self.J = cost_func.sum
			self.dJ = cost_func.part_deri
		self.w0 = None
		self.w1 = None
		self.i = 0
		self.j = 0

	def set_learn_rate(self, learn_rate):
		self.a = learn_rate

	def set_cost(self, cost_func):
		self.J = cost_func.sum
		self.dJ = cost_func.part_deri

	def init_params(self, X, y):
		self.w0 = np.random.rand(X.shape[1],)
		self.w1 = self.update(self.w0, X, y)

	def update_params(self, X, y):
		self.w0 = self.w1.copy()
		self.w1 -= self.update(self.w1, X, y)
		self.i += 1

	def update(self, w, X, y):
		return w - self.a * self.dJ(self.w1, X, y)

	def stop(self):
		return self.i > 10000 or np.sum((self.w0-self.w1)**2) < 1e-6

	def run(self, X, y, print_cost=False):
		self.init_params(X, y)

		X, y = shuffle(X, y)

		while not self.stop():
			if print_cost: print self.J(self.w1, X, y)
			xj, yj = self.sample(X, y)
			self.update_params(xj, yj)

	def get_params(self):
		return self.w1

	def sample(self, X, y):
		return X, y


class batchGradDecent(gradDecent):
	"""docstring for batchGradDecent"""
	def __init__(self, cost_func, a=0.1):
		super(batchGradDecent, self).__init__(cost_func, a)

	# def sample(self, X, y):
	# 	return X, y

class stochasticGradDecent(gradDecent):
	"""docstring for stochasticGradDecent"""
	def __init__(self, cost_func, a=0.1):
		super(stochasticGradDecent, self).__init__(cost_func, a)
	
	def sample(self, X, y):
		j = self.j
		if j >= y.shape[0]:
			self.j = 0
			X, y = shuffle(X, y)
			return self.sample(X, y)
		self.j += 1
		return X[j,:], y[j]

		
class minibatchGradDecent(gradDecent):
	"""docstring for minibatchGradDecent"""
	def __init__(self, cost_func, a=0.1):
		super(minibatchGradDecent, self).__init__(cost_func, a)
		self.batch_size = size

	def sample(self, X, y):
		b = self.batch_size
		j = self.j
		if b * (j+1) >= y.shape[0]:
			self.j = 0
			X, y = shuffle(X, y)
			return self.sample(X, y, b)
		self.j += 1
		return X[j*b:(j+1)*b,:], y[j*b:(j+1)*b]


def tune_a(algo, params, cost):
	if not algo.J:
		algo.set_cost(cost)
	for a in params:
		algo.set_learn_rate(a)
		algo.run()

	
		