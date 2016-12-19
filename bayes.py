import numpy as np


class naiveBayesClassifier(object):
	"""docstring for naiveBayesClassifier"""
	def __init__(self, l):
		super(naiveBayesClassifier, self).__init__()
		self.y_unique = None
		self.x_unique = []
		self.py = None
		self.pxy = []
		self.l = l

	def fit(self, X, y):
		self.y_unique, counts = np.unique(y, return_counts=True)
		self.py = (counts + self.l) / (y.shape[0] + self.l * self.y_unique.shape[0])

		types = self.y_unique.shape[0]
		for x in X.T:
			x_unique, x_counts = np.unique(x, return_counts=True)
			k = x_unique.shape[0]

			pxy = np.zeros((types, k))
			for i in range(types):
				idx = np.argwhere(y == c)
				xi_unique, xi_counts = np.unique(x[idx], return_counts=True)
				
				idx = []
				for x in xi_unique:
					for j in range(k):
						if x_unique[j] == x:
							idx.append(j)
							break

				pxy[i, idx] = (xi_counts + self.l) / (counts[i] + self.l * x_unique.shape[0])

			self.x_unique.append(x_unique)
			self.pxy.append(pxy)

	def predict(self, X):
		y_pred = np.empty((X.shape[0], ))
		for i in range(X.shape[0]):
			x = X[i]
			py = np.zeros_like(self.py)
			for j in range(py.shape[0]):
				py[j] = self.py[j]
				for k in range(len(self.pxy)):
					pxy = self.pxy[k]
					x_unique = self.x_unique[k]
					xk_idx = x_unique.argwhere(x_unique == X[k])[0][0]
					py[j] *= pxy[j, xk_idx]
			y_idx = np.argmax(py)
			y_pred[i] = self.y_unique[i]
		return y_pred
