import numpy as np
from numpy.random import rand
from scipy.spatial.distance import euclidean

def cost(X, c, l):
	total_cost = 0
	for i in range(len(l)):
		total_cost += euclidean(X[i], c[l[i]])
	return total_cost / float(len(l))

def kmean(k, X, return_cost=True):
	m, n = X.shape[0], X.shape[1]
	c2 = rand(k, n)
	c = c2 - 1
	l = np.zeros((m, ))
	
	min_cost = np.inf
	opt_c = c2
	# d = np.zeros((m, ))
	for j in range(100):
		while np.sum((c2 - c)**2) > 0.001:
			c, c2, d = c2, np.zeros((k, n)), np.zeros((k, ))
			for i in range(m):
				l[i] = np.argmin(np.sum((X[i] - c)**2, axis=1))
			for i in range(m):
				c2[l[i]] += X[i]
				d[l[i]] += 1
			for i in range(k):
				c2[i] /= d[i]
		curr_cost = cost(X, c2, l)
		if curr_cost < min_cost:
			opt_c = c2
			min_cost = curr_cost

	if return_cost:	return c2, min_cost
	return c2

def cluster_number(X):	#elbow
	for k in range(1, 10):
		c, cost = kmean(k, X)
		print cost
