import numpy as np
# from scipy.misc import derivative

# def partial_derivative(func, var=0, point=[]):
# 	args = point[:]
# 	def wraps(x):
# 		args[var] = x
# 		return func(*args)
# 	return derivative(wraps, point[var], dx = 1e-6)
	

# class linear_square_cost(object):
# 	"""docstring for linear_square_cost"""
# 	def __init__(self):
# 		super(linear_square_cost, self).__init__()		

# 	def sum(self, w, x, y):
# 		return np.sum((w*x-y)**2) / (2 * y.shape[0])

# 	def part_deri(self, w, x, y):
# 		s = np.sum(w*x-y)
# 		return np.array([xi*s / y.shape[0] for xi in x])


# class log_cost(object):
# 	"""docstring for log_cost"""
# 	sigmod = lambda w, X: 1 / (1+np.exp(-np.sum(w*X, axis=1)))
# 	def __init__(self, arg):
# 		super(log_cost, self).__init__()
	
# 	def sum(self, w, X, Y):
# 		hX = log_cost.sigmod(w, X) 
# 		return -(np.sum(Y*np.log(hX)+(1-Y)*np.log(1-hX))) / Y.shape[0]

# 	def part_deri(self, w, X, Y):
# 		hX = log_cost.sigmod(w, X) 
# 		return np.sum((hX-Y)*X.T, axis=1) / Y.shape[0]