import numpy as np

from numpy.random import rand

# from gradDecent import gradDecent

# class recommendGradDecent(object):
# 	"""docstring for recommendGradDecent"""
# 	def __init__(self, a):
# 		super(recommendGradDecent, self).__init__(cf_cost, a)
	
# 	def init_params(self, X, y):
# 		self.


class cf_cost(object):
	"""docstring for linear_square_cost"""
	def __init__(self, l=0):
		super(linear_square_cost, self).__init__()		
		self.l = 0

	def sum(self, w, X, y):
		m = X.shape[0]
		hX = X.dot(w)
		return ( np.nansum(( hX - y ) ** 2) + np.sum(self.l * (w ** 2 + x ** 2))) / 2

	def part_deri(self, w, X, y):
		hX = X.dot(w)
		hX[np.isnan(hX)] = 0
		dw = ( np.dot(X.T, ( hX - y )) + self.l * w )
		dx = ( np.dot(( hX - y ), w.T) + self.l * x )
		return dw, dx
		# print a
		# return a

	def set_lambda(self, l):
		self.l = l

def recommend(s):
	m, n = s.shape[0], s.shape[1]
	a = 0.1
	l = 0.1
	k = 5

	w = rand(k, n)
	x = rand(m, k)

	cost = cf_cost(l)
	# g = batchGradDecent()
	# g.run(x, s, cost.part_deri, cost.sum)
	i = 0
	while i < 100:
		i += 1
		dw, dx = cost.part_deri(w, x, s)
		w -= a * dw
		x -= a * dx

	hX = x.dot(w)
	for i in range(m):
		for j in range(n):
			if s[i,j] == np.nan: s[i,j] = hX[i,j]

	return s


def main():
	pass

if __name__ == '__main__':
	main()