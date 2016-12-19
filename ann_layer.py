import numpy as np

sigmod = lambda x: 1 / (1 + np.exp(-x))

class layer(object):
	"""docstring for layer"""
	def __init__(self, nodes, up=None, down=None, bias=False):
		super(layer, self).__init__()
		if bias: 
			self.o = np.zeros((nodes+1, ))
			self.o[0] = 1
		else: self.o = np.zeros((nodes, ))
		self.N = nodes
		self.down, self.up = down, up
		self.bias = bias

	def size(self):
		return self.N

	def output(self):
		return self.o

class inputLayer(layer):
	"""docstring for inputLayer"""
	def __init__(self, nodes, down=None, bias=False):
		super(inputLayer, self).__init__(nodes, None, down)

	def set_output(self, vals):
		if self.bias: self.o[1:] = vals
		else: self.o = vals

class otherLayer(layer):
	"""docstring for otherLayer"""
	def __init__(self, nodes, up, down, bias=False):
		super(otherLayer, self).__init__(nodes, up, down)
		self.w = np.random.rand(nodes, up.size())
		self.d = np.empty_like(self.o)

	def coef(self):
		return self.w

	def cal_output(self):
		# if inputLayer: 
		# 	# if preVals.shape != self.o.shape:
		# 	# 	print "inconsistent nodes"
		# 	# 	return
		# 	self.o = preVals
		if self.bias: self.o[1:] = sigmod(self.w.dot(self.up.output()))
		else: self.o = sigmod(self.w.dot(self.up.output()))

	def update_w(self, y, a):
		o, down, up = self.o, self.down, self.up
		if down: self.d = o * (1 - o) * np.dot(down.d, down.coef())
		else: 
			self.d = o * (1 - o) * (y - o)
			self.d = self.d.reshape((1, self.d.shape[0]))
		o2 = up.output()
		o2 = o2.reshape((1, o2.shape[0]))
		self.w += a * np.dot(self.d.T, o2)