import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing

from ann_layer import *

class NN(object):
	"""docstring for NN"""
	def __init__(self, a=0.1):
		super(NN, self).__init__()
		self.input_layer = None
		self.output_layer = None
		self.a = a
		self.multiClass = False
		self.lb = None

	def add_layer(self, nodes, bias=False):
		if not self.input_layer:
			self.output_layer = self.input_layer = inputLayer(nodes)
		else:
			up = self.input_layer
			while up.down:
				up = up.down
			curr = otherLayer(nodes, up, None, bias)
			up.down, curr.up = curr, up
			self.output_layer = curr

	def fit(self, X, y):
		if self.input_layer == self.output_layer:
			print "no layer in the network!"
			return
		y_train = y.copy()
		labels = np.unique(y_train)
		counts = len(labels)
			
		if (counts == 2 and self.output_layer.size() != 1) or (counts != 2 and counts != self.output_layer.size()): 
			print 'inconsistent classes number!'
			return
		if counts > 2: 
			self.multiClass = True
			lb = preprocessing.LabelBinarizer()
			y_train = lb.fit_transform(y_train)


		i, j, N = 0, 0, X.shape[0]
		X, y_train = shuffle(X, y_train)
		# and np.sum((y - self.output_layer.output())**2) > 0.1
		while i < 1000:
			self.forward_propagate(X[j,:])
			if i > 2: self.back_propagate(y[j], i)
			else: self.back_propagate(y[j])
			i += 1
			j += 1
			if j >= N: 
				X, y_train = shuffle(X, y_train)
				j = 0

	def forward_propagate(self, x):
		self.input_layer.set_output(x)
		curr = self.input_layer.down
		while curr:
			curr.cal_output()
			curr = curr.down
		return self.output_layer.output()

	def back_propagate(self, y, n=2):
		self.output_layer.update_w(y, self.a)
		curr = self.output_layer.up
		while curr != self.input_layer:
			curr.update_w(y, self.a*(n-1))
			curr = curr.up

	def predict(self, X):
		val = np.empty((X.shape[0], self.output_layer.size()))
		for i in range(X.shape[0]):
			val[i] = self.forward_propagate(X[i])
		if self.multiClass: y_pred = np.argmax(val,axis=1)
		else: y_pred = (val > 0.5).astype(int).flatten()

		return y_pred


def main():
	from test import classification_test
	ann = NN()
	ann.add_layer(64)
	ann.add_layer(20, True)
	ann.add_layer(4)
	classification_test(ann, 4)

if __name__ == '__main__':
	main()