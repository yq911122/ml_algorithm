import numpy as np
import scipy


class node(object):
	"""docstring for node"""
	def __init__(self, feature_idx):
		super(node, self).__init__()
		self.children = None
		self.j = feature_idx
		self.c = None
		self.vals = None

class decisionTree(object):
	"""docstring for decisionTree"""
	def __init__(self, arg):
		super(decisionTree, self).__init__()
		self.arg = arg

	def fit(self, X, y):
		pass

	def predict(self, X):
		pass
		

def information_gain(x, y):
	


def entropy(x):
	px = prob(x, return_counts=False)
	return scipy.stats.entropy(px)

def prob(x, return_labels=True):
	x_unique, counts = np.unique(x, return_counts=True)
	px = counts / x.shape[0]
	if return_labels: return px, x_unique
	return px

def conditional_prob(x, y):
	"""
	calculate p(x|y)
	"""
	

def conditional_entropy(x, y, px=None):
	pass