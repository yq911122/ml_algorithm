import numpy as np
from numpy.linalg import svd
# from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def encode(x):
	le = LabelEncoder()
	le.fit(x)
	return le


def pca(X, k):
	n = X.shape[0]
	sigma = X.T.dot(X) / n
	u, s, _ = svd(sigma)
	print sum([s[i,i] for i in range(k)]) / sum([s[i,i] for i in range(n)])
	return u[:,:k].T.dot(X), u[:,:k]