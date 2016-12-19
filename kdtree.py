import numpy as np

import scipy.spatial.distance

class node(object):
	"""docstring for node"""
	def __init__(self, val, axis_idx):
		super(node, self).__init__()
		self.val = val
		self.left = None
		self.right = None
		self.father = None
		self.axis_idx = axis_idx
		self.visited = False

	def intersect(self, x, r):
		v, j = self.val, self.axis_idx
		return abs(v[j] - x[j]) <= r

	def __str__(self, level=0):
		ret = "\t"*level+repr(self.val)+", "+repr(self.axis_idx)+"\n"
		if self.left: ret += self.left.__str__(level+1)
		if self.right: ret += self.right.__str__(level+1)
		return ret


class kdTree(object):
	"""docstring for kdTree"""
	def __init__(self, distance='euclidean'):
		super(kdTree, self).__init__()
		self.root = None
		self.n = 0
		self.dist = getattr(scipy.spatial.distance, distance)

	def partition_by_median(self, x):
		M = np.median(x)

		left, middle, right = np.argwhere(x < M).flatten(), np.argwhere(x == M).flatten(), np.argwhere(x > M).flatten()

		if middle is None or middle.size == 0: 
			temp = np.argmin(x[right])
			middle = np.argwhere(x == x[right][temp]).flatten()
			right = np.delete(right, temp)
			# right = right[1:] if right.shape[0] > 1 else np.array([])
		if middle.shape[0] > 1:
			right, middle = np.vstack((middle[1:], right)), middle[0]

		return left, middle, right

	def fit(self, X):	
		self.n = X.shape[1]
		self.root = self._build_tree(X, 0)
 
	def _build_tree(self, X, j):
		if X is None or X.size == 0: return None
		if X.shape[0] == 1: return node(X.flatten(), j)

		left, middle, right = self.partition_by_median(X[:,j])
		curr_node = node(X[middle].flatten(), j)

		curr_node.left = self._build_tree(X[left], (j + 1) % self.n)
		if right.size > 0:
			curr_node.right = self._build_tree(X[right], (j + 1) % self.n)

		if curr_node.left:
			curr_node.left.father = curr_node
		if curr_node.right:
			curr_node.right.father = curr_node
		return curr_node

	def nearest_point(self, x, k):
		if not self.root: return
		nearest_nodes = []
		nearest_distance = np.empty((k,))
		for i in range(k):
			min_dist, min_node = self._search(x, self.root, np.inf, None)
			nearest_nodes.append(min_node)
			nearest_distance[i] = min_dist
			self._reset(self.root)
			for node in nearest_nodes:
				node.visited = True
		nearest_points = [node.val for node in nearest_nodes]
		return nearest_points, nearest_distance

	def _reset(self, node):
		node.visited = False
		if node.left: self._reset(node.left)
		if node.right: self._reset(node.right)

	def _search(self, x, node, min_dist, min_node):
		v, j = node.val, node.axis_idx

		if x[j] < v[j] and node.left:
			min_dist, min_node = self._search(x, node.left, min_dist, min_node)
		elif node.right:
			min_dist, min_node = self._search(x, node.right, min_dist, min_node)

		if not node.visited:
			curr_dist = self.dist(x, v)

			if curr_dist < min_dist: min_dist, min_node = curr_dist, node

			node.visited = True

			father_node = node.father
			if father_node:
				if father_node.intersect(x, min_dist):
					other_node = father_node.left if node == father_node.right else father_node.right
					if other_node and not other_node.visited:
						min_dist, min_node = self._search(x, other_node, min_dist, min_node)

		return min_dist, min_node

	def __str__(self):
		return self.root.__str__(0)


def main():
	X = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
	tree = kdTree()
	tree.fit(X)
	print tree
	print tree.nearest_point(np.array([7,2]), 2)

if __name__ == '__main__':
	main()