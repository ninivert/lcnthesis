"""Compute mappings from R² -> R"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from abc import abstractmethod

__all__ = [
	'Mapping', 'BinMapping',
	'RecursiveQuadrantMapping', 'Box',
	'LinearMapping',
]

class Mapping:
	"""Neural mapping from R² -> R"""

	@abstractmethod
	def __call__(self, F: np.ndarray) -> np.ndarray:
		"""Apply the mapping of R² -> R
		
		Parameters
		----------
		F : np.ndarray of shape (N, 2)
			2D embedding
		
		Returns
		-------
		np.ndarray of shape (N,)
			1D embedding
		"""
		pass


class BinMapping(Mapping):
	"""Neural mapping from R² -> discrete bins in [0,1]"""

	@property
	@abstractmethod
	def num_bins(self) -> int:
		"""Total number of bins"""
		pass

	@abstractmethod
	def inverse(self, alpha: np.ndarray) -> np.ndarray:
		"""Inverse mapping from [0,1] -> R²
		
		Parameters
		----------
		alpha : np.ndarray of shape (N,)
			1D embedding

		Returns
		-------
		np.ndarray of shape (N, 2)
			2D embedding
		"""
		pass

	@abstractmethod
	def indices(self, F: np.ndarray) -> np.ndarray:
		"""Index of the bins corresponding to the mapping"""
		pass

	def binned_statistic(self, F: np.ndarray, h: np.ndarray, fill_na: float | None = 0.0) -> np.ndarray:
		"""Compute the mean of `h` inside each bin"""
		m = self.indices(F)
		s = stats.binned_statistic(m, h, bins=range(self.num_bins+1)).statistic
		if fill_na is not None:
			s = np.nan_to_num(s, nan=fill_na)
		return s


@dataclass
class Box:
	xmin: float
	xmax: float
	ymin: float
	ymax: float

	@property
	def xmid(self):
		return (self.xmin+self.xmax)/2

	@property
	def ymid(self):
		return (self.ymin+self.ymax)/2

	def split_quadrants(self) -> tuple['Box', 'Box', 'Box', 'Box']:
		return (
			Box(self.xmin, self.xmid, self.ymin, self.ymid),  # bottom left
			Box(self.xmid, self.xmax, self.ymin, self.ymid),  # bottom right
			Box(self.xmid, self.xmax, self.ymid, self.ymax),  # top right
			Box(self.xmin, self.xmid, self.ymid, self.ymax),  # top left
		)

	def contains(self, points: np.ndarray) -> np.ndarray:
		"""Compute a mask for all the points contained in the bounding box
		
		Parameters
		----------
		points : np.ndarray of shape (N, 2)
			points to bound
		
		Returns
		-------
		np.ndarray of type ``bool`` of shape (N,)
			boolean mask
		"""
		return (self.xmin <= points[:, 0]) & (points[:, 0] <= self.xmax) & (self.ymin <= points[:, 1]) & (points[:, 1] <= self.ymax)

	@staticmethod
	def new_bbox(points: np.ndarray) -> 'Box':
		"""Construct a ``Box`` that bounds given ``points``
		
		Parameters
		----------
		points : np.ndarray of shape (N, 2)
			points to bound
		
		Returns
		-------
		Box
			bounding box of ``points``
		"""
		return Box(points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max())


class RecursiveQuadrantMapping(BinMapping):
	def __init__(self, n: int = 5):
		"""Recursive quadrant mapping R² -> [0,1]

		See Pham 2018

		Parameters
		----------
		n : int
			number of iterations, by default 5.
		"""
		self.n = n

	@property
	def num_bins(self) -> int:
		return 4**self.n
	
	def __call__(self, F: np.ndarray, box: Box | None = None) -> np.ndarray:
		coords = self.coords(F, n=self.n, box=box)
		return (coords-1) @ np.logspace(1, self.n, num=self.n, base=1/4)

	def inverse(self, coords: np.ndarray) -> np.ndarray:
		"""Given coordinates (generated from ``coords`` or ``index_to_coords``), maps each point back to a bin in [0,1]²"""
		F = np.zeros((len(coords), 2))

		for n in range(self.n):
			offsets = np.array([[0.0, 0.0], [1/2**(n+1), 0.0], [1/2**(n+1), 1/2**(n+1)], [0.0, 1/2**(n+1)]])
			F += offsets[coords[:, n]-1]

		return F

	def indices(self, F: np.ndarray) -> np.ndarray:
		coords = self.coords(F, self.n)
		return (coords-1) @ np.logspace(self.n-1, 0, num=self.n, base=4, dtype=int)

	def indices_to_coords(self, indices: np.ndarray) -> np.ndarray:
		"""Convert indices obtained from ``mapping_index`` to the corresponding quadrant coordinates"""
		C = np.zeros((len(indices), self.n), dtype=int)
		vals = indices.copy()

		for n in reversed(range(0, self.n)):
			C[:, n] = vals % 4
			vals -= C[:, n]
			vals >>= 0x2

		C += 1
		return C

	def values_to_coords(self, values: np.ndarray) -> np.ndarray:
		"""Convert values obtained from ``__call__`` to the corresponding quadrant coordinates.
		When possible, prefer using ``mapping_index``, which avoid floating precision errors.
		
		Example
		-------
		Because of floating point errors, this assertion might fail

		>>> rnn = LowRankRNN.new_valentin(p=2, N=20_000, phi=sigmoid, I_ext=zero, exclude_self_connections=True)
		>>> mapping = RecursiveQuadrantMapping(n=6)
		>>> (mapping.values_to_coords(indices) == mapping.coords()).all()
		False

		Using the indices however works

		>>> (mapping.index_to_coords(indices) == mapping.coords()).all()
		True
		"""
		# cast to index_to_coords algorithm
		return self.indices_to_coords((values*self.num_bins).astype(int))

		# base 4 algorithm
		# C = np.zeros((len(values), self.n), dtype=int)
		# vals = values.copy()
		# 
		# for n in range(0, self.n):
		# 	vals *= 4
		# 	C[:, n] = vals.astype(int)
		# 	vals -= C[:, n]
		# 
		# C += 1
		# return C

	def coords(self, F: np.ndarray, n: int | None = None, box: Box | None = None) -> np.ndarray:
		"""Generates coordinates (j1, ..., jn) for the recursive quadrant mapping box -> [0,1]
		
		Parameters
		----------
		F : np.ndarray of shape (N, p)
			positions of the neurons in R^p
		n : int or None
			number of recursive iterations. If `None`, is set to ``self.n``
		box : Box or None
			bounding box for the mapping. If `None`, is set to the smallest bounding box that contains points ``F``

		Returns
		-------
		np.ndarray of ``int`` of shape (N, n)
			recursive quadrant coordinates for each neuron
		"""

		if n is None:
			n = self.n
		
		coords = np.zeros((len(F), n), dtype=int)

		# stopping condition
		if len(F) == 0 or n == 0:
			return coords

		if box is None:
			box = Box.new_bbox(F)
		else:
			assert box.xmin <= F[:, 0].min(), 'bbox does not contain all neurons'
			assert F[:, 0].max() <= box.xmax, 'bbox does not contain all neurons'
			assert box.ymin <= F[:, 1].min(), 'bbox does not contain all neurons'
			assert F[:, 1].max() <= box.ymax, 'bbox does not contain all neurons'
		
		box1, box2, box3, box4 = box.split_quadrants()

		# generate masks
		# WARNING : we need to be careful with equality on the edges
		# *---------*---------*
		# |         |         |
		# |  box4   |  box3   |
		# |         |         |
		# |---------*---------|
		# |         |         |
		# |  box1   |  box2   |
		# |         |         |
		# *---------*---------*
		mask1 = (box1.xmin <= F[:, 0]) & (F[:, 0] <  box1.xmax) & (box1.ymin <= F[:, 1]) & (F[:, 1] <  box1.ymax)
		mask2 = (box2.xmin <= F[:, 0]) & (F[:, 0] <= box2.xmax) & (box2.ymin <= F[:, 1]) & (F[:, 1] <  box2.ymax)
		mask3 = (box3.xmin <= F[:, 0]) & (F[:, 0] <= box3.xmax) & (box3.ymin <= F[:, 1]) & (F[:, 1] <= box3.ymax)
		mask4 = (box4.xmin <= F[:, 0]) & (F[:, 0] <  box4.xmax) & (box4.ymin <= F[:, 1]) & (F[:, 1] <= box4.ymax)

		# generate coordinates for this layer
		coords[mask1, 0] = 1
		coords[mask2, 0] = 2
		coords[mask3, 0] = 3
		coords[mask4, 0] = 4

		# call for coordinates of lower layers
		coords[mask1, 1:] = self.coords(F[mask1, :], n-1, box1)
		coords[mask2, 1:] = self.coords(F[mask2, :], n-1, box2)
		coords[mask3, 1:] = self.coords(F[mask3, :], n-1, box3)
		coords[mask4, 1:] = self.coords(F[mask4, :], n-1, box4)

		return coords


class LinearMapping(Mapping):
	def __init__(self, tangent: np.ndarray):
		"""Linear projection mapping R² -> R by projecting on the tangent vector

		Parameters
		----------
		tangent : np.ndarray
			tangent vector of shape (2,)
		"""
		self.tangent = tangent.copy()

	def __call__(self, F: np.ndarray) -> np.ndarray:
		return F @ self.tangent