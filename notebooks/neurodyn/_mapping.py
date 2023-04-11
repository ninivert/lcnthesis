"""Compute mappings from R² -> R"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from abc import abstractmethod

__all__ = [
	'Box',
	'Mapping', 'BinMapping',
	'RecursiveQuadrantMapping',
	'ReshapeMapping',
	'DiagonalMapping',
	'FarMapping',
	'LinearMapping',
]


@dataclass
class Box:
	xmin: float = 0.0
	xmax: float = 1.0
	ymin: float = 0.0
	ymax: float = 1.0

	@property
	def xmid(self):
		return (self.xmin+self.xmax)/2

	@property
	def ymid(self):
		return (self.ymin+self.ymax)/2

	@property
	def xspan(self):
		return self.xmax - self.xmin

	@property
	def yspan(self):
		return self.ymax - self.ymin

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

	def scale(self, points: np.ndarray) -> np.ndarray:
		"""Given points in [0,1]², scale and translate back to points inside the bbox"""
		return points * np.array([self.xspan, self.yspan]) + np.array([self.xmin, self.xmax])

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


class Mapping:
	"""Neural mapping from R² -> R"""

	@abstractmethod
	def __call__(self, F: np.ndarray, bbox: Box | None) -> np.ndarray:
		"""Apply the mapping of R² -> R
		
		Parameters
		----------
		F : np.ndarray of shape (N, 2)
			2D embedding
		bbox : Box of None
			bounding box to use. If None, will be set to the smallest bounding box of points ``F``
		
		Returns
		-------
		np.ndarray of shape (N,)
			1D embedding
		"""
		pass


class BinMapping(Mapping):
	"""Neural mapping from R² -> discrete bins in [0,1]"""

	def __init__(self, nx: int, ny: int):
		"""
		Parameters
		----------
		nx : int
			number of bins in the `x` coordinate
		ny : int
			number of bins in the `y` coordinate
		"""
		self.nx = nx
		self.ny = ny

	def __call__(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		return self.indices(F, bbox) / self.num_bins

	@abstractmethod
	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		"""Index of the bins corresponding to the mapping"""
		pass

	def inverse(self, alpha: np.ndarray, bbox: Box = Box()) -> np.ndarray:
		"""Inverse mapping from [0,1] -> R²

		Note : this method suffers from floating points errors.
		If indices are available, prefer using ``BinMapping.inverse_indices``.
		
		Parameters
		----------
		alpha : np.ndarray of shape (N,)
			1D embedding
		box : Box
			bounding box for the points in R²

		Returns
		-------
		np.ndarray of shape (N, 2)
			2D embedding
		"""
		return self.inverse_indices((alpha*self.num_bins).astype(int))

	def inverse_indices(self, indices: np.ndarray, bbox: Box = Box()) -> np.ndarray:
		"""Compute point in 2D corresponding to the bins in 1D"""
		return self.inverse_indices2d(self.indices_to_indices2d(indices), bbox)

	def inverse_indices2d(self, indices2d: np.ndarray, bbox: Box = Box()) -> np.ndarray:
		return bbox.scale(indices2d / indices2d.max(axis=0))

		# F = np.zeros((len(coords), 2))
		# for n in range(self.n):
		# 	offsets = np.array([[0.0, 0.0], [1/2**(n+1), 0.0], [1/2**(n+1), 1/2**(n+1)], [0.0, 1/2**(n+1)]])
		# 	F += offsets[coords[:, n]-1]
		# if centered:
		# 	F += np.array([2**(-self.n)/2, 2**(-self.n)/2])  # center the 2D bins
		# if bbox is not None:
		# 	F[:, 0] = (F[:, 0]-0.5)*(bbox.xmax - bbox.xmin)
		# 	F[:, 1] = (F[:, 1]-0.5)*(bbox.ymax - bbox.ymin)
		# return F

	@property
	def num_bins(self) -> int:
		"""Total number of bins"""
		return self.nx * self.ny

	def binned_statistic(self, F: np.ndarray, h: np.ndarray, fill_na: float | None = 0.0) -> np.ndarray:
		"""Compute the mean of `h` inside each bin"""
		m = self.indices(F)
		s = stats.binned_statistic(m, h, bins=range(self.num_bins+1)).statistic
		if fill_na is not None:
			s = np.nan_to_num(s, nan=fill_na)
		return s

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		return np.vstack(np.unravel_index(indices, shape=(self.nx, self.ny))).T

	def indices2d_to_indices(self, indices2d: np.ndarray) -> np.ndarray:
		return np.ravel_multi_index(indices2d.T, dims=(self.nx, self.ny))

	def indices2d(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		"""Compute 2D indices of points ``F`` in R²
		
		Parameters
		----------
		F : np.ndarray of shape (N, 2)
			points in R²
		bbox : Box | None
			bounding box of the points in R². If ``None``, set to be smallest bounding box of ``F``

		Returns
		-------
		np.ndarray of ``int`` of shape (N, 2)
		"""
		if bbox is None:
			bbox = Box.new_bbox(F)

		return (np.vstack([
			# NOTE : we need to clip, because we need equality on the right edge of the last bin : value <= bin[a-1]
			np.clip(np.digitize(F[:, 0], bins=np.linspace(bbox.xmin, bbox.xmax, self.nx+1)), a_min=0, a_max=self.nx),
			np.clip(np.digitize(F[:, 1], bins=np.linspace(bbox.ymin, bbox.ymax, self.ny+1)), a_min=0, a_max=self.ny),
		]) - 1).T


class RecursiveQuadrantMapping(BinMapping):
	def __init__(self, nrec: int):
		"""
		Parameters
		----------
		nrec : int
			number of recursions for the recursive quadrant split
		"""
		super().__init__(2**nrec, 2**nrec)
		self.nrec = nrec

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		if bbox is None:
			bbox = Box.new_bbox(F)
		coords = self.j_coords(F, bbox, self.nrec)
		return (coords-1) @ np.logspace(self.nrec-1, 0, num=self.nrec, base=4, dtype=int)

	def j_coords(self, F: np.ndarray, box: Box | None = None, n: int | None = None) -> np.ndarray:
		"""Generates coordinates (j1, ..., jn) for the recursive quadrant mapping box -> [0,1]
		
		Parameters
		----------
		F : np.ndarray of shape (N, p)
			positions of the neurons in R^p
		n : int or None
			number of recursive iterations. If `None`, is set to ``self.nrec``
		box : Box or None
			bounding box for the mapping. If `None`, is set to the smallest bounding box that contains points ``F``

		Returns
		-------
		np.ndarray of ``int`` of shape (N, n)
			recursive quadrant coordinates for each neuron
		"""

		if n is None:
			n = self.nrec
		
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
		coords[mask1, 1:] = self.j_coords(F[mask1, :], box1, n-1)
		coords[mask2, 1:] = self.j_coords(F[mask2, :], box2, n-1)
		coords[mask3, 1:] = self.j_coords(F[mask3, :], box3, n-1)
		coords[mask4, 1:] = self.j_coords(F[mask4, :], box4, n-1)

		return coords

	def indices_to_j_coords(self, indices: np.ndarray) -> np.ndarray:
		"""Convert indices obtained from ``mapping_index`` to the corresponding quadrant coordinates"""
		C = np.zeros((len(indices), self.n), dtype=int)
		vals = indices.copy()

		for n in reversed(range(0, self.n)):
			C[:, n] = vals % 4
			vals -= C[:, n]
			vals >>= 0x2

		C += 1
		return C

	def values_to_j_coords(self, values: np.ndarray) -> np.ndarray:
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
		return self.indices_to_j_coords((values*self.num_bins).astype(int))

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


class ReshapeMapping(BinMapping):
	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		return self.indices2d_to_indices(self.indices2d(F, bbox))


class DiagonalMapping(BinMapping):
	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		indices2d = self.indices2d(F, bbox).T
		indices_diag = (1/2*(indices2d[0] + indices2d[1])*(indices2d[0]+indices2d[1]+1)+indices2d[0]).astype(int)
		# at this stage the diagonals are clipped, so we get something like
		# 6  11 17 24
		# 3  7  12 18
		# 1  4  8  13
		# 0  2  5  9
		# -> we need to deduplicate and rename
		_, idx_inverse = np.unique(indices_diag, return_inverse=True)
		indices_diag = np.arange(self.nx*self.ny)[idx_inverse]
		return indices_diag


class FarMapping(BinMapping):
	def __init__(self, nrec: int):
		"""
		Parameters
		----------
		nrec : int
			number of recursions for the recursive quadrant split
		"""
		super().__init__(2**nrec, 2**nrec)
		self.nrec = nrec

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		if bbox is None:
			bbox = Box.new_bbox(F)

		def _indices(F: np.ndarray, bbox: Box, n: int, offset: int = 0, stride: int = 1) -> np.ndarray:
			indices = np.zeros(len(F), dtype=int)

			box1, box2, box4, box3 = bbox.split_quadrants()  # NOTE : reordering of the boxes

			mask1 = (box1.xmin <= F[:, 0]) & (F[:, 0] <  box1.xmax) & (box1.ymin <= F[:, 1]) & (F[:, 1] <  box1.ymax)
			mask2 = (box2.xmin <= F[:, 0]) & (F[:, 0] <= box2.xmax) & (box2.ymin <= F[:, 1]) & (F[:, 1] <  box2.ymax)
			mask3 = (box3.xmin <= F[:, 0]) & (F[:, 0] <= box3.xmax) & (box3.ymin <= F[:, 1]) & (F[:, 1] <= box3.ymax)
			mask4 = (box4.xmin <= F[:, 0]) & (F[:, 0] <  box4.xmax) & (box4.ymin <= F[:, 1]) & (F[:, 1] <= box4.ymax)

			if n == 1:
				indices[mask1] = offset+0*stride
				indices[mask2] = offset+1*stride
				indices[mask3] = offset+2*stride
				indices[mask4] = offset+3*stride
				return indices

			indices[mask1] = _indices(F[mask1, :], box1, n-1, offset=offset+0*stride, stride=stride*4)
			indices[mask2] = _indices(F[mask2, :], box2, n-1, offset=offset+1*stride, stride=stride*4)
			indices[mask3] = _indices(F[mask3, :], box3, n-1, offset=offset+2*stride, stride=stride*4)
			indices[mask4] = _indices(F[mask4, :], box4, n-1, offset=offset+3*stride, stride=stride*4)
			return indices

		return _indices(F, bbox, n=self.nrec)


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