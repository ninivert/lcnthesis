"""Compute mappings from R² -> R"""

from typing import Self
import numpy as np
from scipy import stats
from dataclasses import dataclass
from abc import abstractmethod

__all__ = [
	'Box',
	'Mapping', 'BinMapping',
	'RecursiveLocalMapping',
	'ReshapeMapping', 'ColumnMapping',  # alias
	'DiagonalMapping',
	'SzudzikMapping',
	'RecursiveFarMapping', 'AntiZMapping',  # alias
	'ZMapping',
	'RandomMapping',
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
		return points * np.array([self.xspan, self.yspan]) + np.array([self.xmin, self.ymin])

	def normalize(self, points: np.ndarray) -> np.ndarray:
		return (points - np.array([self.xmin, self.ymin])) / np.array([self.xspan, self.yspan])

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

	@staticmethod
	def new_nrec(nrec: int) -> 'BinMapping':
		raise NotImplementedError()

	def __call__(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		return self.indices(F, bbox) / self.num_bins

	@abstractmethod
	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		"""Index of the bins corresponding to the mapping"""
		raise NotImplementedError()

	@abstractmethod
	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		raise NotImplementedError()

	def inverse_from_values(self, alpha: np.ndarray, bbox: Box = Box()) -> np.ndarray:
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
		return self.inverse_from_indices((alpha*self.num_bins).astype(int), bbox)

	def inverse_from_indices(self, indices: np.ndarray, bbox: Box = Box(), centered: bool = True) -> np.ndarray:
		"""Compute point in 2D corresponding to the bins in 1D"""
		return self.inverse_from_indices2d(self.indices_to_indices2d(indices), bbox=bbox, centered=centered)

	def inverse_from_indices2d(self, indices2d: np.ndarray, bbox: Box = Box(), centered: bool = True) -> np.ndarray:
		F = indices2d / np.array([self.nx, self.ny])
		if centered: F += np.array([1/(2*self.nx), 1/(2*self.ny)])  # center the 2D bins
		F = bbox.scale(F)
		return F

	def inverse_samples(self, bbox: Box = Box(), centered: bool = True, use_inverse: bool = True) -> np.ndarray:
		"""Generate samples in 2D, ordered according to the mapping in 1D

		Parameters
		----------
		bbox : Box, optional
			bounding box for the samples in 2D, by default Box()
		centered : bool, optional
			if true, the samples are at the centers of each 2D bin, else on the upper left, by default True
		use_inverse : bool, optional
			use the inverse mapping implementation, by default True.
			if True, the mapping needs to implement ``indices_to_indices2d``.
			if False, a grid of points is generated in 2D, and the mapping is used to order the points.

		Returns
		-------
		np.ndarray of shape (self.num_bins, 2)
			samples in 2D, ordered according to the mapping in 1D
		"""
		if use_inverse:
			return self.inverse_from_indices(np.arange(self.num_bins, dtype='uint64'), bbox=bbox, centered=centered)
		else:
			# generate a grid of points
			xx, yy = np.meshgrid(np.linspace(0, 1-1/self.nx, self.nx), np.linspace(0, 1-1/self.ny, self.ny))
			F = np.vstack((xx.flatten(), yy.flatten())).T
			if centered: F += np.array([1/(2*self.nx), 1/(2*self.ny)])
			F = bbox.scale(F)
			# apply the mapping of 2D to 1D
			F = F[np.argsort(self.indices(F, bbox=bbox))]
			return F

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


class RecursiveLocalMapping(BinMapping):
	"""Implements the mapping [0,1]² -> [0,1] from https://arxiv.org/abs/1602.00800
	
	This mapping converges to a measurable bijection [0,1]² -> [0,1].
	"""

	def __init__(self, nrec: int):
		"""
		Parameters
		----------
		nrec : int
			number of recursions for the recursive quadrant split
		"""
		super().__init__(2**nrec, 2**nrec)
		self.nrec = nrec

	@staticmethod
	def new_nrec(nrec: int) -> 'RecursiveLocalMapping':
		return RecursiveLocalMapping(nrec=nrec)

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		if bbox is None:
			bbox = Box.new_bbox(F)
		coords = self.j_coords(F, bbox, self.nrec)
		return ((coords-1) @ np.logspace(self.nrec-1, 0, num=self.nrec, base=4, dtype=int)).astype('uint64')

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		j_coords = self.indices_to_j_coords(indices)
		indices2d = np.zeros((len(indices), 2), dtype='uint64')

		for n in range(self.nrec):
			offsets = np.array([[0, 0], [2**(self.nrec-n-1), 0], [2**(self.nrec-n-1), 2**(self.nrec-n-1)], [0, 2**(self.nrec-n-1)]], dtype='uint64')
			indices2d += offsets[j_coords[:, n]-1]

		return indices2d

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
		C = np.zeros((len(indices), self.nrec), dtype='uint64')
		vals = indices.copy()

		for n in reversed(range(0, self.nrec)):
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

	def __str__(self) -> str:
		return f'RecursiveLocalMapping{{nrec={self.nrec}}}'


class ReshapeMapping(BinMapping):
	"""Implements mapping column-by-column (reshape operation)
	
	This mapping converges to a projection on x.
	"""

	@staticmethod
	def new_nrec(nrec: int) -> 'ReshapeMapping':
		return ReshapeMapping(nx=2**nrec, ny=2**nrec)

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		return np.ravel_multi_index(self.indices2d(F, bbox).T, dims=(self.nx, self.ny))

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		return np.vstack(np.unravel_index(indices, shape=(self.nx, self.ny))).T

	def __str__(self) -> str:
		return f'ReshapeMapping{{nx={self.nx}, ny={self.ny}}}'

ColumnMapping = ReshapeMapping


class DiagonalMapping(BinMapping):
	"""Implements [Cantor mapping](https://en.wikipedia.org/wiki/Pairing_function)
	
	This mapping converges to a projection on a line with normal (1, 1).
	"""

	def __init__(self, nxy: int):
		super().__init__(nxy, nxy)
		assert self.nx == self.ny
		self.nxy = nxy

	@staticmethod
	def new_nrec(nrec: int) -> 'DiagonalMapping':
		return DiagonalMapping(nxy=2**nrec)

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		indices2d = self.indices2d(F, bbox)
		indices = np.zeros(len(indices2d), dtype=int)
		mask_lower = (indices2d[:, 0] + indices2d[:, 1] < self.nxy)
		indices[mask_lower] = (1/2*(indices2d[mask_lower, 0] + indices2d[mask_lower, 1])*(indices2d[mask_lower, 0]+indices2d[mask_lower, 1]+1)+indices2d[mask_lower, 0]).astype(int)
		indices[~mask_lower] = self.num_bins - (1/2*((self.nx-indices2d[~mask_lower, 0]-1) + (self.ny-indices2d[~mask_lower, 1]-1)) * ((self.nx-indices2d[~mask_lower, 0]-1) + (self.ny-indices2d[~mask_lower, 1]-1) + 1) + (self.nx-indices2d[~mask_lower, 0])).astype(int)
		return indices

		# old method
		# indices2d = self.indices2d(F, bbox).T
		# indices_diag = (1/2*(indices2d[0] + indices2d[1])*(indices2d[0]+indices2d[1]+1)+indices2d[0]).astype(int)
		# at this stage the diagonals are clipped, so we get something like
		# 6  11 17 24
		# 3  7  12 18
		# 1  4  8  13
		# 0  2  5  9
		# -> we need to deduplicate and rename
		# # NOTE : this doesn't work when not all the points are sampled
		# _, idx_inverse = np.unique(indices_diag, return_inverse=True)
		# indices_diag = np.arange(self.nx*self.ny)[idx_inverse]
		# return indices_diag

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		# https://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
		def inv(indices):
			w = np.floor((np.sqrt(8*indices+1)-1)/2)
			t = (w**2 + w)/2
			y = indices - t
			indices2d = np.zeros((len(indices), 2), dtype='uint64')
			indices2d[:, 0] = y
			indices2d[:, 1] = w - y
			return indices2d
		mask_lower = indices < self.nxy*(self.nxy+1)//2
		indices2d = np.zeros((len(indices), 2), dtype='uint64')
		indices2d[mask_lower] = inv(indices[mask_lower])
		indices2d[~mask_lower] = np.array([self.nx-1, self.ny-1])[None, :] - inv(self.num_bins - indices[~mask_lower] - 1)
		return indices2d

	def __str__(self) -> str:
		return f'DiagonalMapping{{nx={self.nx}, ny={self.ny}}}'


class SzudzikMapping(BinMapping):
	"""Implements [Szudzik's "Elegant pairing function"](http://szudzik.com/ElegantPairing.pdf).
	
	This mapping converges to a projection on x if x >= y else a projection on y.
	"""

	def __init__(self, nxy: int):
		super().__init__(nxy, nxy)
		assert self.nx == self.ny
		self.nxy = nxy

	@staticmethod
	def new_nrec(nrec: int) -> 'SzudzikMapping':
		return SzudzikMapping(nxy=2**nrec)

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		indices2d = self.indices2d(F, bbox)
		return np.where(
			indices2d[:, 0] != indices2d.max(axis=1),
			indices2d[:, 0] + indices2d[:, 1]**2,
			indices2d[:, 0]**2 + indices2d[:, 0] + indices2d[:, 1]
		)

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		sqrt = np.sqrt(indices)
		fsqrt = np.floor(sqrt)
		return np.where(
			(indices - fsqrt**2 < fsqrt)[:, None],
			np.vstack((indices - fsqrt**2, fsqrt)).T,
			np.vstack((fsqrt, indices - fsqrt**2 - fsqrt)).T,
		)


class ZMapping(BinMapping):
	"""Implements [Z-order curve](https://en.wikipedia.org/wiki/Z-order_curve)"""

	def __init__(self, nrec: int):
		"""
		Parameters
		----------
		nrec : int
			number of recursions for the recursive quadrant split
		"""
		super().__init__(2**nrec, 2**nrec)
		self.nrec = nrec

	@staticmethod
	def new_nrec(nrec: int) -> 'ZMapping':
		return ZMapping(nrec=nrec)

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		indices2d = self.indices2d(F, bbox)
		return ZMapping.part1by1_64(indices2d[:, 0]) | (ZMapping.part1by1_64(indices2d[:, 1]) << 1)

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		indices2d = np.zeros((len(indices), 2), dtype='uint64')
		indices2d[:, 0] = ZMapping.unpart1by1_64(indices)
		indices2d[:, 1] = ZMapping.unpart1by1_64(indices >> 1)
		return indices2d

	@staticmethod
	def unpart1by1_64(n: int) -> int:
		# https://github.com/smatsumt/pyzorder/blob/master/pyzorder/pymorton.py#L72
		n = n & 0x5555555555555555                # binary: 101010101010101010101010101010101010101010101010101010101010101, len: 63
		n = (n ^ (n >> 1))  & 0x3333333333333333  # binary: 11001100110011001100110011001100110011001100110011001100110011,  len: 62
		n = (n ^ (n >> 2))  & 0x0f0f0f0f0f0f0f0f  # binary: 111100001111000011110000111100001111000011110000111100001111,    len: 60
		n = (n ^ (n >> 4))  & 0x00ff00ff00ff00ff  # binary: 11111111000000001111111100000000111111110000000011111111,        len: 56
		n = (n ^ (n >> 8))  & 0x0000ffff0000ffff  # binary: 1111111111111111000000001111111111111111,                        len: 40
		n = (n ^ (n >> 16)) & 0x00000000ffffffff  # binary: 11111111111111111111111111111111,                                len: 32
		return n

	@staticmethod
	def part1by1_64(n: int) -> int:
		# https://github.com/smatsumt/pyzorder/blob/master/pyzorder/pymorton.py#L50
		n = n & 0x00000000ffffffff                # binary: 11111111111111111111111111111111,                                len: 32
		n = (n | (n << 16)) & 0x0000FFFF0000FFFF  # binary: 1111111111111111000000001111111111111111,                        len: 40
		n = (n | (n << 8))  & 0x00FF00FF00FF00FF  # binary: 11111111000000001111111100000000111111110000000011111111,        len: 56
		n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F  # binary: 111100001111000011110000111100001111000011110000111100001111,    len: 60
		n = (n | (n << 2))  & 0x3333333333333333  # binary: 11001100110011001100110011001100110011001100110011001100110011,  len: 62
		n = (n | (n << 1))  & 0x5555555555555555  # binary: 101010101010101010101010101010101010101010101010101010101010101, len: 63
		return n

	def __str__(self) -> str:
		return f'ZMapping{{nrec={self.nrec}}}'


class RecursiveFarMapping(BinMapping):
	def __init__(self, nrec: int):
		"""
		Parameters
		----------
		nrec : int
			number of recursions for the recursive quadrant split
		"""
		super().__init__(2**nrec, 2**nrec)
		self.nrec = nrec

	@staticmethod
	def new_nrec(nrec: int) -> 'RecursiveFarMapping':
		return RecursiveFarMapping(nrec=nrec)

	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		if bbox is None:
			bbox = Box.new_bbox(F)

		def _indices(F: np.ndarray, bbox: Box, n: int, offset: int = 0, stride: int = 1) -> np.ndarray:
			indices = np.zeros(len(F), dtype=int)

			box1, box3, box4, box2 = bbox.split_quadrants()  # NOTE : reordering of the boxes

			mask1 = (box1.xmin <= F[:, 0]) & (F[:, 0] <  box1.xmax) & (box1.ymin <= F[:, 1]) & (F[:, 1] <  box1.ymax)
			mask2 = (box2.xmin <= F[:, 0]) & (F[:, 0] < box2.xmax) & (box2.ymin <= F[:, 1]) & (F[:, 1] <=  box2.ymax)
			mask3 = (box3.xmin < F[:, 0]) & (F[:, 0] <= box3.xmax) & (box3.ymin <= F[:, 1]) & (F[:, 1] < box3.ymax)
			mask4 = (box4.xmin < F[:, 0]) & (F[:, 0] <=  box4.xmax) & (box4.ymin <= F[:, 1]) & (F[:, 1] <= box4.ymax)

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

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		# TODO
		raise NotImplementedError()

	def __str__(self) -> str:
		return f'RecursiveFarMapping{{nrec={self.nrec}}}'

AntiZMapping = RecursiveFarMapping


class RandomMapping(BinMapping):
	"""Randomly maps 2D bins to 1D bins
	
	Since 2D bins have some implicit ordering (given by a reshape), we do

	```txt
	   ravel          permutation
	2D ----------> 1D ------------+
	↑                             ↓
	+------------- 1D <---------- 1D
	       unravel        argsort
	```
	"""

	def __init__(self, nx: int, ny: int, random_seed: int = 42):
		super().__init__(nx, ny)
		self.random_seed = random_seed
		self.permutation = np.random.default_rng(random_seed).permutation(np.arange(self.num_bins))
		self.permutation_inverse = np.argsort(self.permutation)

	@staticmethod
	def new_nrec(nrec: int) -> 'RandomMapping':
		return RandomMapping(2**nrec, 2**nrec)
	
	def indices(self, F: np.ndarray, bbox: Box | None = None) -> np.ndarray:
		indices = np.ravel_multi_index(self.indices2d(F, bbox).T, dims=(self.nx, self.ny))
		return indices[self.permutation]

	def indices_to_indices2d(self, indices: np.ndarray) -> np.ndarray:
		indices2d = np.vstack(np.unravel_index(indices, shape=(self.nx, self.ny))).T
		return indices2d[self.permutation_inverse]

	def __str__(self) -> str:
		return f'RandomMapping{{nx={self.nx}, ny={self.ny}, random_seed={self.random_seed}}}'


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

	def __str__(self) -> str:
		return f'LinearMapping{{tangent={self.tangent}}}'