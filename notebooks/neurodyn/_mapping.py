"""Compute mappings from R² -> R"""

import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

__all__ = [
	'Mapping', 'BinMapping',
	'RecursiveQuadrantMapping',
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
	def mapping_index(self, F: np.ndarray) -> np.ndarray:
		"""Index of the bins corresponding to the mapping"""
		pass


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
	
	def __call__(self, F: np.ndarray) -> np.ndarray:
		coords = self.mapping_coords(F, self.n)
		return (coords-1) @ np.logspace(1, self.n, num=self.n, base=1/4)

	def mapping_index(self, F: np.ndarray) -> np.ndarray:
		coords = self.mapping_coords(F, self.n)
		return (coords-1) @ np.logspace(self.n-1, 0, num=self.n, base=4, dtype=int)

	def mapping_coords(self, F: np.ndarray, n: int, box: Box | None = None) -> np.ndarray:
		"""Generates coordinates (j1, ..., jn) for the recursive quadrant mapping box -> [0,1]"""

		coords = np.zeros((len(F), n), dtype=int)

		# stopping condition
		if len(F) == 0 or n == 0:
			return coords

		if box is None:
			box = Box(F[:, 0].min(), F[:, 0].max(), F[:, 1].min(), F[:, 1].max())
		
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
		coords[mask1, 1:] = self.mapping_coords(F[mask1, :], n-1, box1)
		coords[mask2, 1:] = self.mapping_coords(F[mask2, :], n-1, box2)
		coords[mask3, 1:] = self.mapping_coords(F[mask3, :], n-1, box3)
		coords[mask4, 1:] = self.mapping_coords(F[mask4, :], n-1, box4)

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