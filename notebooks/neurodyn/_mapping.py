"""Compute mappings from R² -> R"""

import numpy as np
from dataclasses import dataclass

__all__ = [
	'Box', 'recursive_quadrant_mapping', 'recursive_quadrant_mapping_coords', 'recursive_quadrant_mapping_index',
	'linear_mapping', 'x_projection_mapping', 'y_projection_mapping', 'xy_projection_mapping', 'xy45_projection_mapping', 'xy135_projection_mapping',
	'irrational_mapping',
]

# Recursive quadrant mapping

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


def recursive_quadrant_mapping_coords(F: np.ndarray, n: int, box: Box | None = None) -> np.ndarray:
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
	coords[mask1, 1:] = recursive_quadrant_mapping_coords(F[mask1, :], n-1, box1)
	coords[mask2, 1:] = recursive_quadrant_mapping_coords(F[mask2, :], n-1, box2)
	coords[mask3, 1:] = recursive_quadrant_mapping_coords(F[mask3, :], n-1, box3)
	coords[mask4, 1:] = recursive_quadrant_mapping_coords(F[mask4, :], n-1, box4)

	return coords

def recursive_quadrant_mapping(F: np.ndarray, n: int = 5) -> np.ndarray:
	"""Recursive quadrant mapping R² -> [0,1]

	See Pham 2018

	Parameters
	----------
	F : np.ndarray of shape (N, 2)
		2D embedding
	n : int
		number of iterations, by default 5

	Returns
	-------
	np.ndarray of shape (N,)
		1D embedding. Values are contained in [0, 1]
	"""		
	coords = recursive_quadrant_mapping_coords(F, n)
	return (coords-1) @ np.logspace(1, n, num=n, base=1/4)

def recursive_quadrant_mapping_index(F: np.ndarray, n: int = 5) -> np.ndarray:
	"""Index of the bins corresponding to the mapping"""
	coords = recursive_quadrant_mapping_coords(F, n)
	return (coords-1) @ np.logspace(n-1, 0, num=n, base=4, dtype=int)

# Linear mappings

def linear_mapping(F: np.ndarray, tangent: np.ndarray) -> np.ndarray:
	"""Linear projection mapping R² -> R by projecting on the tangent vector


	Parameters
	----------
	F : np.ndarray
		2d embedding of shape (N, 2)
	tangent : np.ndarray
		tangent vector of shape (2,)

	Returns
	-------
	np.ndarray
		projected embedding of shape (N,)
	"""
	return F @ tangent

def x_projection_mapping(F: np.ndarray) -> np.ndarray:
	"""Linear projection mapping R² -> R by projecting on (1, 0)"""
	tangent = np.zeros(F.shape[1])
	tangent[0] = 1
	return linear_mapping(F, tangent)

def y_projection_mapping(F: np.ndarray) -> np.ndarray:
	"""Linear projection mapping R² -> R by projecting on (0, 1)"""
	tangent = np.zeros(F.shape[1])
	tangent[1] = 1
	return linear_mapping(F, tangent)

def xy_projection_mapping(F: np.ndarray, angle: float = np.pi/4) -> np.ndarray:
	"""Linear projection mapping R² -> R by projecting on (sin(angle), cos(angle))"""
	tangent = np.zeros(F.shape[1])
	tangent[0] = np.cos(angle)
	tangent[1] = np.sin(angle)
	return linear_mapping(F, tangent)

def xy45_projection_mapping(F: np.ndarray) -> np.ndarray:
	"""Linear projection mapping R² -> R by projecting on (sin(pi/4), cos(pi/4))"""
	return xy_projection_mapping(F, angle=np.pi/4)

def xy135_projection_mapping(F: np.ndarray) -> np.ndarray:
	"""Linear projection mapping R² -> R by projecting on (sin(3pi/4), cos(3pi/4))"""
	return xy_projection_mapping(F, angle=3*np.pi/4)

# Irrational mapping

def irrational_mapping(F: np.ndarray, c: float = np.sqrt(2)) -> np.ndarray:
	"""Injective and continuous mapping which uses the fact `c` is irrational to map lattice aZ² -> R, `a` size of lattice"""
	return linear_mapping(F, np.array([1, c]))