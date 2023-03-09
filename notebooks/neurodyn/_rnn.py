"""Simulate rate networks"""

import scipy.integrate
import numpy as np
from typing import Callable, Union
from ._lagging import LaggingFunction
from tqdm import tqdm

__all__ = ['DiscreteRNN', 'LowRankRNN', 'LowRankCyclingRNN']

# TODO
# 	- have a way to save the params into a string/json representation (for reproducibility)

class LowRankRNN:
	"""RNN with low-rank connectivity $J = \sum_\mu F^\mu_i G^\mu_i$"""

	def __init__(self,
		F: np.ndarray,  # N, p matrix
		G: np.ndarray,  # N, p matrix
		phi: Callable[[np.ndarray], np.ndarray],  # phi(u) : R -> R 
		I_ext: Callable[[float], np.ndarray],  # I_ext(t): R -> N vector (current for each neuron)
		exclude_self_connections: bool = True,  # whether to include self-connections
	):
		assert F.shape == G.shape, 'F and G must have the same shape'
		self.F = F
		self.G = G
		self.phi = phi
		self.I_ext = I_ext
		self.exclude_self_connections = exclude_self_connections
		self.N = self.F.shape[0]
		self.pbar: tqdm | None = None

	def dh(self, t: float, h: np.ndarray) -> np.ndarray:
		rhs = np.zeros_like(h)
		rhs -= h  # exponential decay
		rhs += self.I_rec(t, h)  # recurrent drive
		rhs += self.I_ext(t)  # external drive
		if self.pbar is not None:
			self.pbar.update(t-self.pbar.n)
		return rhs

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		rate = self.phi(h)  # firing rate
		drive += np.einsum('im,jm,j->i', self.F, self.G, rate, optimize=['einsum_path', (1, 2), (0, 1)])
		if self.exclude_self_connections:  # remove self-connections
			drive -= np.einsum('im,im,i->i', self.F, self.G, rate, optimize=['einsum_path', (0, 1), (0, 1)])
		drive /= self.N
		return drive

	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1, progress: bool = False):
		if progress:
			self.pbar = tqdm(total=t_span[1], desc='simulation time', bar_format='{desc}: {percentage:.2f}%|{bar}| t={n:.3f} of {total_fmt} [{elapsed}<{remaining}]')
		res = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
		if progress:
			self.pbar.close()
			self.pbar = None
		return res


class LowRankCyclingRNN(LowRankRNN):
	"""RNN with low-rank connectivity, cycling through patterns"""

	def __init__(self,
		F: np.ndarray,  # N, p matrix
		G: np.ndarray,  # N, p matrix
		phi: Callable[[np.ndarray], np.ndarray],  # phi(u) : R -> R 
		I_ext: Callable[[float], np.ndarray],  # I_ext(t): R -> N vector (current for each neuron)
		exclude_self_connections: bool = True,  # whether to include self-connections
		delta: float = 1.0,  # delay for the activation
		shift: int = 1,  # roll shift
	):
		super().__init__(F, G, phi, I_ext, exclude_self_connections)
		self.F = np.roll(self.F, shift=shift, axis=1)  # implement the cycling behavior
		self.delta = delta
		self.shift = shift

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		h_lag = self.h_lagging(t, h)  # lagging firing rate
		rate = self.phi(h_lag)
		drive += np.einsum('im,jm,j->i', self.F, self.G, rate, optimize=['einsum_path', (1, 2), (0, 1)])
		if self.exclude_self_connections:	# remove self-connections
			drive -= np.einsum('im,im,i->i', self.F, self.G, rate, optimize=['einsum_path', (0, 1), (0, 1)])
		drive /= self.N
		return drive

	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1, progress: bool = False):
		# TODO : test this
		self.h_lagging = LaggingFunction([t_span[0]], [h0], self.delta)
		res = super().simulate_h(h0, t_span, dt_max, progress)
		del self.h_lagging
		return res


class DiscreteRNN:
	"""RNN with an arbitrary connectivity matrix J and current I_ext"""

	def __init__(self, J: np.ndarray, phi: Callable[[np.ndarray], np.ndarray], I_ext: Callable[[float], np.ndarray]):
		self.J = J  # connectivity matrix
		self.phi = phi  # firing rate
		self.I_ext = I_ext  # external current
		# tau : can be set to 1, since we can just rescale time
		# R : can be set to 1, since we can just rescale J

	def dh(self, t: float, h: np.ndarray) -> np.ndarray:
		rhs = np.zeros_like(h)
		rhs -= h  # exponential decay
		rhs += self.J @ self.phi(h)
		rhs += self.I_ext(t)
		return rhs

	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1):
			res = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
			return res

# NOTE : not actually needed
# class ContinuousRNN:
# 	# TODO : type hint with shapes
# 	def __init__(self,
# 		w: Callable[[np.ndarray, np.ndarray], float],  # w(y, z) : R^d, R^d -> R
# 		phi: Callable[[float], float],  # phi(u) : R -> R
# 		rho: Callable[[np.ndarray], np.ndarray],  # rho(z) : R^d -> R+
# 		I_ext: Callable[[float, np.ndarray], float],  # I_ext(t, z) : R, R^d -> R
# 		grid: np.ndarray
# 	):
# 		self.w = w  # connectivity kernel (callable)
# 		self.phi = phi  # firing rate
# 		self.rho = rho  # distribution on the field
# 		self.I_ext = I_ext  # external current

# 		# TODO : we need a better way of integrating in R^d (monte-carlo ?)
# 		self.grid = grid  # grid on which to evaluate the field

# 	def dh(self, t: float, h: np.ndarray) -> np.ndarray:
# 		# h = h(t) is evaluated on the grid, h(t, grid)
# 		rhs = np.zeros_like(h)
# 		rhs -= h

# 		# integrate on the field


# 		rhs += self.I_ext(t)