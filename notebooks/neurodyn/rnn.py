"""Simulate rate networks"""

import scipy.integrate
import numpy as np
from typing import Callable, Union

__all__ = ['DiscreteRNN']

class DiscreteRNN:
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

class ContinuousRNN:
	# TODO : type hint with shapes
	def __init__(self,
		w: Callable[[np.ndarray, np.ndarray], float],  # w(y, z) : R^d, R^d -> R
		phi: Callable[[float], float],  # phi(u) : R -> R
		rho: Callable[[np.ndarray], np.ndarray],  # rho(z) : R^d -> R+
		I_ext: Callable[[float, np.ndarray], float],  # I_ext(t, z) : R, R^d -> R
		grid: np.ndarray
	):
		self.w = w  # connectivity kernel (callable)
		self.phi = phi  # firing rate
		self.rho = rho  # distribution on the field
		self.I_ext = I_ext  # external current

		# TODO : we need a better way of integrating in R^d (monte-carlo ?)
		self.grid = grid  # grid on which to evaluate the field

	def dh(self, t: float, h: np.ndarray) -> np.ndarray:
		# h = h(t) is evaluated on the grid, h(t, grid)
		rhs = np.zeros_like(h)
		rhs -= h

		# integrate on the field


		rhs += self.I_ext(t)

# from dataclasses import dataclass
# @dataclass
# class Populations:
# 	# h: np.ndarray  # input potentials at time t. shape=(K,)
# 	W: np.ndarray  # interaction weights. shape=(K, K)
# 	tau: Union[np.ndarray, float]  # time constant of population activity, shape=(K,)
# 	R: Union[np.ndarray, float]  # resistivity, shape=(K,)
# 	gain_fn: Callable[[np.ndarray], np.ndarray]  # F(h) : R^K -> R^K, gain function of each population
# 	I_ext: Callable[[float], np.ndarray]  # I_ext(t) : R+ -> R^K, external stimulus
# 	# ASSUMPTION : filter_fn is a dirac delta
# 	# filter_fn: Callable[[float], np.ndarray]  # alpha(s) : R -> R^K, filter function of each population

# 	def convolve(self, h):
# 		gain = self.gain_fn(h)
# 		convolved = gain  # filter_fn is a dirac delta, normally this is in integral form
# 		return convolved

# 	def I_network_recurrent(self, h):
# 		W_recurrent = np.diagflat(np.diagonal(self.W))  # diagonal matrix
# 		return W_recurrent @ self.convolve(h)

# 	def I_network_others(self, h):
# 		W_others = self.W - np.diagflat(np.diagonal(self.W))  # zero out diagonal
# 		return W_others @ self.convolve(h)

# 	def I_network(self, h):
# 		convolved = self.convolve(h)
# 		I_network = self.W @ convolved
# 		# I_network = np.einsum('kn,n->k', self.W, convolved)  # same as matrix multiplication
# 		return I_network

# 	def dh(self, t: float, h: np.ndarray):
# 		rhs = np.zeros_like(h)
# 		rhs -= h  # Exponential decay
# 		rhs += self.R * self.I_network(h)  # Network currents
# 		rhs += self.R * self.I_ext(t)  # External currents
# 		rhs /= self.tau  # Apply tau
# 		return rhs

# 	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1):
# 		res = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
# 		return res