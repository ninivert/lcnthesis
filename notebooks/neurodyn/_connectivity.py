"""Generate connectivity matrices J"""

import numpy as np
from typing import Callable
import warnings

__all__ = ['make_low_rank_valentin', 'make_F_G_valentin']

def make_low_rank_valentin(p: int, N: int, phi: Callable[[np.ndarray], np.ndarray], random_state: int = 42, exclude_self_connections: bool = True):
	J = np.zeros((N, N))
	z_samples = []
	s = np.random.default_rng(random_state)

	# we take a large sample for s to standardize
	z0 = s.normal(loc=0, scale=1, size=1_000_000)
	phi_z0 = phi(z0)
	a = np.mean(phi_z0)
	c = np.var(phi_z0)

	for mu in range(p):
		z = s.normal(loc=0, scale=1, size=N)
		phi_z = (phi(z) - a) / c
		J += np.einsum('i,j->ij', z, phi_z)
		z_samples.append(z)
	J /= N  # O(1/N) scaling
	if exclude_self_connections:
		J -= np.diagflat(np.diagonal(J))  # remove self-connections

	return J, z_samples

def make_F_G_valentin(p: int, N: int, phi: Callable[[np.ndarray], np.ndarray], random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
	warnings.warn('use LowRankRNN.new_valentin instead')

	F, G = np.zeros((N, p)), np.zeros((N, p))
	s = np.random.default_rng(random_state)

	# we take a large sample for s to standardize
	z0 = s.normal(loc=0, scale=1, size=1_000_000)
	phi_z0 = phi(z0)
	a = np.mean(phi_z0)
	c = np.var(phi_z0)

	for mu in range(p):
		z = s.normal(loc=0, scale=1, size=N)
		F[:, mu] = z
		G[:, mu] = (phi(z) - a) / c

	return F, G

def make_F_G(p: int, f: Callable[[np.ndarray], np.ndarray], g: Callable[[np.ndarray], np.ndarray], rvgen: Callable[[], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
	# TODO
	pass