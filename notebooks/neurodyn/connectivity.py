"""Generate connectivity matrices J"""

import numpy as np
from typing import Callable

__all__ = ['make_low_rank', 'make_low_rank_valentin']

def make_low_rank(p: int, f: Callable[[np.ndarray], np.ndarray], g: Callable[[np.ndarray], np.ndarray], rvgen: Callable[[], np.ndarray]) -> np.ndarray:
	# TODO
	pass

def make_low_rank_valentin(p: int, N: int, phi: Callable[[np.ndarray], np.ndarray], random_state: int = 42, self_connections: bool = False):
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
		J += np.einsum('i,j->ij', phi_z, z)
		z_samples.append(z)
	J /= N  # O(1/N) scaling
	if not self_connections:
		J -= np.diagflat(np.diagonal(J))  # remove self-connections
		
	return J, z_samples