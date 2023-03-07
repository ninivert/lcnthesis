"""Overlap metric"""

import numpy as np
from ._rnn import LowRankRNN

__all__ = ['overlap']

def overlap(rnn: LowRankRNN, h: np.ndarray) -> np.ndarray:
	"""Compute overlap

	Parameters
	----------
	rnn : LowRankRNN
	h : np.ndarray
		neuron potentials, shape (N, T)

	Returns
	-------
	np.ndarray
		overlap with patterns, shape (p, T)
	"""
	return np.einsum('im,i...->m...', rnn.G, rnn.phi(h)) / rnn.N