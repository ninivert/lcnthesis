"""Overlap metric"""

import numpy as np
from ._rnn import LowRankRNN, BinMappedRNN

__all__ = ['overlap']

def overlap(rnn: LowRankRNN | BinMappedRNN, h: np.ndarray) -> np.ndarray:
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
	if isinstance(rnn, LowRankRNN):
		return np.einsum('im,i...->m...', rnn.G, rnn.phi(h)) / rnn.N
	elif isinstance(rnn, BinMappedRNN):
		return np.einsum('am,a...,a->m...', rnn.mapping.binned_statistic(rnn.F, h=rnn.G.T).T, rnn.phi(h), rnn.bincounts) / rnn.bincounts.sum()