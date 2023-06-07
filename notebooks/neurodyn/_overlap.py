"""Overlap metric"""

import numpy as np
import warnings
from ._rnn import LowRankRNN, BinMappedRNN, DenseRNN

__all__ = ['overlap', 'projection']

def overlap(rnn: LowRankRNN | BinMappedRNN | DenseRNN, h: np.ndarray) -> np.ndarray:
	"""Compute overlap

	Parameters
	----------
	rnn : LowRankRNN | BinMappedRNN
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
		return np.einsum('am,a...,a->m...', rnn.tildeG, rnn.phi(h), rnn.bincounts) / rnn.bintotal
	elif isinstance(rnn, DenseRNN):
		warnings.warn(f'attempting to compute overlap on a DenseRNN {rnn}, unless rnn.G is set, this will probably not work')
		return np.einsum('im,i...->m...', rnn.G, rnn.phi(h)) / rnn.N
	else:
		raise RuntimeError(f'cannot compute overlap on {rnn} of type {type(rnn)}')


def projection(rnn: LowRankRNN | BinMappedRNN | DenseRNN, h: np.ndarray) -> np.ndarray:
	"""Compute the projection of the neural field onto the fixed point

	Parameters
	----------
	rnn : LowRankRNN | BinMappedRNN
	h : np.ndarray
		neuron potentials, shape (N, T)

	Returns
	-------
	np.ndarray
		overlap with patterns, shape (p, T)
	"""
	if isinstance(rnn, LowRankRNN):
		return np.einsum('im,i...->m...', rnn.F, h) / rnn.N
	elif isinstance(rnn, BinMappedRNN):
		return np.einsum('am,a...,a->m...', rnn.F, h, rnn.bincounts) / rnn.bintotal
	elif isinstance(rnn, DenseRNN):
		warnings.warn(f'attempting to compute overlap on a DenseRNN {rnn}, unless rnn.G is set, this will probably not work')
		return np.einsum('im,i...->m...', rnn.F, h) / rnn.N
	else:
		raise RuntimeError(f'cannot compute overlap on {rnn} of type {type(rnn)}')