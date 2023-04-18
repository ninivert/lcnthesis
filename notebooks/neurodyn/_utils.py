import numpy as np
from ._activation import sigmoid

__all__ = ['fake_activity']

def fake_activity(F: np.ndarray, n=(1,1), phi=sigmoid):
	"""Generate some fake activity field, with normal vector ``n``"""
	n = np.array(n, dtype=float)
	n /= np.linalg.norm(n)
	return phi(F.dot(n))