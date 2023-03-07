"""Implementation of a numerically estimated lagging function"""

__all__ = ['LaggingFunction']

import numpy as np

class LaggingFunction:
	"""Uses h(t) history in order to estimate h(t-delta)"""

	def __init__(self, ts: list[float], hs: list[np.ndarray], delta: float):
		self.ts = list(ts)  # take a copy, we are modifying these
		self.hs = list(hs)
		self.delta = delta

	def __call__(self, t: float, h: np.ndarray) -> np.ndarray:
		"""Return an estimation of h(t-delta)
		
		Note: this must be called increasing values of t"""

		self.ts.append(t)
		self.hs.append(h)

		t_minus_delta = t - self.delta
		idx_upper = np.searchsorted(self.ts, t_minus_delta)
		idx_lower = max(idx_upper - 1, 0)

		if abs(self.ts[idx_upper]-t_minus_delta) < 1e-13:
			# we have found the h_delay exactly
			h_delay = self.hs[idx_upper]

		elif idx_upper == 0:
			# we do not yet have a history for h
			h_delay = self.hs[0]
		
		else:
			# we interpolate linearly
			t_lower, t_upper = self.ts[idx_lower], self.ts[idx_upper]
			h_lower, h_upper = self.hs[idx_lower], self.hs[idx_upper]
			h_delay = (t_minus_delta - t_lower)/(t_upper - t_lower)*(h_upper - h_lower) + h_lower

		# remove useless history
		del self.ts[:idx_lower]
		del self.hs[:idx_lower]

		return h_delay