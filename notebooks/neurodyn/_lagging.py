"""Implementation of a numerically estimated lagging function"""

__all__ = ['LaggingFunction']

import numpy as np
import bisect

class LaggingFunction:
	"""Uses h(t) history in order to estimate h(t-delta)"""

	def __init__(self, ts: list[float], hs: list[np.ndarray], delta: float):
		self.ts = list(ts)  # take a copy, we are modifying these
		self.hs = list(hs)
		self.delta = delta

	def __call__(self, t: float, h: np.ndarray) -> np.ndarray:
		"""Return an estimation of h(t-delta)
		
		A call with a (t, h_new) pair will override the stored (t, h) pair
		"""

		# OLD : does not sort !
		# the problem is that scipy.integrate.solve_ivp can call with t going backwards
		# or the same t as before (which is why we do the if statement)
		# self.ts.append(t)
		# self.hs.append(h)

		# print(f'-- called with {t=} --')
		# print('h0 log : ', [h[0] for h in self.hs])
		# print('t log :  ', self.ts)

		# find the index at which to cut
		# this is needed because the solve_ivp can reject steps and refine guesses
		idx_insert = bisect.bisect_left(self.ts, t)
		# when we call with a t < max(self.ts), delete the upper entries (reset history !)
		del self.ts[idx_insert:]
		del self.hs[idx_insert:]
		self.ts.append(t)
		self.hs.append(h.copy())

		# find the location of t-delta
		t_minus_delta = t - self.delta
		idx_upper = bisect.bisect_left(self.ts, t_minus_delta)
		idx_lower = max(idx_upper - 1, 0)

		if self.ts[idx_upper] == t_minus_delta:
		# if abs(self.ts[idx_upper]-t_minus_delta) < 1e-13:
			# we have found the h_delay exactly
			h_delay = self.hs[idx_upper]
			# print('found exact match')

		elif idx_upper == 0:
			# we do not yet have a history for h
			h_delay = self.hs[0]
			# print('no history')
		
		else:
			# we interpolate linearly
			t_lower, t_upper = self.ts[idx_lower], self.ts[idx_upper]
			h_lower, h_upper = self.hs[idx_lower], self.hs[idx_upper]
			h_delay = (t_minus_delta - t_lower)/(t_upper - t_lower)*(h_upper - h_lower) + h_lower
			# print('interpolation')

		# OLD : we cannot remove useless history because the solver might call t < t_current
		# too bad this noticeably affects performance
		# del self.ts[:idx_lower-1]
		# del self.hs[:idx_lower-1]

		# print(f'returning {h_delay[0]=}')
		# print(f'                {h[0]=}')

		return h_delay