import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union

__all__ = ['Delta', 'Flat', 'Heaviside']

class FunctionalFunction:
	"""A functional which implements mathematical operations"""

	def __add__(self, other: Union[float, 'FunctionalFunction']):
		if isinstance(other, (float, int)):
			return FunctionalFunction(lambda t: self(t) + other)
		return FunctionalFunction(lambda t: self(t) + other(t))

	def __sub__(self, other: Union[float, 'FunctionalFunction']):
		if isinstance(other, (float, int)):
			return FunctionalFunction(lambda t: self(t) - other)
		return FunctionalFunction(lambda t: self(t) - other(t))

	def __mul__(self, other: Union[float, 'FunctionalFunction']):
		if isinstance(other, (float, int)):
			return FunctionalFunction(lambda t: self(t) * other)
		return FunctionalFunction(lambda t: self(t) * other(t))

	def __div__(self, other: Union[float, 'FunctionalFunction']):
		if isinstance(other, (float, int)):
			return FunctionalFunction(lambda t: self(t) / other)
		return FunctionalFunction(lambda t: self(t) / other(t))

	def __neg__(self):
		return FunctionalFunction(lambda t: -self(t))

	def __radd__(self, *args, **kwargs):
		return self.__add__(*args, **kwargs)

	def __rmul__(self, *args, **kwargs):
		return self.__mul__(*args, **kwargs)

	def __rsub__(self, *args, **kwargs):
		return -self.__sub__(*args, **kwargs)

	def __call__(self, t: Union[float, np.ndarray]):
		return self.fn(t)

	def __init__(self, fn: Callable[[Union[float, np.ndarray]], np.ndarray]):
		self.fn = fn


class Delta(FunctionalFunction):
	"""Numerical implementation of the Dirac delta using a sharp bump function"""

	# https://www.wolframalpha.com/input?i2d=true&i=Integrate%5Bexp%5C%2840%29-Divide%5B1%2C1-Power%5Bx%2C2%5D%5D%5C%2841%29%2C%7Bx%2C-1%2C1%7D%5D
	BUMP_INTEGRAL = 0.44399381616807943

	def __init__(self, t0: float, eps: float = 0.1):
		"""Numerical implementation of the Dirac delta using a sharp bump function
	
		During integration, use ``dt < eps/10`` for a good approximation of the integral

		Parameters
		----------
		t0 : float
			Peak of the delta approximation
		eps : float, optional
			Width of the peak, by default 0.1
		"""

		self.t0 = t0
		self.eps = eps

	def __call__(self, t: Union[float, np.ndarray]):
		x = 2/self.eps * (t - self.t0)
		return 2/self.eps * np.where(
			np.abs(x) < 1,
			np.exp(-1/(1-x**2)),
			np.zeros_like(t)
		) / Delta.BUMP_INTEGRAL


class Flat(FunctionalFunction):
	"""Constant current
	
	Parameters
	----------
	I0 : float or numpy.ndarray
		Value of the constant current
	"""

	def __init__(self, I0: Union[float, np.ndarray]):
		self.I0 = I0

	def __call__(self, t: Union[float, np.ndarray]):
		if isinstance(t, (float, int)):
			return np.copy(self.I0)
		return np.full((*t.shape, *self.I0.shape), self.I0)


class Heaviside(FunctionalFunction):
	"""Heaviside function

	Parameters
	----------
	t0 : float
		Location of the step
	"""

	def __init__(self, t0: float):
		self.t0 = t0

	def __call__(self, t: Union[float, np.ndarray]):
		return np.where(
			t - self.t0 < 0,
			np.zeros_like(t),
			np.ones_like(t)
		)


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	# Test dirac delta approximation
	t = np.linspace(0, 10, 5000)
	plt.close('all')
	plt.plot(t, Delta(5)(t))
	plt.show()
	print(f'integral = {np.trapz(Delta(5)(t), t)} ~ 1')