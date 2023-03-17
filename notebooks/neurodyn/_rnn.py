"""Simulate rate networks"""

import scipy.integrate
import numpy as np
from typing import Callable, Union, Self
from tqdm import tqdm
from dataclasses import dataclass, field
import hashlib
import pickle
import os
import itertools
import logging
from pathlib import Path
from ._lagging import LaggingFunction


__all__ = ['DiscreteRNN', 'LowRankRNN', 'LowRankCyclingRNN', 'LowRankRNNParams', 'LowRankCyclingRNNParams', 'Result']

_logger = logging.getLogger(__name__)

@dataclass
class LowRankRNNParams:
	F: np.ndarray = field(repr=False)  # N, p matrix
	G: np.ndarray = field(repr=False)  # N, p matrix
	phi: Callable[[np.ndarray], np.ndarray]  # phi(u) : R -> R 
	I_ext: Callable[[float], np.ndarray]  # I_ext(t): R -> N vector (current for each neuron)
	exclude_self_connections: bool  # whether to include self-connections

	def __post_init__(self):
		assert self.F.shape == self.G.shape, 'F and G must have the same shape'
		self.N = self.F.shape[0]
		self.p = self.F.shape[1]

	@classmethod
	def new_valentin(cls: Self, p: int, N: int, phi: Callable[[np.ndarray], np.ndarray], random_state: int = 42, **kwargs) -> Self:
		"""Make a new RNN using the setup from Valentin's paper

		Parameters
		----------
		p : int
			rank of the connectivity
		N : int
			number of neurons
		phi : Callable[[np.ndarray], np.ndarray]
			activation function
		random_state : int, optional
			random seed to be used for the sampling, by default 42
		**kwargs :
			additionnal arguments to be passed to ``LowRankRNNParams.__init__``

		Returns
		-------
		LowRankRNNParams
		"""

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

		return cls(F=F, G=G, phi=phi, **kwargs)


@dataclass
class LowRankCyclingRNNParams(LowRankRNNParams):
	delta: float  # delay for the activation
	shift: int  # roll shift


@dataclass
class SimulationParams:
	h0: np.ndarray
	t_span: tuple[float, float]
	dt_max: float


@dataclass
class Result:
	params: LowRankRNNParams
	simparams: SimulationParams
	t: np.ndarray | None
	h: np.ndarray | None


class LowRankRNN:
	"""RNN with low-rank connectivity $J = \sum_\mu F^\mu_i G^\mu_i$"""

	def __init__(self, params: LowRankRNNParams):
		self.params = params

		self.F = params.F
		self.G = params.G
		self.phi = params.phi
		self.I_ext = params.I_ext
		self.exclude_self_connections = params.exclude_self_connections
		self.N = params.N
		self.p = params.p

		self._pbar: tqdm | None = None

	def dh(self, t: float, h: np.ndarray) -> np.ndarray:
		rhs = np.zeros_like(h)
		rhs -= h  # exponential decay
		rhs += self.I_rec(t, h)  # recurrent drive
		rhs += self.I_ext(t)  # external drive
		if self._pbar is not None:
			self._pbar.update(t-self._pbar.n)
		
		return rhs

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		rate = self.phi(h)  # firing rate
		drive += np.einsum('im,jm,j->i', self.F, self.G, rate, optimize=['einsum_path', (1, 2), (0, 1)])
		if self.exclude_self_connections:  # remove self-connections
			drive -= np.einsum('im,im,i->i', self.F, self.G, rate, optimize=['einsum_path', (0, 1), (0, 1)])
		drive /= self.params.N
		return drive

	def simulate(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1, progress: bool = False, cache: bool = False) -> 'Result':
		simparams = SimulationParams(h0=h0.copy(), t_span=t_span, dt_max=dt_max)

		if cache:
			hsh = sha256_params(self.params, simparams)
			# attempt to load a previous simulation
			res_or_none = load_or_none(self.params, simparams)
			if res_or_none is not None:
				_logger.info(f'[{str(self)}] loading cached simulation {hsh[:10]}...')
				return res_or_none

		if progress:
			self._pbar = tqdm(total=t_span[1], desc=f'simulating {str(self)}', bar_format='{desc}: {percentage:.2f}%|{bar}| t={n:.3f} of {total_fmt} [{elapsed}<{remaining}]')
		
		sol = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
		res = Result(self.params, simparams, t=sol.t, h=sol.y)
		
		if progress:
			self._pbar.close()
			self._pbar = None

		if cache:
			# if we get to here, that means this simulation has not been cached
			_logger.info(f'[{str(self)}] writing cache for simulation {hsh[:10]}...')
			dump(res)

		return res

	def __str__(self) -> str:
		return f'LowRankRNN{{N={self.params.N}, p={self.params.p}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}}}'

	@staticmethod
	def new_valentin(*args, **kwargs) -> 'LowRankRNN':
		return LowRankRNN(LowRankRNNParams.new_valentin(*args, **kwargs))


class LowRankCyclingRNN(LowRankRNN):
	"""RNN with low-rank connectivity, cycling through patterns"""

	def __init__(self, params: LowRankCyclingRNNParams):
		super().__init__(params)

		self.shift = params.shift
		self.delta = params.delta

		self.F_rolled = np.roll(self.F, shift=self.shift, axis=1)  # implement the cycling behavior

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		h_lag = self.h_lagging(t, h)  # lagging firing rate
		rate = self.phi(h_lag)
		drive += np.einsum('im,jm,j->i', self.F_rolled, self.G, rate, optimize=['einsum_path', (1, 2), (0, 1)])
		if self.exclude_self_connections:	# remove self-connections
			drive -= np.einsum('im,im,i->i', self.F_rolled, self.G, rate, optimize=['einsum_path', (0, 1), (0, 1)])
		drive /= self.params.N
		return drive

	def simulate(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1, progress: bool = False, cache: bool = False) -> 'Result':
		self.h_lagging = LaggingFunction([t_span[0]], [h0], self.delta)
		res = super().simulate(h0, t_span, dt_max, progress=progress, cache=cache)
		del self.h_lagging
		return res

	def __str__(self) -> str:
		return f'LowRankCyclingRNN{{N={self.params.N}, p={self.params.p}, delta={self.delta}, shift={self.shift}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}}}'

	@staticmethod
	def new_valentin(*args, **kwargs) -> 'LowRankCyclingRNN':
		return LowRankCyclingRNN(LowRankCyclingRNNParams.new_valentin(*args, **kwargs))


CACHEDIR = Path('cache')

def dump(res: Result):
	filedir = CACHEDIR / sha256_params(res.params, res.simparams)
	os.makedirs(filedir, exist_ok=True)

	with open(filedir / 'params.pkl', 'wb') as file:
		pickle.dump(res.params, file)

	with open(filedir / 'simparams.pkl', 'wb') as file:
		pickle.dump(res.simparams, file)

	np.savez(filedir / 'arrays.npz', t=res.t, h=res.h)


def load_or_none(params: LowRankRNNParams, simparams: SimulationParams) -> Result | None:
	filedir = CACHEDIR / sha256_params(params, simparams)
	
	if not filedir.exists():
		return None

	arrays = np.load(filedir / 'arrays.npz')
	return Result(params, simparams, arrays['t'], arrays['h'])


def sha256_params(params: LowRankRNNParams, simparams: SimulationParams) -> str:
	h = hashlib.sha256()

	for key, value in itertools.chain(params.__dict__.items(), simparams.__dict__.items()):
		if isinstance(value, str):
			h.update(value.encode('utf-8'))
		elif isinstance(value, (tuple, list, bool, int, float)) or value is None:
			h.update(str(value).encode('utf-8'))
		elif callable(value):
			h.update(f'{value.__module__}.{value.__name__}'.encode('utf-8'))
		else:
			h.update(value)

	return h.hexdigest()


class DiscreteRNN:
	# TODO : harmonize this class with the other two classes
	"""RNN with an arbitrary connectivity matrix J and current I_ext"""

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

	def simulate(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1):
			res = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
			return res