"""Simulate rate networks"""

import scipy.integrate
import numpy as np
from typing import Callable, Union
from tqdm import tqdm
from dataclasses import dataclass, field
import hashlib
import pickle
import os
from pathlib import Path
from ._lagging import LaggingFunction


__all__ = ['DiscreteRNN', 'LowRankRNN', 'LowRankCyclingRNN', 'Params', 'Result']


class LowRankRNN:
	"""RNN with low-rank connectivity $J = \sum_\mu F^\mu_i G^\mu_i$"""

	def __init__(self,
		F: np.ndarray,  # N, p matrix
		G: np.ndarray,  # N, p matrix
		phi: Callable[[np.ndarray], np.ndarray],  # phi(u) : R -> R 
		I_ext: Callable[[float], np.ndarray],  # I_ext(t): R -> N vector (current for each neuron)
		exclude_self_connections: bool = True,  # whether to include self-connections
	):
		assert F.shape == G.shape, 'F and G must have the same shape'
		self.F = F
		self.G = G
		self.phi = phi
		self.I_ext = I_ext
		self.exclude_self_connections = exclude_self_connections
		self.N = self.F.shape[0]
		self.p = self.F.shape[1]
		self.pbar: tqdm | None = None

	def dh(self, t: float, h: np.ndarray) -> np.ndarray:
		# self.t_log.append(t)  # DEBUG
		# self.h_log.append(h.copy())  # DEBUG

		rhs = np.zeros_like(h)
		rhs -= h  # exponential decay
		rhs += self.I_rec(t, h)  # recurrent drive
		rhs += self.I_ext(t)  # external drive
		if self.pbar is not None:
			self.pbar.update(t-self.pbar.n)

		# self.rhs_log.append(rhs.copy())  # DEBUG
		
		return rhs

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		rate = self.phi(h)  # firing rate
		drive += np.einsum('im,jm,j->i', self.F, self.G, rate, optimize=['einsum_path', (1, 2), (0, 1)])
		if self.exclude_self_connections:  # remove self-connections
			drive -= np.einsum('im,im,i->i', self.F, self.G, rate, optimize=['einsum_path', (0, 1), (0, 1)])
		drive /= self.N
		return drive

	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1, progress: bool = False, cache: bool = False) -> 'Result':
		params = Params.from_rnn(self, h0=h0, t_span=t_span, dt_max=dt_max)

		if cache:
			# attempt to load a previous simulation
			res_or_none = load_or_none(params)
			if res_or_none is not None:
				print(f'[{str(self)}] loading cached simulation {params.sha256_digest()[:10]}...')
				return res_or_none

		if progress:
			self.pbar = tqdm(total=t_span[1], desc=f'simulating {str(self)}', bar_format='{desc}: {percentage:.2f}%|{bar}| t={n:.3f} of {total_fmt} [{elapsed}<{remaining}]')
		
		# self.h_log, self.t_log, self.rhs_log = [], [], []  # DEBUG
		res = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
		res = Result(params, t=res.t, h=res.y)
		
		if progress:
			self.pbar.close()
			self.pbar = None

		if cache:
			# if we get to here, that means this simulation has not been cached
			print(f'[{str(self)}] writing cache for simulation {params.sha256_digest()[:10]}...')
			dump(res)

		return res

	def __str__(self) -> str:
		return f'LowRankRNN{{N={self.N}, p={self.p}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}}}'


class LowRankCyclingRNN(LowRankRNN):
	"""RNN with low-rank connectivity, cycling through patterns"""

	def __init__(self,
		F: np.ndarray,  # N, p matrix
		G: np.ndarray,  # N, p matrix
		phi: Callable[[np.ndarray], np.ndarray],  # phi(u) : R -> R 
		I_ext: Callable[[float], np.ndarray],  # I_ext(t): R -> N vector (current for each neuron)
		exclude_self_connections: bool = True,  # whether to include self-connections
		delta: float = 1.0,  # delay for the activation
		shift: int = 1,  # roll shift
	):
		super().__init__(F, G, phi, I_ext, exclude_self_connections)
		self.F_rolled = np.roll(self.F, shift=shift, axis=1)  # implement the cycling behavior
		self.delta = delta
		self.shift = shift

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		h_lag = self.h_lagging(t, h)  # lagging firing rate
		rate = self.phi(h_lag)
		drive += np.einsum('im,jm,j->i', self.F_rolled, self.G, rate, optimize=['einsum_path', (1, 2), (0, 1)])
		if self.exclude_self_connections:	# remove self-connections
			drive -= np.einsum('im,im,i->i', self.F_rolled, self.G, rate, optimize=['einsum_path', (0, 1), (0, 1)])
		drive /= self.N
		return drive

	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1, progress: bool = False, cache: bool = False) -> 'Result':
		self.h_lagging = LaggingFunction([t_span[0]], [h0], self.delta)
		# self.h_lagging = LaggingFunction([], [], self.delta)  # DEBUG
		# self.h_lagging = lambda t, h: h  # DEBUG
		res = super().simulate_h(h0, t_span, dt_max, progress=progress, cache=cache)
		del self.h_lagging  # DEBUG
		return res

	def __str__(self) -> str:
		return f'LowRankCyclingRNN{{N={self.N}, p={self.p}, delta={self.delta}, shift={self.shift}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}}}'


class DiscreteRNN:
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

	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1):
			res = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
			return res


@dataclass(frozen=True)
class Params:
	# RNN network
	F: np.ndarray = field(repr=False)  # N, p matrix
	G: np.ndarray = field(repr=False)  # N, p matrix
	phi: str | Callable[[np.ndarray], np.ndarray]  # phi(u) : R -> R 
	I_ext: str | Callable[[float], np.ndarray]  # I_ext(t): R -> N vector (current for each neuron)
	exclude_self_connections: bool  # whether to include self-connections
	# Cycling RNN network
	delta: float | None  # delay for the activation
	shift: int | None  # roll shift
	# simulation params
	h0: np.ndarray = field(repr=False)
	t_span: tuple[float, float]
	dt_max: float

	@staticmethod
	def from_rnn(rnn: LowRankRNN | LowRankCyclingRNN, h0: np.ndarray, t_span: tuple[float, float], dt_max: float) -> 'Params':
		return Params(
			# rnn network
			F=rnn.F.copy(), G=rnn.G.copy(),
			phi=rnn.phi.__name__, I_ext=rnn.I_ext.__name__,  # we cannot pickle callables, so we store the function names
			exclude_self_connections=rnn.exclude_self_connections,
			# cycling rnn network
			delta=rnn.delta if isinstance(rnn, LowRankCyclingRNN) else None,
			shift=rnn.shift if isinstance(rnn, LowRankCyclingRNN) else None,
			# simulation params
			h0=h0.copy(), t_span=t_span, dt_max=dt_max
		)

	def __eq__(self, other: 'Params') -> bool:
		return \
			(self.F == other.F).all() and (self.G == other.G).all() and \
			self.phi == other.phi and self.I_ext == other.I_ext and \
			self.exclude_self_connections == other.exclude_self_connections and \
			self.delta == other.delta and self.shift == other.shift and \
			(self.h0 == other.h0).all() and \
			self.t_span == other.t_span and self.dt_max == other.dt_max

	def sha256_digest(self) -> str:
		h = hashlib.sha256()

		for value in self.__dict__.values():
			if isinstance(value, str):
				h.update(value.encode('utf-8'))
			elif isinstance(value, (tuple, list, bool, int, float)) or value is None:
				h.update(str(value).encode('utf-8'))
			else:
				h.update(value)

		return h.hexdigest()


@dataclass
class Result:
	params: Params
	t: np.ndarray
	h: np.ndarray


CACHEDIR = Path('cache')

def dump(res: Result):
	filedir = CACHEDIR / res.params.sha256_digest()
	os.makedirs(filedir, exist_ok=True)

	with open(filedir / 'params.pkl', 'wb') as file:
		pickle.dump(res.params, file)

	np.savez(filedir / 'arrays.npz', t=res.t, h=res.h)


def load_or_none(params: Params) -> Result | None:
	filedir = CACHEDIR / params.sha256_digest()
	
	if not filedir.exists():
		return None

	arrays = np.load(filedir / 'arrays.npz')
	return Result(params, arrays['t'], arrays['h'])