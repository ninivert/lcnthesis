"""Simulate rate networks"""

from scipy import stats
import scipy.integrate
import numpy as np
from typing import Callable, Union, Self
from tqdm import tqdm
from dataclasses import dataclass, field
from abc import abstractmethod
import hashlib
import pickle
import os
import itertools
import logging
from pathlib import Path
from ._lagging import LaggingFunction
from ._mapping import BinMapping


__all__ = [
	'RNN', 'DenseRNN', 'BinMappedRNN', 'LowRankRNN', 'LowRankCyclingRNN',
	'RNNParams', 'DenseRNNParams', 'BinMappedRNNParams', 'LowRankRNNParams', 'LowRankCyclingRNNParams',
	'Result'
]

_logger = logging.getLogger(__name__)


@dataclass
class RNNParams:
	phi: Callable[[np.ndarray], np.ndarray]  # phi(u) : R -> R 
	I_ext: Callable[[float], np.ndarray]  # I_ext(t): R -> N vector (current for each neuron)
	exclude_self_connections: bool  # whether to include self-connections
	# tau : can be set to 1, since we can just rescale time
	# R : can be set to 1, since we can just rescale J


@dataclass
class DenseRNNParams(RNNParams):
	J: np.ndarray = field(repr=False)  # N, N connectivity matrix

	def __post_init__(self):
		assert self.J.shape[0] == self.J.shape[1], 'J must be a square matrix'
		self.N = self.J.shape[0]


@dataclass
class BinMappedRNNParams(RNNParams):
	mapping: BinMapping

	# the embedding in the original network
	F: np.ndarray = field(repr=False)  # N, p matrix
	G: np.ndarray = field(repr=False)  # N, p matrix

	def __post_init__(self):
		assert self.F.shape == self.G.shape, 'F and G must have the same shape'
		self.N = self.mapping.num_bins
		self.p = self.F.shape[1]

		mapping_index = self.mapping.indices(self.F)
		binmasks: list[np.ndarray] = [ mapping_index == alpha for alpha in range(self.N) ]
		# |alpha|, number of original neurons in each bin
		self.bincounts: np.ndarray = np.array([ binmasks[alpha].sum() for alpha in range(self.N) ])
		# sum(|alpha|) is the number of neurons in the original network !
		self.bintotal: int = self.bincounts.sum()
		# "effective" low-rank vectors
		self.tildeF = np.vstack([ self.F[binmasks[alpha]].mean(axis=0) if self.bincounts[alpha] != 0 else np.zeros(self.p) for alpha in range(self.N) ])
		self.tildeG = np.vstack([ self.G[binmasks[alpha]].mean(axis=0) if self.bincounts[alpha] != 0 else np.zeros(self.p) for alpha in range(self.N) ])
		self.tildegamma = np.array([ (self.F[binmasks[alpha]] * self.G[binmasks[alpha]]).sum() / self.bincounts[alpha]**2 if self.bincounts[alpha] != 0 else 0 for alpha in range(self.N) ])

	def to_dense(self) -> DenseRNNParams:
		J_ab = np.zeros((self.N, self.N))
		J_ab += np.einsum('am,bm->ab', self.tildeF, self.tildeG)

		if self.exclude_self_connections:  # exclude the self-connections corresponding to the original connectivity
			di = np.diag_indices_from(J_ab)
			J_ab[di] -= self.tildegamma

		J_ab[:, self.bincounts.nonzero()[0]] *= self.bincounts[None, self.bincounts.nonzero()[0]]
		J_ab /= self.bintotal

		return DenseRNNParams(phi=self.phi, I_ext=self.I_ext, exclude_self_connections=self.exclude_self_connections, J=J_ab)


@dataclass
class LowRankRNNParams(RNNParams):
	F: np.ndarray = field(repr=False)  # N, p matrix
	G: np.ndarray = field(repr=False)  # N, p matrix

	def __post_init__(self):
		assert self.F.shape == self.G.shape, 'F and G must have the same shape'
		self.N = self.F.shape[0]
		self.p = self.F.shape[1]
		self.gamma = (self.F * self.G).sum(axis=1)

	def to_dense(self) -> DenseRNNParams:
		"""Compute $J_{ij} = \sum_{\mu=1}^p F_{\mu,i} G_{\mu,j}$"""
		J = np.zeros((self.N, self.N), dtype=float)
		J += np.einsum('im,jm->ij', self.F, self.G)
		J /= self.N  # O(1/N) scaling
		if self.exclude_self_connections:
			J -= np.diagflat(np.diagonal(J))  # remove self-connections

		return DenseRNNParams(phi=self.phi, I_ext=self.I_ext, exclude_self_connections=self.exclude_self_connections, J=J)

	def to_binmapped(self, binmapping: BinMapping) -> BinMappedRNNParams:
		return BinMappedRNNParams(
			phi=self.phi, I_ext=self.I_ext, exclude_self_connections=self.exclude_self_connections, 
			mapping=binmapping,
			F=self.F.copy(), G=self.G.copy()
		)

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

	@classmethod
	def new_mapped_valentin(cls: Self, phi: Callable[[np.ndarray], np.ndarray], mapping_cls: BinMapping, nrec: int, random_state: int = 42, **kwargs) -> Self:
		"""Generate the low-rank RNN corresponding to the 1D mapping of the 2D gaussian space.

		The fractal is generated with ``2*nrec`` samples, then downsampled to ``N=4**nrec`` fractal segments.

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
		
		nfrac = 2*nrec
		mapping = mapping_cls.new_nrec(nfrac)

		z0 = np.random.default_rng(random_state).normal(loc=0, scale=1, size=1_000_000)
		phi_z0 = phi(z0)
		a = np.mean(phi_z0)
		c = np.var(phi_z0)

		Z = stats.norm.ppf(mapping.inverse_samples())
		F = Z
		G = (phi(Z) - a) / c

		F_avg = F.reshape((2**nfrac, 2**nfrac, 2)).mean(axis=1)
		G_avg = G.reshape((2**nfrac, 2**nfrac, 2)).mean(axis=1)
		
		return cls(F=F_avg, G=G_avg, phi=phi, **kwargs)


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
	params: RNNParams | DenseRNNParams | LowRankRNNParams | LowRankCyclingRNNParams | BinMappedRNNParams
	simparams: SimulationParams
	t: np.ndarray | None
	h: np.ndarray | None


class RNN:
	"""Base class for RNNs with current I_ext, activation function phi"""

	def __init__(self, params: RNNParams | DenseRNNParams | LowRankRNNParams | LowRankCyclingRNNParams | BinMappedRNNParams):
		self.params = params

		self.phi = params.phi
		self.I_ext = params.I_ext
		self.exclude_self_connections = params.exclude_self_connections

		self._pbar: tqdm | None = None

	@abstractmethod
	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		"""Compute the recurrent current at time t, given RNN potentials h_i"""
		pass

	def dh(self, t: float, h: np.ndarray) -> np.ndarray:
		rhs = np.zeros_like(h)
		rhs -= h  # exponential decay
		rhs += self.I_rec(t, h)  # recurrent drive
		rhs += self.I_ext(t)  # external drive
		if self._pbar is not None:
			self._pbar.update(t-self._pbar.n)
		
		return rhs

	def simulate(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1, progress: bool = False, cache: bool = False) -> Result:
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


class DenseRNN(RNN):
	"""RNN with a dense connectivity matrix J"""

	def __init__(self, params: DenseRNNParams):
		super().__init__(params)

		self.J = params.J
		self.N = params.N

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		return self.J @ self.phi(h)

	def __str__(self) -> str:
		return f'DenseRNN{{N={self.params.N}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}, exclude_self_connections={self.exclude_self_connections}}}'


class LowRankRNN(RNN):
	"""RNN with low-rank connectivity $J = \sum_\mu F^\mu_i G^\mu_i$"""

	def __init__(self, params: LowRankRNNParams):
		super().__init__(params)

		self.N = params.N
		self.p = params.p
		self.F = params.F
		self.G = params.G
		self.gamma = params.gamma

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		rate = self.phi(h)  # firing rate
		drive += np.einsum('im,jm,j->i', self.F, self.G, rate, optimize=['einsum_path', (1, 2), (0, 1)])
		if self.exclude_self_connections:  # remove self-connections
			drive -= self.gamma * rate
		drive /= self.params.N
		return drive

	def __str__(self) -> str:
		return f'LowRankRNN{{N={self.params.N}, p={self.params.p}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}, exclude_self_connections={self.exclude_self_connections}}}'

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
		# override this method because we need to handle the lagging function
		if self.delta == 0:
			self.h_lagging = lambda t, h: h
		else:
			self.h_lagging = LaggingFunction([t_span[0]], [h0], self.delta)
		res = super().simulate(h0, t_span, dt_max, progress=progress, cache=cache)
		del self.h_lagging
		return res

	def __str__(self) -> str:
		return f'LowRankCyclingRNN{{N={self.params.N}, p={self.params.p}, delta={self.delta}, shift={self.shift}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}, exclude_self_connections={self.exclude_self_connections}}}'

	@staticmethod
	def new_valentin(*args, **kwargs) -> 'LowRankCyclingRNN':
		return LowRankCyclingRNN(LowRankCyclingRNNParams.new_valentin(*args, **kwargs))


class BinMappedRNN(RNN):
	"""RNN representing the [0,1] mapping of a low rank RNN"""

	def __init__(self, params: BinMappedRNNParams):
		super().__init__(params)
		
		self.N = params.N  # this is equal to mapping.num_bins
		self.p = params.p
		self.tildeF = params.tildeF
		self.tildeG = params.tildeG
		self.tildegamma = params.tildegamma

		self.mapping = params.mapping
		self.bincounts = params.bincounts
		self.bintotal = params.bintotal

	def I_rec(self, t: float, h: np.ndarray) -> np.ndarray:
		drive = np.zeros_like(h)
		rate = self.phi(h)  # firing rate
		drive += np.einsum('b,am,bm,b->a', self.bincounts, self.tildeF, self.tildeG, rate, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
		
		if self.exclude_self_connections:  # remove self-connections
			drive -= self.bincounts * self.tildegamma * rate

		drive /= self.bintotal
		return drive

	def __str__(self) -> str:
		return f'BinMappedRNN{{N={self.N}, p={self.p}, phi={self.phi.__name__}, I_ext={self.I_ext.__name__}, exclude_self_connections={self.exclude_self_connections}}}'


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

	# TODO : sanity check, test that the stored params.pkl matches the input (no hash collisions)

	return Result(params, simparams, arrays['t'], arrays['h'])


def sha256_params(params: RNNParams | DenseRNNParams | LowRankRNNParams | LowRankCyclingRNNParams, simparams: SimulationParams) -> str:
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