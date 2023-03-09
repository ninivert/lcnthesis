"""Plotting utilities"""

# TODO
# 	- cleanup animation writer

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.integrate._ivp.ivp import OdeResult
from ._rnn import LowRankRNN
from ._overlap import overlap

__all__ = ['plot_neuron_trajectory', 'plot_overlap_trajectory', 'plot_dh_hist']

def _unwrap_figax(figax: tuple[Figure, Axes] | None = None) -> tuple[Figure, Axes]:
	if figax is None:
		figax = plt.subplots()

	return figax

def plot_neuron_trajectory(res: OdeResult, n: int = 5, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	ax.set_xlabel('Time $t$ [s]')
	ax.set_ylabel('Potential $h_i$ [V]')

	for i in range(n):
		ax.plot(res.t, res.y[i, :], label=f'${i=}$', **kwargs)

	ax.legend()

	return fig, ax


def plot_overlap_trajectory(rnn: LowRankRNN, res: OdeResult, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	ax.set_xlabel('Time $t$ [s]')
	ax.set_ylabel('Overlap $m^{{\\mu}}$')
	ax.set_title('Overlap trajectory')
	m = overlap(rnn, res.y)

	for mu in range(len(m)):
		ax.plot(res.t, m[mu], label=f'$\\mu={mu}$', **kwargs)

	ax.grid(axis='y')
	ax.legend()

	return fig, ax


def plot_dh_hist(rnn: LowRankRNN, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	for mu in range(rnn.F.shape[1]):
		ax.hist(rnn.dh(0, rnn.F[:, mu]), bins=30, histtype='step', label=f'$\\mu = {mu}$', **kwargs)

	ax.set_xlabel('$\\dot h_i$')
	ax.set_title('Derivative $\dot h_i$ at $\\xi_i^{\\mu}$')
	ax.legend()

	return fig, ax