"""Plotting utilities"""

import matplotlib.pyplot as plt, matplotlib.ticker as ticker
import colorsys
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ._rnn import LowRankRNN, Result
from ._overlap import overlap

__all__ = ['plot_neuron_trajectory', 'plot_overlap_trajectory', 'plot_dh_hist', 'add_headers', 'scale_lightness']

def _unwrap_figax(figax: tuple[Figure, Axes] | None = None) -> tuple[Figure, Axes]:
	if figax is None:
		figax = plt.subplots()

	return figax

def plot_neuron_trajectory(res: Result, n: int = 5, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	ax.set_xlabel('Time $t$ [s]')
	ax.set_ylabel('Potential $h_i$ [V]')

	for i in range(n):
		ax.plot(res.t, res.h[i, :], label=f'${i=}$', **kwargs)

	ax.legend()

	return fig, ax


def plot_overlap_trajectory(rnn: LowRankRNN, res: Result, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	ax.set_xlabel('Time $t$ [s]')
	ax.set_ylabel('Overlap $m^{{\\mu}}$')
	ax.set_title('Overlap trajectory')
	m = overlap(rnn, res.h)

	for mu in range(len(m)):
		ax.plot(res.t, m[mu], label=f'$\\mu={mu}$', **kwargs)

	ax.grid(axis='y')
	ax.legend()

	return fig, ax


def plot_dh_hist(rnn: LowRankRNN, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	for mu in range(rnn.F.shape[1]):
		ax.hist(rnn.dh(0, rnn.F[:, mu]), bins=30, histtype='step', label=f'$\\mu = {mu}$', density=True, **kwargs)

	ax.yaxis.set_major_locator(ticker.NullLocator())
	ax.yaxis.set_minor_locator(ticker.NullLocator())
	ax.set_xlabel('$\\dot h_i$')
	ax.set_title('Derivative $\dot h_i$ at $\\xi_i^{\\mu}$')
	ax.legend()

	return fig, ax


def add_headers(
	fig,
	*,
	row_headers=None,
	col_headers=None,
	row_pad=1,
	col_pad=5,
	rotate_row_headers=True,
	**text_kwargs
):
	# Stolen from : https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
	# Based on https://stackoverflow.com/a/25814386

	axes = fig.get_axes()

	for ax in axes:
		sbs = ax.get_subplotspec()

		# Putting headers on cols
		if (col_headers is not None) and sbs.is_first_row():
			ax.annotate(
				col_headers[sbs.colspan.start],
				xy=(0.5, 1),
				xytext=(0, col_pad),
				xycoords="axes fraction",
				textcoords="offset points",
				ha="center",
				va="baseline",
				**text_kwargs,
			)

		# Putting headers on rows
		if (row_headers is not None) and sbs.is_first_col():
			ax.annotate(
				row_headers[sbs.rowspan.start],
				xy=(0, 0.5),
				xytext=(-ax.yaxis.labelpad - row_pad, 0),
				xycoords=ax.yaxis.label,
				textcoords="offset points",
				ha="right",
				va="center",
				rotation=rotate_row_headers * 90,
				**text_kwargs,
			)


def scale_lightness(rgb, scale_l):
	# Stolen from : https://stackoverflow.com/a/60562502

	# convert rgb to hls
	h, l, s = colorsys.rgb_to_hls(*rgb)
	# manipulate h, l, s values and return as rgb
	return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)