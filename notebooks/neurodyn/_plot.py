"""Plotting utilities"""

import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.ticker as ticker, matplotlib.tri as tri, matplotlib.collections as collections
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import colorsys
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ._rnn import LowRankRNN, Result
from ._overlap import overlap, projection

__all__ = [
	'plot_neuron_trajectory', 'plot_overlap_trajectory', 'plot_dh_hist',
	'plot_overlap_phase2D',
	'plot_2D_embedding_contour', 'plot_2D_embedding_scatter',
	'plot_2D_to_1D_mapping',
	'add_headers', 'scale_lightness',
	'cmap_bi', 'cmap_bi_r', 'cmap_trans',
]

def _unwrap_figax(figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	if figax is None:
		figax = plt.subplots(**kwargs)

	return figax

def plot_neuron_trajectory(res: Result, n: int = 5, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	ax.set_xlabel('Time $t$ [s]')
	ax.set_ylabel('Potential $h_i$ [V]')
	ax.set_title('Neuron trajectory')

	for i in range(n):
		ax.plot(res.t, res.h[i, :], label=f'${i=}$', **kwargs)

	ax.legend()

	return fig, ax


def plot_overlap_trajectory(rnn: LowRankRNN, res: Result, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	ax.set_xlabel('Time $t$ [s]')
	ax.set_ylabel('Latent $\\kappa_{{\\mu}}$')
	ax.set_title('Latent trajectory')
	kappa = projection(rnn, res.h)

	for mu in range(len(kappa)):
		ax.plot(res.t, kappa[mu], label=f'$\\mu={mu+1}$', **kwargs)

	ax.grid(visible=True, axis='y')
	ax.legend()

	return fig, ax


def plot_overlap_phase2D(rnn: LowRankRNN, res: Result, point_start: bool = True, point_end: bool = True, lim01: bool = True, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	ax.set_xlabel('Latent $\\kappa_1(t)$')
	ax.set_ylabel('Latent $\\kappa_2(t)$')
	ax.set_title('Latent trajectory')
	ax.set_aspect('equal')
	kappa = projection(rnn, res.h)
	line, = ax.plot(kappa[0], kappa[1], **kwargs)
	if point_start: ax.plot(kappa[0, 0], kappa[1, 0], 'o', color=line.get_color(), **kwargs)
	if point_end: ax.plot(kappa[0, -1], kappa[1, -1], 'x', color=line.get_color(), **kwargs)
	if lim01:
		ax.set_xlim((0,1))
		ax.set_ylim((0,1))

	return fig, ax

def plot_dh_hist(rnn: LowRankRNN, figax: tuple[Figure, Axes] | None = None, **kwargs) -> tuple[Figure, Axes]:
	fig, ax = _unwrap_figax(figax)

	for mu in range(rnn.F.shape[1]):
		ax.hist(rnn.dh(0, rnn.F[:, mu]), bins=30, histtype='step', label=f'$\\mu = {mu}$', density=True, **kwargs)

	ax.yaxis.set_major_locator(ticker.NullLocator())
	ax.yaxis.set_minor_locator(ticker.NullLocator())
	ax.set_xlabel('$\\dot h_i$')
	ax.set_title('Derivative $\\dot h_i$ at $z_i^{\\mu}$')
	ax.legend()

	return fig, ax


def plot_2D_embedding_contour(
	rnn: LowRankRNN, activity: np.ndarray, draw_cbar: bool = True,
	figax: tuple[Figure, Axes] | None = None
) -> tuple[tuple[Figure, Axes], tri.TriContourSet]:
	fig, ax = _unwrap_figax(figax)

	contour = ax.tricontourf(rnn.F[:, 0], rnn.F[:, 1], activity, levels=np.linspace(0, 1, 20+1), cmap='RdBu_r')
	if draw_cbar:
		cbar = fig.colorbar(contour, ax=ax, label='Activity $A = \\phi(h_i)$ [Hz]')

	ax.set_xlabel('$z^1_i$')
	ax.set_ylabel('$z^2_i$')
	ax.set_aspect('equal')

	return (fig, ax), contour


def plot_2D_embedding_scatter(
	rnn: LowRankRNN, activity: np.ndarray, Nmax: int = 1500,
	figax: tuple[Figure, Axes] | None = None,
	lightness: float | None = None,
	**kwargs
) -> tuple[tuple[Figure, Axes], collections.PathCollection]:
	fig, ax = _unwrap_figax(figax)

	cmap = mpl.colormaps['coolwarm']
	if lightness is not None:
		c = [ scale_lightness(c[:3], lightness) for c in cmap(activity[:Nmax]) ]
	else:
		c = cmap(activity[:Nmax])
	sc = ax.scatter(
		rnn.F[:Nmax, 0], rnn.F[:Nmax, 1], s=5,
		c=c, edgecolor=None, **kwargs,
		zorder=1000  # we need this, otherwise the new contours get drawn on top of the scatterpoints
	)

	ax.set_xlabel('$z^1_i$')
	ax.set_ylabel('$z^2_i$')
	ax.set_aspect('equal')

	return (fig, ax), sc


def plot_2D_to_1D_mapping(
	rnn: LowRankRNN, mapping: np.ndarray, activity: np.ndarray,
	figax: tuple[Figure, Axes] | None = None,
	**kwargs
) -> tuple[tuple[Figure, Axes], collections.PathCollection, collections.PathCollection]:
	fig, axes = _unwrap_figax(figax, nrows=2)
	
	scat2d = axes[0].scatter(rnn.F[:, 0], rnn.F[:, 1], c=mapping, **kwargs)
	axes[0].set_xlabel('$z_i^1$')
	axes[0].set_ylabel('$z_i^2$')
	axes[0].set_aspect('equal')

	scat1d = axes[1].scatter(mapping, activity, c=mapping, **kwargs)
	axes[1].set_xlabel('Mapping')
	axes[1].set_ylabel('Activity $A = \\phi(h_i)$ [Hz]')

	fig.colorbar(scat1d, label='Mapping')

	return (fig, axes), scat1d, scat2d

cmap_bi = LinearSegmentedColormap.from_list("", list(reversed(["#ff0080","#ff0080","#a349a4","#0000ff","#0000ff"])))
cmap_bi_r = LinearSegmentedColormap.from_list("", ["#ff0080","#ff0080","#a349a4","#0000ff","#0000ff"])
cmap_trans = LinearSegmentedColormap.from_list("", ["#55CDFC","#55CDFC","#FFFFFF","#FFFFFF","#F7A8B8","#F7A8B8"])

def add_headers(
	fig=None, axes=None,
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

	if fig is None and axes is None:
		raise RuntimeError('fig or axes must be specified')

	if axes is None:
		axes = fig.get_axes()

	for ax in axes:
		sbs = ax.get_subplotspec()
		if isinstance(sbs, mpl.gridspec.SubplotSpec):
			sbs = sbs.get_topmost_subplotspec()
		
		# Putting headers on cols
		if (col_headers is not None) and (sbs is not None) and sbs.is_first_row():
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
		if (row_headers is not None) and (sbs is not None) and sbs.is_first_col():
			ax.annotate(
				row_headers[sbs.rowspan.start],
				xy=(0, 0.5),
				xytext=(-ax.yaxis.labelpad - row_pad, 0),
				xycoords=ax.yaxis.label,
				textcoords="offset points",
				ha="center",
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