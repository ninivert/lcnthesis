import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.animation as animation
from tqdm import tqdm
from scipy import stats
from scipy.integrate._ivp.ivp import OdeResult
from pathlib import Path
from math import ceil
from ._rnn import LowRankRNN
from ._plot import plot_overlap_trajectory, scale_lightness

__all__ = ['animate2d']

def midpoints(arr):
	return (arr[:-1] + arr[1:]) / 2

def animate2d(rnn: LowRankRNN, res: OdeResult, outpath: Path, time_stride: int = 1, Nmax = 1500):
	idt = 0
	activity = rnn.phi(res.y)
	cmap = mpl.colormaps['RdBu_r']

	fig, axes = plt.subplot_mosaic([
		['a','b'],
		['m','m'],
	], figsize=(10, 8), constrained_layout=True)

	# 2d plotting

	contour = axes['b'].tricontourf(rnn.F[:, 0], rnn.F[:, 1], activity[:, idt], levels=np.linspace(0, 1, 20+1), cmap='RdBu_r')
	cbar = fig.colorbar(contour, ax=axes['b'], label='Activity $A = \\phi(h_i)$ [Hz]')
	sc = axes['b'].scatter(
		rnn.F[:Nmax, 0], rnn.F[:Nmax, 1], s=5,
		facecolors=[ scale_lightness(c[:3], 0.7) for c in cmap(activity[:Nmax, idt]) ], edgecolor=None, alpha=0.6,
		zorder=1000  # we need this, otherwise the new contours get drawn on top of the scatterpoints
	)

	axes['b'].set_title(f'$(\\xi^0_i, \\xi^1_i)$ embedding contour plot\nand {Nmax} scattered neurons', fontsize='medium')
	axes['b'].set_xlabel('$\\xi^0_i$')
	axes['b'].set_ylabel('$\\xi^1_i$')
	axes['b'].set_xlim((-4, 4))
	axes['b'].set_ylim((-4, 4))
	axes['b'].set_aspect('equal')

	# 3d plotting

	ss = axes['a'].get_subplotspec()
	axes['a'].remove()
	axes['a'] = fig.add_subplot(ss, projection='3d')  # replace by 3D axis

	# note : when we plot the bins, we have [y, x] indexing, so we need to invert the x and y arguments here
	bins = stats.binned_statistic_2d(rnn.F[:, 1], rnn.F[:, 0], activity[:, idt], statistic='mean', bins=30, range=((-4, 4), (-4, 4)))
	xx, yy = np.meshgrid(midpoints(bins.x_edge), midpoints(bins.y_edge))
	surf = axes['a'].plot_surface(xx, yy, bins.statistic, cmap='RdBu_r', vmin=0, vmax=1)

	axes['a'].set_zlim((0, 1))
	axes['a'].set_xlim((-4, 4))
	axes['a'].set_ylim((-4, 4))
	axes['a'].set_xlabel('$\\xi^0_i$')
	axes['a'].set_ylabel('$\\xi^1_i$')
	axes['a'].set_zlabel('Activity $A = \\phi(h_i)$ [Hz]')
	axes['a'].view_init(azim=180+45)
	axes['a'].set_title('$(\\xi^0_i, \\xi^1_i)$ embedding surface plot', fontsize='medium')

	# trajectory

	plot_overlap_trajectory(rnn, res, figax=(fig, axes['m']))
	line = axes['m'].axvline(res.t[idt], color='tab:gray', linestyle='--')

	# we need a container to hold references
	things = {
		'contour': contour,
		'line': line,
		'surf': surf,
		'sc': sc,
	}

	with tqdm(total=ceil(len(res.t)/time_stride)+1) as pbar:
		def update(idt: int):

			# update the contour
			for artist in things['contour'].collections:
				artist.remove()
			things['contour'] = axes['b'].tricontourf(rnn.F[:, 0], rnn.F[:, 1], activity[:, idt], levels=np.linspace(0, 1, 20+1), cmap='RdBu_r')
			
			# update the surface
			things['surf'].remove()
			bins = stats.binned_statistic_2d(rnn.F[:, 1], rnn.F[:, 0], activity[:, idt], statistic='mean', bins=30, range=((-4, 4), (-4, 4)))
			things['surf'] = axes['a'].plot_surface(xx, yy, bins.statistic, cmap='RdBu_r', vmin=0, vmax=1)

			# update the line
			things['line'].set_xdata([res.t[idt], res.t[idt]])

			# update the scatter
			things['sc'].set_facecolors([ scale_lightness(c[:3], 0.7) for c in cmap(activity[:Nmax, idt]) ])

			pbar.update(1)

		ani = animation.FuncAnimation(fig, update, frames=range(0, len(res.t), time_stride))
		ani.save(outpath, writer='ffmpeg', fps=15)

		plt.close()
