import get_fixed_points as gfp
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import numpy as np
from itertools import product



def get_phase_space_grow(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250, ax=None):
    xvals = np.linspace(*xlims, n_size + 2)[1:-1]
    yvals = np.linspace(*ylims, n_size + 2)[1:-1]
    acore, bcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
    acore[:], bcore[:] = np.nan, np.nan
    acore2, bcore2 = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
    acore2[:], bcore2[:] = np.nan, np.nan
    if ax is None:
        fig, ax = plt.subplots()
    for (j, xv), (i, yv) in product(enumerate(xvals), enumerate(yvals)):
        print(i,j)
        params[x] = xv
        params[y] = yv
        dens, ca, cb = gfp.growth_fixed_points_density(**params)
        acore, bcore = classify_densities(dens, ca, cb, acore, bcore, i, j)
        acore2[i,j] = ca[0] if ca[0] > max([cb[0],0]) else np.nan
        bcore2[i,j] = cb[0] if cb[0] > max([ca[0],0]) else np.nan
    border = get_border(n_size, acore, bcore)
    return acore, bcore, acore2, bcore2, border


def get_border(n_size, acore, bcore):
    border = np.zeros((n_size, n_size)); border[:] = np.nan
    for i, j in product(range(n_size), range(n_size)):
        if i > 0:
            if np.isnan(acore[i-1, j]) and not np.isnan(acore[i, j]):
                border[i-1, j] = 1
            if np.isnan(bcore[i-1, j]) and not np.isnan(bcore[i, j]):
                border[i-1, j] = 1
        if i-1 < n_size:
            if np.isnan(acore[i+1, j]) and not np.isnan(acore[i, j]):
                border[i+1, j] = 1
            if np.isnan(bcore[i+1, j]) and not np.isnan(bcore[i, j]):
                border[i+1, j] = 1
        if j > 0:
            if np.isnan(acore[i, j-1]) and not np.isnan(acore[i, j]):
                border[i, j-1] = 1
            if np.isnan(bcore[i, j-1]) and not np.isnan(bcore[i, j]):
                border[i, j-1] = 1
        if j+1 < n_size:
            if np.isnan(acore[i, j+1]) and not np.isnan(acore[i, j]):
                border[i, j+1] = 1
            if np.isnan(bcore[i, j+1]) and not np.isnan(bcore[i, j]):
                border[i, j+1] = 1
    return border






def classify_densities(dens, ca, cb, acore, bcore, i, j):
    paa, pab, pbb = dens[0]
    if (paa >= pab) & (pab >= pbb):
        acore[i, j] = ca[0]
    elif (pbb >= pab) & (pab >= paa):
        bcore[i, j] = cb[0]
    return acore, bcore





