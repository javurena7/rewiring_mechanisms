import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import numpy as np
import get_fixed_points as gfp


def plot_grow_evol(sa, sb, c, na, P_obs=None, ax=None, evol_samp=100, t=None, color=None, title=''):
    if not ax:
        fig, ax = plt.subplots()
    if not t:
        t = np.linspace(0, 10000, 1000)
    Pa = np.random.uniform(0, 1, evol_samp)
    Pb = [np.random.uniform(0, pa) for pa in Pa]
    L = np.random.poisson(100, evol_samp)
    for pa, pb, l in zip(Pa, Pb, L):
        ysol = gfp.growth_path(c, na, sa, sb, [pa, pb], l, t)
        ax.plot(t, ysol[:, 0], alpha=.1, color='orangered')
        ax.plot(t, ysol[:, 1], alpha=.1, color='royalblue')
    if P_obs:
        ax.axhline(P_obs[0], linestyle='--', label=r'$P_{aa}$', color='orangered')
        ax.axhline(P_obs[1], linestyle='--', label=r'$P_{bb}$', color='royalblue')
        ax.legend()
    ax.set_ylabel(r'$P$' + '-matrix')
    ax.set_xlabel(r'$t$')
    if title:
        ax.set_title(title)


def plot_rewire_evol(sa, sb, c, na, P_obs=None, ax=None, evol_samp=100, t=None, color=None, title=''):
    if not ax:
        fig, ax = plt.subplots()
    if not t:
        t = np.linspace(0, 100, 3000)
    Pa = np.random.uniform(0, 1, evol_samp)
    Pb = [np.random.uniform(0, pa) for pa in Pa]
    for pa, pb in zip(Pa, Pb):
        ysol = gfp.rewire_path(c, na, sa, sb, [pa, pb], t)
        ax.plot(t, ysol[:, 0], alpha=.1, color='orangered')
        ax.plot(t, ysol[:, 1], alpha=.1, color='royalblue')
    if P_obs:
        ax.axhline(P_obs[0], linestyle='--', label=r'$P_{aa}$', color='orangered')
        ax.axhline(P_obs[1], linestyle='--', label=r'$P_{bb}$', color='royalblue')
        ax.legend()
    ax.set_ylabel(r'$P$' + '-matrix')
    ax.set_xlabel(r'$t$')
    if title:
        ax.set_title(title)

