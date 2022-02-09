import matplotlib.pyplot as plt; plt.ion()
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import get_fixed_points as gfp
import json


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
    ax.set_ylim(0, 1)
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
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)

def plot_params_full(sas, sbs, c, na, ax):
    ax.text(0, sas[0], r'$A$', color='orangered')
    ax.text(0, sas[1], r'$A_0$', color='orangered')
    ax.text(0, sbs[0], r'$B$', color='royalblue')
    ax.text(0, sbs[1], r'$B_0$', color='royalblue')
    ax.text(0, c, r'$c$', color='k')
    ax.text(0, na, r'$n_a$', color='k')
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(0, 1)

def plot_params_simple(sa, sb, c, na, ax):
    ax.text(0, sa, r'$A$', color='orangered')
    ax.text(0, sb, r'$B$', color='royalblue')
    ax.text(0, c, r'$c$', color='k')
    ax.text(0, na, r'$n_a$', color='k')
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(0, 1)

def plot_full_grow_scenario(sas, sbs, c, na, P_obs=None, title='', evol_samp=100, t=None, outname=''):
    fig = plt.figure(figsize=(7, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 3])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    plot_params_full(sas, sbs, c, na, ax0)
    plot_grow_evol(sas[0], sbs[0], c, na, P_obs=P_obs, ax=ax1, evol_samp=evol_samp, t=t, color=None,  title='PA')
    plot_grow_evol(sas[1], sbs[1], 0, na, P_obs=P_obs, ax=ax2, evol_samp=evol_samp, t=t, color=None,  title='No PA')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outname)

def plot_simple_grow_scenario(sa, sb, c, na, P_obs=None, title='', evol_samp=100, t=None, outname=''):
    fig = plt.figure(figsize=(4, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    plot_params_simple(sa, sb, c, na, ax0)
    plot_grow_evol(sa, sb, c, na, P_obs=P_obs, ax=ax1, evol_samp=evol_samp, t=t, color=None,  title='PA')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outname)

def plot_isi_countries():
    plt.ioff()
    path = 'isi/country_plots/{}.pdf'
    obs = pd.read_csv('isi/cp_country_results.txt', sep='|', dtype={'a': str, 'b': str, 'yr0': int, 'yrn': int})
    obs[['a', 'b']] = obs[['a', 'b']].applymap(lambda x: '+'.join(eval(x)))
    est = pd.read_csv('isi/country_estimated_params.txt', sep='|', names=['a', 'b', 'sa', 'sb', 'c', 'na', 'yr0', 'yrn'])
    est[['a', 'b']] = est[['a', 'b']].applymap(lambda x: '+'.join(eval(x)))

    est = est[(est.na > .1) & (est.na < .9)]
    print(est.shape)
    for (_, row) in est.iterrows():
        #a = '+'.join(eval(row.a))
        #b = '+'.join(eval(row.b))
        title = 'A: {}, B: {}\n{}-{}'.format(row.a, row.b, row.yr0, row.yrn)
        outname = '{}_{}_{}-{}'.format(row.a, row.b, row.yr0, row.yrn)
        outname = path.format(outname)

        obs_row = obs[(obs.a == row.a) & (obs.b == row.b) & (obs.yr0 == row.yr0) & (obs.yrn == row.yrn)]
        if not obs_row.empty:
            L = obs_row.laa + obs_row.lab + obs_row.lbb
            paa = obs_row.laa / L
            pbb = obs_row.lbb / L
            P_obs = [paa.values[0], pbb.values[0]]
        else:
            P_obs = None

        plot_simple_grow_scenario(row.sa, row.sb, row.c, row.na, P_obs=P_obs, title=title, outname=outname)
        plt.close()

def plot_asp():
    #plt.ioff()
    path = 'asp/plots/{}.pdf'
    dtypes = {'a': str, 'b': str}
    est = pd.read_csv('asp/cp_top_pseq_fit.csv', sep='|', dtype=dtypes)
    obs = pd.read_csv('asp_citation_cp_pairs.csv', sep='|', dtype=dtypes)

    group_names = json.load(open('utils/pacs_ref.json', 'r'))
    for (_, row) in est.iterrows():
        #names = [group_names[i] for i in groups]

        title = 'A: {}\n B: {}'.format(group_names[row.a], group_names[row.b])
        outname = '{}_{}'.format(row.a, row.b)
        outname = path.format(outname)

        obs_row = obs[(obs.a == row.a) & (obs.b == row.b)]
        if not obs_row.empty:
            P_obs = [obs_row.paa.values[0], obs_row.pbb.values[0]]
            na = obs_row.na.values[0]

            sas = [row.sa, row.sa0]
            sbs = [row.sb, row.sb0]
            c = row.c

            plot_full_grow_scenario(sas, sbs, c, na, P_obs=P_obs, title=title, evol_samp=100, t=None,  outname=outname)
            plt.close()
        else:
            print('No empirical observations for {}-{}'.format(row.a, row.b))





