import matplotlib.pyplot as plt; plt.ion()
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import get_fixed_points as gfp
import json


def plot_grow_evol(sa, sb, c, na, T_obs=None, ax=None, evol_samp=100, t=None, colors=None, title='', linestyle='-'):
    if not ax:
        fig, ax = plt.subplots()
    if not t:
        t = np.linspace(0, 10000, 1000)
    if not colors:
        colors = ('orangered', 'royalblue')
    Pa = np.random.uniform(0, 1, evol_samp)
    Pb = [np.random.uniform(0, pa) for pa in Pa]
    L = np.random.poisson(100, evol_samp)
    for pa, pb, l in zip(Pa, Pb, L):
        ysol = gfp.growth_path(c, na, sa, sb, [pa, pb], l, t)
        ysol = np.array([p_to_t((x[0], x[1])) for x in ysol])
        ax.plot(t, ysol[:, 0], alpha=.1, color=colors[0], linestyle=linestyle)
        ax.plot(t, ysol[:, 1], alpha=.1, color=colors[1], linestyle=linestyle)
    if T_obs:
        ax.axhline(T_obs[0], linestyle='--', label=r'$T_{aa}$', color=colors[0])
        ax.axhline(T_obs[1], linestyle='--', label=r'$T_{bb}$', color=colors[1])
        ax.legend()
    ax.set_ylabel(r'$T$' + '-matrix')
    ax.set_xlabel(r'$t$')
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)

def p_to_t(p):
    pa, pb = p
    pab = 1-pa-pb
    taa = 2*pa / (2*pa + pab)
    tbb = 2*pb / (2*pb + pab)
    return [taa, tbb]

def p_to_t_sols(sols):
    t = []
    for y in sols:
        pa, pb = y
        pab = 1-pa-pb
        taa = 2*pa / (2*pa + pab)
        tbb = 2*pb / (2*pb + pab)
        t.append([taa, tbb])
    return np.array(t)

def plot_rewire_evol(sa, sb, c, na, T_obs=None, ax=None, evol_samp=100, t=None, color=None, title=''):
    if not ax:
        fig, ax = plt.subplots()
    if not t:
        t = np.linspace(0, 100, 3000)
    Pa = np.random.uniform(0, 1, evol_samp)
    Pb = [np.random.uniform(0, pa) for pa in Pa]
    for pa, pb in zip(Pa, Pb):
        ysol = gfp.rewire_path(c, na, sa, sb, [pa, pb], t)
        ysol = p_to_t_sols(ysol)
        ax.plot(t, ysol[:, 0], alpha=.1, color='orangered')
        ax.plot(t, ysol[:, 1], alpha=.1, color='royalblue')
    if T_obs is not None:
        ax.axhline(T_obs[0], linestyle='--', label=r'$T_{aa}$', color='orangered')
        ax.axhline(T_obs[1], linestyle='--', label=r'$T_{bb}$', color='royalblue')
        ax.legend()
    ax.set_ylabel(r'$T$' + '-matrix')
    ax.set_xlabel(r'$t$')
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)

def plot_rewire_predict(sa, sb, c, na, P0=(0,0), T_obs=(0,0), ax=None, t=None, color=None, title='', r_steps=5000, t_size=2500, rewiring_metadata=[(0, '')], extra=1000):
    """
    r_steps: rewiring steps between the intial condition and the observed value
    t_size: average number of rewiring steps included in one theoretical timestep
    """
    if not ax:
        fig, ax = plt.subplots()
    if not t:
        t_tot = (r_steps + extra) / t_size
        t = np.linspace(0, t_tot, 5000)
    Pa0, Pb0 = P0
    ysol = gfp.rewire_path(c, na, sa, sb, P0, t)

    ysol = p_to_t_sols(ysol)
    ax.plot(t*t_size, ysol[:, 0], alpha=.9, color='orangered') #t*tsize = rewireing steps
    ax.plot(t*t_size, ysol[:, 1], alpha=.9, color='royalblue')
    #ax.axhline(T_obs[0], linestyle='--', label='Obs '+r'$T_{aa}$', color='orangered')
    #ax.axhline(T_obs[1], linestyle='--', label='Obs '+r'$T_{bb}$', color='royalblue')

    t_obs = r_steps # / t_size
    ax.plot(t_obs, T_obs[0], 'x', color='orangered', markersize=12, label='Obs ' + r'$T_{aa}$')
    ax.plot(t_obs, T_obs[1], 'x', color='royalblue', markersize=12, label='Obs' + r'$T_{bb}$')

    for (loc, txt) in rewiring_metadata:
        ax.axvline(loc, linestyle='--', alpha=.3, color='grey')
        ax.text(loc+5, .07, str(txt), color='grey', alpha=.8)

    ax.legend(loc='upper right')
    ax.set_ylabel(r'$T$' + '-matrix')
    ax.set_xlabel(r'$t$' + ' (rewiring events)')
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)

def plot_rewire_policy(params, changes=[['sa', (.4, .6)]], P0=(0, 0), ax=None, ts=None, t_size=2500):
    """
    Plot the effect of a change in parameters, where the change of params occurs only once (two values for a single parameter)
    params: dict with stable values 'c', 'sa', 'sb' or 'na'
    changes: list of lists of changes. Each sublist contains ['param', (val1, val2)]
    P0: initial value
    ts: list of two timeseries (length of evolution)
    """
    if not ax:
        fig, ax = plt.subplots()
    params['P'] = P0
    params['t'] = ts[0]
    for change in changes:
        params[change[0]] = change[1][0]

    ysol = gfp.rewire_path(**params)
    P1 = ysol[-1, :]
    ysol = p_to_t_sols(ysol)
    ax.plot(ts[0]*t_size, ysol[:, 0], alpha=.9, color='orangered') #t*tsize = rewireing steps
    ax.plot(ts[0]*t_size, ysol[:, 1], alpha=.9, color='royalblue')

    params['P'] = P1
    params['t'] = ts[1]
    for change in changes:
        params[change[0]] = change[1][1]

    ysol = gfp.rewire_path(**params)
    ysol = p_to_t_sols(ysol)
    t2 = (ts[0][-1] + ts[1]) * t_size
    ax.plot(t2, ysol[:, 0], alpha=.9, color='orangered', label=r'$T_{aa}$')
    ax.plot(t2, ysol[:, 1], alpha=.9, color='royalblue', label=r'$T_{bb}$')

    txt_dict = {'sa': r'$s_a$', 'sb': r'$s_b$', 'c': r'$c$', 'na': r'$n_a$'}

    txt = '\n'.join([txt_dict[v] + f': {k[0]}' + r'$\rightarrow$' + f'{k[1]}' for v, k in changes])
    ax.axvline(t2[0], linestyle='--', alpha=.3, color='grey')
    ax.text(t2[0]+5, .07, txt, color='grey', alpha=.8)

    ax.legend(loc='upper right')
    ax.set_ylabel(r'$T$' + '-matrix')
    ax.set_xlabel(r'$t$' + ' (rewiring events)')
    ax.set_ylim(0, 1)













def plot_grow_predic(sa, sb, c, na, P0=(0,0,0), T_obs=None, ax=None, evol_samp=100, t=None, colors=None, title='', linestyle='-'):
    if not ax:
        fig, ax = plt.subplots()
    if not t:
        t = np.linspace(0, 10000, 1000)
    if not colors:
        colors = ('orangered', 'royalblue')
    Pa0, Pb0, L0 = P0
    #L = np.random.poisson(100, evol_samp)
    ysol = gfp.growth_path(c, na, sa, sb, [Pa0, Pb0], L0, t)
    ysol = np.array([p_to_t((x[0], x[1])) for x in ysol])
    ax.plot(t, ysol[:, 0], alpha=.9, color=colors[0], linestyle=linestyle)
    ax.plot(t, ysol[:, 1], alpha=.1, color=colors[1], linestyle=linestyle)
    if T_obs:
        Tar, Tbr, Lr = T_obs
        ax.axhline(Tar, linestyle='--', label='Obs' + r'$T_{aa}$', color=colors[0])
        ax.axhline(Tbr, linestyle='--', label='Obs' + r'$T_{bb}$', color=colors[1])
        ax.legend()
    ax.set_ylabel(r'$T$' + '-matrix')
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
    # TODO: change from P matrix to T matrix
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
    # TODO: change from P matrix to T matrix
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
    # TODO: change from P matrix to T matrix
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

def plot_param_timeline(df, ax):
    ax.plot(df.y0, df.sa, '.', color='orangered', label=r'$s_a$')
    ax.plot(df.y0, df.sa, '-', color='orangered', alpha=.3)

    ax.plot(df.y0, df.sb, '.', color='royalblue', label=r'$s_b$')
    ax.plot(df.y0, df.sb, '-', color='royalblue', alpha=.3)

    ax.plot(df.y0, df.c, '.', color='k', label=r'$c$')
    ax.plot(df.y0, df.c, '-', color='k', alpha=.3)

    ax.plot(df.y0, df.na, '.', color='forestgreen', label=r'$n_a$')
    ax.plot(df.y0, df.na, '-', color='forestgreen', alpha=.3)

    ax.legend()

def plot_isi_countires_prediction(w_path, rval=10):
    plt.ioff()
    path = 'isi/country_plots/{}.pdf'

    obs = pd.read_csv(w_path, sep='|',dtype={'f1': str, 'f2':str, 'y0': int, 'yn': int})
    obs[['a', 'b']] = obs[['f1', 'f2']].applymap(lambda x: '+'.join(eval(x)))
    obs = obs[obs['yn'] - obs['y0'] == rval]
    obs = obs[obs.c > 0]
    obs = obs[(obs.laa > 100) & (obs.lbb > 100) & (obs.lab > 100)]

    axs_len = obs.shape[0] + 1
    fig, axs = plt.subplots(1, axs_len, figsize=(axs_len*3, 3))

    plot_param_timeline(obs, axs[0])

    title = 'A: {}, B: {}'.format(obs.a.iloc[0], obs.b.iloc[0])
    fig.suptitle(title)
    outname = '{}_{}_evolution'.format(obs.a.iloc[0], obs.b.iloc[0])
    outname = path.format(outname)
    i = 1
    import pdb; pdb.set_trace()
    for (_, row) in obs.iterrows():
        L = row.laa + row.lab + row.lbb
        paa = row.laa / L
        pbb = row.lbb / L
        T_obs = p_to_t([paa, pbb])
        row_title = '{}-{}'.format(row.y0, row.yn)

        #plot_grow_evol(row.sa, row.sb, 0, row.na, ax=axs[i], evol_samp=40, t=None, colors=('orange', 'deepskyblue'), linestyle=':')
        plot_grow_evol(row.sa, row.sb, row.c, row.na, T_obs=T_obs, ax=axs[i], evol_samp=100, t=None, colors=None, title=row_title)
        i += 1
    fig.savefig(outname)
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





