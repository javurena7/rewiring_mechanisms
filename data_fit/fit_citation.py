import growth_degree_fit as gdf
import sys
from utils import pacs_utils as pu
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
from itertools import combinations
from pandas import DataFrame
import json
import get_fixed_points as gfp
import pandas as pd

def plot_params_citation():
    fig, axs = plt.subplots(1, 2, figsize=(7, 3)) #)3, figsize=(10, 3))
    groups = ['01', '02', '03', '04', '05', '07', '11', '12', '21', '23', '24', '25']
    #groups = ['21', '23', '24', '25']
    groups = ['61', '62', '63', '64', '65', '66', '67', '68']
    #groups = ['41', '42', '43', '44', '45', '46', '47']
    ng = len(groups)
    sas, cvals, nas = np.zeros((ng, ng)), np.zeros((ng, ng)), np.zeros((ng, ng))
    for a, b in combinations(groups, 2):
        print('{} {}'.format(a, b))
        if (a=='03') and (b=='05'):
            na = .489
            sol = np.array([0.90670508, 0.89958437, 0.91045515])
        elif (b=='04') and (a=='03'):
            na = .309
            sol = np.array([0.72704438, 0.7201636 , 0.64134653])
        elif (a=='02') and (b=='03'):
            na = .2753
            sol = np.array([0.6397818 , 0.71169366, 0.73589583])
        elif (a=='21') and (b=='23'):
            na = .2753
            sol = np.array([0.47921657, 0.92070757, 0.87983703])
        elif (a=='21') and (b=='24'):
            na = .2753
            sol = np.array([0.75192532, 0.86477237, 0.90861399])
        elif (a=='21') and (b=='25'):
            na = .2753
            sol = np.array([0.78758477, 0.83090862, 0.93376391])
        elif (a=='23') and (b=='24'):
            na = .2753
            sol = np.array([0.98843447, 0.70408613, 0.69104485])
        elif (a=='23') and (b=='25'):
            na = .2753
            sol = np.array([0.7642502, 0.68182421, 0.63081204])
        elif (a=='24') and (b=='25'):
            na = .2753
            sol = np.array([0.65484339, 0.64167383, 0.90201066])
        else:
            x, n, na = pu.network_stats(a=[a], b=[b])
            GF = gdf.GrowthFit(x, n, na)
            sol = GF.solve()
        print(sol)
        i = groups.index(a)
        j = groups.index(b)
        sas[i, j] = sol[0]
        sas[j, i] = sol[1]
        cvals[i, j] = sol[2]
        cvals[j, i] = sol[2]
        nas[i, j] = na
        nas[j, i] = na

        axs[0].imshow(sas)
        axs[1].imshow(cvals)
        #axs[2].imshow(nas)
        fig.savefig('citation_params.pdf')
    #sns.set(font_scale=1.2)
    group_names = json.load(open('utils/pacs_ref.json', 'r'))
    names = [group_names[i] for i in groups]
    sas = DataFrame(sas, columns=names, index=names)
    cvals = DataFrame(cvals, columns=names)
    nas = DataFrame(nas, columns=names)

    sns.heatmap(sas, ax=axs[0])
    sns.heatmap(cvals, ax=axs[1])
    axs[0].set_yticklabels(axs[0].get_ymajorticklabels(), fontsize = 8)
    axs[1].set_yticklabels(axs[1].get_ymajorticklabels(), fontsize = 8)
    axs[0].set_xticklabels(axs[0].get_xmajorticklabels(), fontsize = 8)
    axs[1].set_xticklabels(axs[1].get_xmajorticklabels(), fontsize = 8)
    #sns.heatmap(nas, ax=axs[2])
    axs[0].set_title('Homophily')
    axs[1].set_title('Pref. Attachment')
    #axs[2].set_title('Minority Size')
    fig.tight_layout()
    fig.savefig('plots/citation_params_40.pdf')


def get_all_cp_pairs(outpath=''):
    group_names = json.load(open('utils/pacs_ref.json', 'r'))
    group_names.pop('01')
    group_names.pop('99')
    df = DataFrame(columns=['a', 'b', 'paa', 'pbb', 'pab', 'acore', 'bcore', 'sbm_pmat', 'na', 'N', 'taa', 'tbb'])
    for a, b in combinations(group_names, 2):
        print(a, b)
        _, _, obs = pu.cp_stats([a], [b])
        obs['a'] = a
        obs['b'] = b
        df = df.append(obs, ignore_index=True)
        df.to_csv(outpath, index=None, sep='|')

def get_average_m(x):
    ms = []
    for xt in x.values():
        mt = sum([xi[3] for xi in xt])
        ms.append(mt)
    return np.mean(ms)


def plot_t_error(n_samp=10):
    groups = ['61', '62', '63', '64', '65', '66', '67', '68']
    groups = ['01', '02', '03', '04', '05', '07', '11', '12', '21', '23', '24', '25']
    fig, ax = plt.subplots(figsize=(4, 4)) #)3, figsize=(10, 3))
    ax.plot([0, 1], [0, 1], alpha=.4)
    ax.set_xlabel('Pref. Attch. Error.')
    ax.set_ylabel('No Pref. Attch. Error.')
    for a, b in combinations(groups, 2):
        print(a, b)
        x, n, obs = pu.network_stats(a=[a], b=[b])
        na = obs['na']
        GF = gdf.GrowthFit(x, n, na)
        print('Fitting model... ')
        sa, sb, c = GF.solve()
        sa0, sb0 = GF.solve_c0()
        del GF
        m = int(get_average_m(x))
        print('Obtaining simulations... ')
        N = obs['N']
        print(N)
        P = gfp.grow_simul_n(c, na, sa, sb, N, m, n_samp)
        P0 = gfp.grow_simul_n(0, na, sa0, sb0, N, m, n_samp)
        T = t_from_p(np.mean(P, 0))
        T0 = t_from_p(np.mean(P0, 0))
        err = np.sqrt((T[0] - obs['taa'])**2 + (T[1]-obs['tbb'])**2)
        err0 = np.sqrt((T0[0] - obs['taa'])**2 + (T0[1]-obs['tbb'])**2)
        ax.plot(err, err0, '.',c='b')
        fig.savefig('plots_sep/t_error_citation.pdf')


def asp_error(n_samp=10):
    df = pd.read_csv('asp_cp_top_pseq.csv', sep=' ', dtype={'a': str, 'b': str})
    fig, ax = plt.subplots(figsize=(4, 4)) #)3, figsize=(10, 3))
    ax.plot([0, 1], [0, 1], alpha=.4)
    ax.set_xlabel('Pref. Attch. Error.')
    ax.set_ylabel('No Pref. Attch. Error.')
    line = 'a|b|sa|sb|c|sa0|sb0|Paa|Pbb|err|err0\n'
    with open('asp/cp_top_pseq_fit.csv', 'w') as w:
        w.write(line)
    for row in zip(df[['a', 'b']].values):
        a, b = row[0][0], row[0][1]
        x, n, obs = pu.network_stats(a=[a], b=[b])
        na = obs['na']
        GF = gdf.GrowthFit(x, n, na)
        print('Fitting model... ')
        sa, sb, c = GF.solve()
        sa0, sb0 = GF.solve_c0()
        del GF
        m = int(get_average_m(x))
        print('Obtaining simulations... ')
        N = obs['N']
        print(N)
        t = np.linspace(0, 10000, 5000)
        ysol = gfp.growth_path(c, na, sa, sb, [1/3, 1/3], 1000, t)
        P = ysol[-1]
        #P = gfp.grow_simul_n(c, na, sa, sb, N, m, n_samp)
        ysol0 = gfp.growth_path(0, na, sa0, sb0, [1/3, 1/3], 1000, t)
        P0 = ysol0[-1]
        #P0 = gfp.grow_simul_n(0, na, sa0, sb0, N, m, n_samp)
        T = t_from_p(P)
        T0 = t_from_p(P0)
        err = np.sqrt((T[0] - obs['taa'])**2 + (T[1]-obs['tbb'])**2)
        err0 = np.sqrt((T0[0] - obs['taa'])**2 + (T0[1]-obs['tbb'])**2)
        line = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(a, b, sa, sb, c, sa0, sb0, P[0], P[1], err, err0)
        with open('asp/cp_top_pseq_fit.csv', 'a') as w:
            w.write(line)
        ax.plot(err, err0, '.',c='b')
        fig.savefig('asp/asp_error_citation.pdf')


def t_from_p(P):
    paa, pbb, _ = P
    pab = 1 - paa - pbb
    taa = (2*paa) / (2*paa + pab)
    tbb = (2*pbb) / (2*pbb + pab)
    return taa, tbb






