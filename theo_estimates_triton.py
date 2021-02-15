import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import pandas as pd
import asikainen_model as am
from scipy.integrate import odeint
from scipy.optimize import fsolve
import pickle
import os


def paa_par_re(na, sa, sb):
    """
    Static point for Paa, for par (preferent attch with rewiring) model, removing edges
    """
    nb = 1 - na
    paa = na*sa*(1-nb*sb) / (1-na*sa + 1-nb*sb)
    pbb = nb*sb*(1-na*sa) / (1-na*sa + 1-nb*sb)
    pab = 2*(-1 + na*sa)*(-1 + nb*sb) / (1-na*sa + 1-nb*sb)
    #if paa < 0:
    #    return 0
    #elif paa > 1:
    #    return 1
    #else:
    #    return paa
    return paa

def taa_par_re(na, sa, sb):
    nb = 1 - na
    paa = paa_par_re(na, sa, sb)
    pbb = paa_par_re(nb, sb, sa)
    pab = - (-1 + 2*na*sa)*(-1 + 2*nb*sb) / (na*sa + nb*sb -1)

    c = paa + pab + pbb
    if paa > 0:
        taa = 2 * paa / (1 + paa - pbb)
    else:
        taa = 0

    if pbb > 0:
        tbb = 2 * pbb / (1 - paa + pbb)
    else:
        tbb = 0

    return taa, tbb

def rho_par_re(na, sa, sb, alpha=.01):
    taa, tbb = taa_par_re(na, sa, sb)
    rho = np.sqrt(2*(taa - 1/(1+alpha))**2 + 2*(tbb)**2)
    return rho


def plot_rho_par_re_eqs(alpha=.01):
    s_vec = np.arange(0, 201)/201
    na_vec = [.1, .3, .5, .7, .9]

    fig, ax = plt.subplots()
    for na in na_vec:
        rhos = []
        for s in s_vec:
            p = solve_paa_pbb(na, s, s)
            t = t_from_p(p)
            rho = rho_from_t(t)
            #rho = rho_par_re(na, s, s, alpha)
            rhos.append(rho)
        ax.plot(s_vec, rhos, label=r'$n_a = $' + '{}'.format(np.round(na, 2)))
    ax.legend(loc=3)
    ax.set_xlabel(r'$s=s_a=s_b$')
    ax.set_ylabel(r'$\rho(T, T_{ideal})$')
    ax.set_ylim([0, 2])
    fig.savefig('theo_rho_PA_rng_eqs.pdf')

def plot2_rho_par_re_eqs():
    s_vec = np.arange(100, 201)/201
    na_vec = [.1, .3, .5, .7 ,.9]

    fig, ax = plt.subplots()
    for na in na_vec:
        rhos = []
        for s in s_vec:
            t = solve2_paa_pbb(na, s, s)
            #t = t_from_p(p)
            rho = rho_from_t(t)
            rhos.append(rho)
        ax.plot(s_vec, rhos, label=r'$n_a = $' + '{}'.format(np.round(na, 2)))
    ax.legend(loc=3)
    ax.set_xlabel(r'$s=s_a=s_b$')
    ax.set_ylabel(r'$\rho(T, T_{ideal})$')
    fig.savefig('theo2_rho_PA_rng_eqs.pdf')

def plot_sim_rho_par_re_eqs(alpha=.01):
    s_vec = np.arange(6, 11)/11
    na_vec = [.3, .5, .9]

    fig, ax = plt.subplots()
    N = 1000
    p_init = 0.01
    p0 = p_init * np.ones((2, 2)) ##[[1, .015], [.015, .015]]
    n_links = p_init * N * (N-1) / 2
    n_iter = int(10000*n_links)
    for na in na_vec:
        rhos = []
        for s in s_vec:
            print('Running: na={}, sa={}'.format(na, np.round(s, 2)))
            p, t, _, rho, T = am.run_rewiring(N, na, 1, [s, s], p0, n_iter, track_steps=n_iter/3, rewire_type="pa_one", remove_neighbor=True)
            import pdb; pdb.set_trace()
            #t = t_from_p(p)
            rho = rho_from_t(t)
            #rho = rho_par_re(na, s, s, alpha)
            rhos.append(rho) #[0])
        ax.plot(s_vec, rhos, label=r'$n_a = $' + '{}'.format(np.round(na, 2)))
    ax.legend(loc=3)
    ax.set_xlabel(r'$s=s_a=s_b$')
    ax.set_ylabel(r'$\rho(T, T_{ideal})$')
    fig.savefig('sim_rho_PA_rng_eqs.pdf')
    return rhos

def rho_from_t(t, alpha=.0):
    rho = np.sqrt(2*(t[0]-1/(1+alpha))**2 + 2*(t[1])**2)
    return rho


def solve_paa_pbb(na, sa, sb, analytic=False):
    def model(p): #, t):
        T = t_from_p(p)
        maa = .5*(p[0] + 1 - p[1])*sa
        mab = .5*(p[1] + 1 - p[0])*(1-sa)
        mba = .5*(p[0] + 1 - p[1])*(1-sb)
        mbb = .5*(p[1] + 1 - p[0])*sb
        cp = na*(maa + mab) + (1-na)*(mba + mbb)
        dpaadt = na*maa - na*T[0]*(maa + mab)#(maa #+ mab + mba)
        dpbbdt = (1-na)*mbb - (1-na)*T[1]*(mba + mbb) #mbb #+ mba + mab)
        dpdt = [dpaadt, dpbbdt]
        return dpdt
    if second:
        denom = -2 + 5*sb - 3*sb*2 + sa**2*(-3 + 4*sb) + sa*(5 - 10*sb + 4*sb**2)
        paa = sa*(-1+sa)*(1-2*sb)**2 / denom
        pbb = sb*(-1+sb)*(1-2*sa)**2 / denom
        pab = -2*(1-3*sa + 2*sa**2)*(1-3*sb + 2*sb**2) / denom
        c = sum([paa, pbb, pab])
        p_star = [paa/c, pbb/c]
        return p_star

    t = np.linspace(0, 100, 1000) #np.arange(0, 1000)
    p0 = [sa, sb]
    p_star = fsolve(model, p0)

    return p_star


def solve2_paa_pbb(na, sa, sb, analytic=False):
    def model(p, t):
        T = t_from_p(p)

        maa = (na*T[0] + (1-na)*(1-T[1]))*sa
        mab = (na*(1-T[0]) + (1-na)*T[1])*(1-sa)
        mba = (na*T[0] + (1-na)*(1-T[1]))*(1-sb)
        mbb = (na*(1-T[0]) + (1-na)*T[1])*sb
        cp = na*(maa + mab) + (1-na)*(mba + mbb)
        dpaadt = na*maa -  na * T[0]*(maa + mab)
        dpbbdt = (1-na)*mbb - (1-na)*T[1]*(mba + mbb)
        dpdt = [dpaadt, dpbbdt]
        return dpdt

    if analytic:
        sab = 1 - sa
        sba = 1 - sb
        nb = 1 - na
        print('sa:{}; sb:{}'.format(sa, sb))
        taa_denom = na*(sa - sab)*(sa*sb - sab*sba)
        taa = sa*(na*sb*(sa - sab) + nb*sab*(sba - sb))
        print('taa: {}; taa_d:{}'.format(taa, taa_denom))
        taa /= taa_denom
        tbb_denom = nb*(sb - sba)*(sa*sb - sab*sba)
        tbb = sb*(na*sa*(sb - sba) + na*sba*(sab - sa))
        print('tbb: {}; tbb_d:{}'.format(tbb, tbb_denom))
        tbb /= tbb_denom
        print('taa: {}; tbb:{}'.format(taa, tbb))
        print('----------------------')
        taa = min([1, taa])
        taa = max([0, taa])
        tbb = min([1, tbb])
        tbb = max([0, tbb])
        return [taa, tbb]


    t = np.linspace(0, 100, 1000) #np.arange(0, 1000)
    p0 = [sa, sb]
    #T = fsolve(model, p0)
    p = odeint(model, p0, t)[-1]
    T = t_from_p(p)

    return T


def t_from_p(p):
    pab = 1-p[0]-p[1]
    try:
        taa = 2*p[0] / (2*p[0] + pab)
    except ZeroDivisionError:
        taa = 0
    try:
        tbb = 2*p[1] / (2*p[1] + pab)
    except ZeroDivisionError:
        tbb = 0
    T = [taa, tbb]
    return T



def plot_rho_par_re_ds(alpha=.01):
    sa_vec = np.round(np.arange(11, 20)/20, 2)
    sb_vec = np.round(np.arange(11, 20)/20, 2)
    na_vec = [.1, .5, .9]

    fig, axs = plt.subplots(1, len(na_vec), figsize=(6, 2), sharey=True)
    for na, ax in zip(na_vec, axs):
        rho_sq = []
        for sa in sa_vec:
            rhos = []
            for sb in sb_vec:
                p = solve_paa_pbb(na, sa, sb)
                t = t_from_p(p)
                rho = rho_from_t(t)
                #rho = rho_par_re(na, sa, sb, alpha)
                rhos.append(rho)
            rho_sq.append(rhos)
        rho_sq = pd.DataFrame(rho_sq, columns=sa_vec, index=sb_vec)
        sns.heatmap(rho_sq, center=1, ax=ax, vmin=0, vmax=2)
        #ax.set_xlabel(r'$s_a$')
        #ax.set_ylabel(r'$s_b$')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig('theo_rho_PA1_rng_ds.pdf')

def plot2_rho_par_re_ds(alpha=.01):
    sa_vec = np.round(np.arange(11, 20)/20, 2)
    sb_vec = np.round(np.arange(11, 20)/20, 2)
    na_vec = [.3, .5, .7]

    fig, axs = plt.subplots(1, len(na_vec), figsize=(6, 2), sharey=True)
    for na, ax in zip(na_vec, axs):
        rho_sq = []
        for sa in sa_vec:
            rhos = []
            for sb in sb_vec:
                t = solve2_paa_pbb(na, sa, sb, analytic=False)
                #t = t_from_p(p)
                rho = rho_from_t(t)
                rhos.append(rho)
            rho_sq.append(rhos)
        rho_sq = pd.DataFrame(rho_sq, columns=sa_vec, index=sb_vec)
        sns.heatmap(rho_sq, center=1, ax=ax, vmin=0, vmax=2)
        #ax.set_xlabel(r'$s_a$')
        #ax.set_ylabel(r'$s_b$')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig('theo_rho_PA2_rng_ds.pdf')


def n_iter_from_specs(p0, N, na):
    Na, Nb = na * N, (1-na)*N
    n_iter = p0[0, 0]*Na*(Na-1)/2 + p0[0, 1]*Na*Nb  + p0[1, 1]*Nb*(Nb-1)/2
    return int(n_iter * 10000)


def parallel_sim(sa, sb, na, specs={}):
    rewire_type = specs.get('rewire_type', 'pa_one')
    N = specs.get('N', 1000)
    c = specs.get('c', 1)
    p0 = specs.get('p0', .015 * np.ones((2,2)))
    n_iter = n_iter_from_specs(p0, N, na)
    remove_neigh = specs.get('remove_neigh', True)
    p, t, W, rho, _ = am.run_rewiring(N, na, c, [sa, sb], p0, n_iter, track_steps=n_iter/10, rewire_type=rewire_type, remove_neighbor=remove_neigh)
    results = {'na': na,
            'sa': sa,
            'sb': sb,
            'N': N,
            'c': c,
            'remove_neigh': True,
            'p0': p0,
            'n_iter': n_iter,
            'p': p,
            't': t,
            'W': W,
            'rho': rho}
    pf = lambda x: int(np.round(100 * x, 2))
    filename = specs.get('filename', 'sa{}_sb{}_na{}.p'.format(pf(sa), pf(sb), pf(na)))
    readwrite_results(filename, results)
    return results

def readwrite_results(filename, results):
    try:
        f = open(filename, 'rb')
        f_res = pickle.load(f)
        f.close()
    except FileNotFoundError:
        f_res = {}

    n = len(f_res)
    f_res[n] = results

    with open(filename, 'wb') as f:
        pickle.dump(results, f)

pf = lambda x: int(np.round(100 * x, 2))

def run_sa_na(sa, na, specs, path):
    sbs = np.arange(.5, 1.05, .05)
    filename = os.path.join(path, 'sa{}_na{}.p'.format(pf(sa), pf(na)))
    specs['filename'] = filename
    for sb in sbs:
        results = parallel_sim(sa, sb, na, specs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sa', type=float, default=0.50)
    parser.add_argument('--p0', type=str, default='equal')
    parser.add_argument('--na', type=float, default=0.50)
    parser.add_argument('--path', type=str, default='./')

    pargs = parser.parse_args()
    if pargs.p0 == 'equal':
        p0 = 0.015 * np.ones((2, 2))
    if pargs.p0 == 'sa-core':
        p0 = 0.015 * np.array([[1, .5], [.5, 0]])
    if pargs.p0 == 'sb-core':
        p0 = 0.015 * np.array([[0, .5], [.5, 1]])
    if pargs.p0 == 'two-com':
        p0 = 0.015 * np.array([[1, 0], [0, 1]])

    specs = {'N': 100,
        'c': 1,
        'p0': p0,
        'remove_neigh': True
        }
    path = os.path.join(pargs.path, pargs.p0)
    run_sa_na(pargs.sa, pargs.na, specs, path)




