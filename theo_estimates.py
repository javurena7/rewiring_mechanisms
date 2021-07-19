import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import pandas as pd
import asikainen_model as am
from scipy.integrate import odeint
from scipy.optimize import fsolve

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
            p, t, W, rho, _ = am.run_rewiring(N, na, 1, [s, s], p0, n_iter, track_steps=n_iter/10, rewire_type="pa_one", remove_neighbor=True)
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
    if analytic:
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


def linear_stab(na, sa, sb, c):
    def model(p, t):
        T = t_from_p(p)

        maa = (c*(na*T[0] + (1-na)*(1-T[1])) + (1-c)*na)*sa
        mab = (c*(na*(1-T[0]) + (1-na)*T[1]) + (1-c)*(1-na))*(1-sa)
        mba = (c*(na*T[0] + (1-na)*(1-T[1])) + (1-c)*na)*(1-sb)
        mbb = (c*(na*(1-T[0]) + (1-na)*T[1]) + (1-c)*(1-na))*sb
        cp = na*(maa + mab) + (1-na)*(mba + mbb)
        dpaadt = na*maa -  na * T[0]*(maa + mab)
        dpbbdt = (1-na)*mbb - (1-na)*T[1]*(mba + mbb)
        dpdt = [dpaadt, dpbbdt]
        return dpdt



def solve2_paa_pbb(na, sa, sb, c=1, analytic=False):
    def model(p, t):
        T = t_from_p(p)

        maa = (c*(na*T[0] + (1-na)*(1-T[1])) + (1-c)*na)*sa
        mab = (c*(na*(1-T[0]) + (1-na)*T[1]) + (1-c)*(1-na))*(1-sa)
        mba = (c*(na*T[0] + (1-na)*(1-T[1])) + (1-c)*na)*(1-sb)
        mbb = (c*(na*(1-T[0]) + (1-na)*T[1]) + (1-c)*(1-na))*sb
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


    t = np.linspace(0, 1000, 1000) #np.arange(0, 1000)
    p0 = [sa, sb]
    #T = fsolve(model, p0)
    p = odeint(model, p0, t)[-1]
    T = t_from_p(p)

    return T

def fixed_point_stability(na, sa, sb, c):
    def model(p, t, return_m=False):
        T = t_from_p(p)
        maa = (c*(na*T[0] + (1-na)*(1-T[1])) + (1-c)*na)*sa
        mab = (c*(na*(1-T[0]) + (1-na)*T[1]) + (1-c)*(1-na))*(1-sa)
        mba = (c*(na*T[0] + (1-na)*(1-T[1])) + (1-c)*na)*(1-sb)
        mbb = (c*(na*(1-T[0]) + (1-na)*T[1]) + (1-c)*(1-na))*sb
        if return_m:
            return [maa, mab, mba, mbb]
        dpaadt = na*maa -  na * T[0]*(maa + mab)
        dpbbdt = (1-na)*mbb - (1-na)*T[1]*(mba + mbb)
        dpdt = [dpaadt, dpbbdt]
        return dpdt
    t = np.linspace(0, 500, 1000) #np.arange(0, 1000)
    p0 = [sa, sb]
    pa, pb = odeint(model, p0, t)[-1]

    ta, tb = t_from_p((pa, pb))
    maa, mab, mba, mbb = model((pa, pb), 1000, return_m=True)

    dtadpa = 2*(1-pb) / (1 + pa - pb)**2
    dtadpb = 2*pa / (1 + pa - pb)**2
    dtbdpa = 2*pb / (1 + pb - pa)**2
    dtbdpb = 2*(1-pa) / (1 + pb - pa)**2

    dmaadta = c*na*sa
    dmaadtb = -c*sa*(1-na)
    dmabdta = -c*na*(1-sa)
    dmabdtb = c*(1-sa)*(1-na)
    dmbadta = c*na*(1-sb)
    dmbadtb = -c*(1-sb)*(1-na)
    dmbbdta = -c*na*sb
    dmbbdtb = c*(1-na)*sb

    dmaadpa = dmaadta*dtadpa + dmaadtb*dtbdpa
    dmaadpb = dmaadta*dtadpb + dmaadtb*dtbdpb
    dmabdpa = dmabdta*dtadpa + dmabdtb*dtbdpa
    dmabdpb = dmabdta*dtadpb + dmabdtb*dtbdpb
    dmbadpa = dmbadta*dtadpa + dmbadtb*dtbdpa
    dmbadpb = dmbadta*dtadpb + dmbadtb*dtbdpb
    dmbbdpa = dmbbdta*dtadpa + dmbbdtb*dtbdpa
    dmbbdpb = dmbbdta*dtadpb + dmbbdtb*dtbdpb

    nb = 1-na
    dfdpa = na*dmaadpa - na*dtadpa*(maa + mab) - na*ta*(dmaadpa + dmabdpa)
    dfdpb = na*dmaadpb - na*dtadpb*(maa + mab) - na*ta*(dmaadpb + dmabdpb)
    dgdpa = nb*dmbbdpa - nb*dtbdpa*(mba + mbb) - nb*tb*(dmbadpa + dmbbdpa)
    dgdpb = nb*dmbbdpb - nb*dtbdpb*(mba + mbb) - nb*tb*(dmbadpb + dmbbdpb)

    tr = dfdpa + dgdpb
    det = dfdpa*dgdpb - dgdpa*dfdpb
    return tr, det


def plot_fixed_sa_sb(c=.95):
    s_vec = np.round(np.linspace(.5, 1, 51), 2)
    na_vec = [.1, .3, .5, .7 ,.9]

    fig, axs = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    fig2, axs2 = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    fig3, axs3 = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    for na, ax, ax2, ax3 in zip(na_vec, axs, axs2, axs3):
        traces = np.zeros((len(s_vec), len(s_vec)))
        dets = np.zeros((len(s_vec), len(s_vec)))
        typ = np.zeros((len(s_vec), len(s_vec)))
        for i, sa in enumerate(s_vec):
            for j, sb in enumerate(s_vec):
                tr, det = fixed_point_stability(na=na, sa=sa, sb=sb, c=c)
                traces[i, j] = tr
                dets[i, j] = det
                typ[i, j] = tr**2 - 4*det
        traces = pd.DataFrame(traces, columns=s_vec, index=s_vec)
        dets = pd.DataFrame(dets, columns=s_vec, index=s_vec)
        typ = pd.DataFrame(typ, columns=s_vec, index=s_vec)
        sns.heatmap(traces, center=0, ax=ax, vmin=-1, vmax=1)
        sns.heatmap(dets, center=0, ax=ax2, vmin=-1, vmax=1)
        sns.heatmap(typ, center=0, ax=ax3, vmin=-1, vmax=1)
        ax.set_title('na={}'.format(na))
        ax2.set_title('na={}'.format(na))
        ax3.set_title('na={}'.format(na))
        ax.set_xlabel('sb')
        ax.set_ylabel('sa')
        ax2.set_xlabel('sb')
        ax2.set_ylabel('sa')
        ax3.set_xlabel('sb')
        ax3.set_ylabel('sa')
    ax.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    fig.suptitle('Traces of fixed points')
    fig2.suptitle('Determinant of fixed points')
    fig3.suptitle('Type of points')
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig.savefig('traces_sa_sb_cv{}.pdf'.format(int(c*100)))
    fig2.savefig('dets_sa_sb_cv{}.pdf'.format(int(c*100)))
    fig3.savefig('types_sa_sb_cv{}.pdf'.format(int(c*100)))


def plot_fixed_cv_sb(sa=.75):
    s_vec = np.round(np.linspace(.5, 1, 51), 2)
    c_vec = np.round(np.linspace(0, 1, 51), 2)
    na_vec = [.1, .3, .5, .7 ,.9]

    fig, axs = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    fig2, axs2 = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    fig3, axs3 = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    for na, ax, ax2, ax3 in zip(na_vec, axs, axs2, axs3):
        traces = np.zeros((len(c_vec), len(s_vec)))
        dets = np.zeros((len(c_vec), len(s_vec)))
        typ = np.zeros((len(c_vec), len(s_vec)))
        for i, c in enumerate(c_vec):
            for j, sb in enumerate(s_vec):
                tr, det = fixed_point_stability(na=na, sa=sa, sb=sb, c=c)
                traces[i, j] = tr
                dets[i, j] = det
                typ[i, j] = tr**2 - 4*det
        traces = pd.DataFrame(traces, columns=s_vec, index=s_vec)
        dets = pd.DataFrame(dets, columns=s_vec, index=s_vec)
        typ = pd.DataFrame(typ, columns=s_vec, index=s_vec)
        sns.heatmap(traces, center=0, ax=ax, vmin=-1, vmax=1)
        sns.heatmap(dets, center=0, ax=ax2, vmin=-1, vmax=1)
        sns.heatmap(typ, center=0, ax=ax3, vmin=-1, vmax=1)
        ax.set_title('na={}'.format(na))
        ax2.set_title('na={}'.format(na))
        ax3.set_title('na={}'.format(na))
        ax.set_xlabel('sb')
        ax.set_ylabel('c')
        ax2.set_xlabel('sb')
        ax2.set_ylabel('c')
        ax3.set_xlabel('sb')
        ax3.set_ylabel('c')
    ax.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    fig.suptitle('Traces of fixed points')
    fig2.suptitle('Determinant of fixed points')
    fig3.suptitle('Types of fixed points')
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig.savefig('traces_sa_sb_cv{}.pdf'.format(int(c*100)))
    fig2.savefig('dets_sa_sb_cv{}.pdf'.format(int(c*100)))
    fig3.savefig('type_sa_sb_cv{}.pdf'.format(int(c*100)))

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

def plot2_rho_par_re_ds(c=1, alpha=.01):
    sa_vec = np.round(np.linspace(.5, 1, 51), 2)
    sb_vec = np.round(np.linspace(.5, 1, 51), 2)
    na_vec = [1, .3, .5, .7, .9]

    fig, axs = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    for na, ax in zip(na_vec, axs):
        rho_sq = []
        for sa in sa_vec:
            rhos = []
            for sb in sb_vec:
                t = solve2_paa_pbb(na, sa, sb, c, analytic=False)
                #t = t_from_p(p)
                rho = rho_from_t(t)
                rhos.append(rho)
            rho_sq.append(rhos)
        rho_sq = pd.DataFrame(rho_sq, columns=sa_vec, index=sb_vec)
        sns.heatmap(rho_sq, center=1, ax=ax, vmin=0, vmax=2)
        #ax.set_xlabel(r'$s_a$')
        #ax.set_ylabel(r'$s_b$')
    ax.invert_yaxis()
    fig.suptitle('Theo measure of CP, c={}'.format(int(c*100)))
    fig.tight_layout()
    fig.savefig('theo_rho_PA2_rng_ds_c{}.pdf'.format(int(c*100)))


def plot_rho_cv_na(sa=.75, alpha=.01):
    cv_vec = np.round(np.linspace(.5, 1, 11), 2)
    sb_vec = np.round(np.linspace(.5, 1, 11), 2)
    na_vec = [.1, .3, .5, .7, .9]

    fig, axs = plt.subplots(1, len(na_vec), figsize=(2*len(na_vec), 2), sharey=True)
    for na, ax in zip(na_vec, axs):
        rho_sq = []
        for cv in cv_vec:
            rhos = []
            for sb in sb_vec:
                t = solve2_paa_pbb(na, sa, sb, cv)
                #t = t_from_p(p)
                rho = rho_from_t(t)
                #rho = rho_par_re(na, sa, sb, alpha)
                rhos.append(rho)
            rho_sq.append(rhos)
        rho_sq = pd.DataFrame(rho_sq, columns=cv_vec, index=sb_vec)
        sns.heatmap(rho_sq, center=1, ax=ax, vmin=0, vmax=2)
        ax.set_title('na={}'.format(int(na*100)))
        ax.set_xlabel(r'$s_ab$')
    ax.set_ylabel(r'$c$')
    ax.invert_yaxis()
    fig.suptitle('Numerical solutions for core-periphery, \n sa={}'.format(sa))
    fig.tight_layout()
    fig.savefig('theo_rho_cv_na_PA2.pdf')


def plot_sim_rho_par_re_ds(alpha=.01):
    sa_vec = [.5, .6, .7, .8, .9, 1] #np.round(np.arange(100, 201)/201, 2)
    sb_vec = [.5, .6, .7, .8, .9, 1] #np.round(np.arange(100, 201)/201, 2)
    na_vec = [.1, .5, .9]

    N = 1500
    n_iter = 250000
    p0 = [[.015, .015], [.015, .015]]

    fig, axs = plt.subplots(1, len(na_vec), figsize=(6, 2), sharey=True)
    for na, ax in zip(na_vec, axs):
        rho_sq = []
        for sa in sb_vec:
            rhos = []
            for sb in sb_vec:
                p, t, _, rho, _ = am.run_rewiring(N, na, 1, [sa, sb], p0, n_iter, track_steps=n_iter, rewire_type="pa_one", remove_neighbor=True)
                #p = solve_paa_pbb(na, sa, sb)
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
