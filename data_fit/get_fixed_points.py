import sympy as sm
from scipy.integrate import odeint
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
import networkx as nx
import asikainen_model as am
from pandas import DataFrame
from itertools import product

# TODO: test correlations as a general measure
# TODO: check if the measure is alright
# TODO: generalize plots for four datasets
# TODO: add CP measure to plot_evol cases


def cp_correlation(laa, lab, lbb, n, na, a):
    nb = n - na
    N = n*(n-1)
    exy = 2*laa / N + 2*lab*a / N
    ex = 2*(laa + lab + lbb) / N
    ey = na*(na-1) / N + (a*2*nb*na) / N
    sx = ex*(1-ex)
    sy = ey*(1-ey)
    #print(exy, ex, ey)
    corr = (exy - ex*ey) / np.sqrt(sx*sy)
    return corr

def cp_correlation_cont(paa, pbb, na, rho, a):
    # rho is the newtork density rho=2L/N(N-1)
    pab = 1 - paa - pbb
    ex = rho
    ey = na**2 + 2*a*na*(1-na)
    exy = (paa + a*pab)*rho
    sx = ex*(1-ex)
    sy = ey*(1-ey)
    corr = (exy - ex*ey) / np.sqrt(sx*sy)
    return corr

def cp_correlation_cont_correctdens(paa, pbb, na, rho):
    # rho is the newtork density rho=2L/N(N-1)
# HERE WE CORRECT over max possible correlation
    corr = cp_correlation_cont(paa, pbb, na, rho, 0)
    corrs = [cp_correlation_cont(paa, pbb, nai, rho, 0) for nai in np.linspace(0, 1, 100)]
    corrs = [cr if np.abs(cr) <= 1 else 0 for cr in corrs]
    adj_corr = corr / max(corrs)

    return adj_corr

def plot_test_cont(adjust=False, a=0):
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    nas = (.1, .2, .3, .4, .5)
    rhos = (.001, .005, .01, .05, .1)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3), sharex=True, sharey=True)
    na_corr = {}
    for i, na in enumerate(nas):
        for j, rho in enumerate(rhos):
            vals = np.zeros((101, 101))
            vals[:] = np.nan
            for paa, pbb in product(ps, ps):
                if paa + pbb <= 100:
                    if not adjust:
                        corr = cp_correlation_cont(paa/100, pbb/100, na, rho, a)
                    else:
                        corr = cp_correlation_cont_correctdens(paa/100, pbb/100, na, rho)
                    corr = corr if corr <= 1 else np.nan
                    vals[paa, pbb] = corr
                    #vals[pbb, paa] = np.nan
            sns.heatmap(vals, ax=axs[j, i], center=0, vmin=-.2, vmax=1)
            axs[j,i].set_xlabel(r'$P_{bb}$')
            axs[j,i].set_ylabel(r'$P_{aa}$')
            axs[j,i].set_title(r'$n_a=$' + f'{na}\n' + r'$\rho=$' + f'{rho}')
            na_corr[str(na)] = vals
    xticks = axs[j,i].get_xticks()
    axs[j,i].invert_yaxis()
    yticks = axs[j,i].get_yticks()
    axs[j,i].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
    axs[j,i].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
    if not adjust:
        fig.suptitle('Continuous correlation to ideal CP matrix')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_mat_a{}.pdf'.format(a))
    else:
        fig.suptitle('Continuous adjusted correlation to ideal CP matrix')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_mat_adjust.pdf')

    return na_corr, (fig, axs)

def plot_test_discrete():
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    nas = (.1, .2, .3, .4, .5)
    rhos = (.001, .005, .01, .05, .1)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3), sharex=True, sharey=True)
    na_corr = {}
    for i, na in enumerate(nas):
        vals = np.zeros((101, 101))
        vals[:] = np.nan
        for paa, pbb in product(ps, ps):
            if paa + pbb <= 100:
                corr = cp_correlation_cont(paa/100, pbb/100, na, 0)
                vals[paa, pbb] = corr
                #vals[pbb, paa] = np.nan
        #vals = DataFrame(vals,columns=['paa', 'pbb', 'cr'])
        sns.heatmap(vals, ax=axs[i], center=0)
        axs[i].set_xlabel(r'$P_{bb}$')
        axs[i].set_ylabel(r'$P_{aa}$')
        axs[i].invert_yaxis()
        axs[i].set_title(r'$n_a=$' + f'{na}' )
        na_corr[str(na)] = vals
    xticks = axs[i].get_xticks()
    yticks = axs[i].get_yticks()
    axs[i].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
    axs[i].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
    fig.suptitle('Continuous correlation to ideal CP matrix')
    fig.tight_layout()
    fig.savefig('cp_measure_plots/disc_corr_cp_mat.pdf')

def plot_test_cont_rho_na(adjust=False):
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    paas = (1, .75, .5, .25, .1)
    pbbs = (0, .25, .5, .75, .9)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3))
    na_corr = {}
    for i, paa in enumerate(paas):
        for j, pbb in enumerate(pbbs):
            vals = np.zeros((101, 101))
            vals[:] = np.nan
            if paa + pbb <= 1:
                for na, rho in product(ps, ps):
                    if not adjust:
                        corr = cp_correlation_cont(paa, pbb, na/100, rho/100, 0)
                    else:
                        corr = cp_correlation_cont_correctdens(paa, pbb, na/100, rho/100)
                    corr = corr if np.abs(corr) <= 1 else np.nan
                    vals[na, rho] = corr
                    #vals[pbb, paa] = np.nan
                sns.heatmap(vals, ax=axs[i, j], center=0, vmin=-1, vmax=1)
                axs[i,j].set_xlabel(r'$\rho$')
                axs[i,j].set_ylabel(r'$n_{a}$')
                axs[i,j].set_title(r'$P_{aa}=$' + f'{paa}    ' + r'$P_{bb}=$' + f'{pbb}')
                xticks = axs[i,j].get_xticks()
                axs[i,j].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
                axs[i,j].invert_yaxis()
                yticks = axs[i,j].get_yticks()
                axs[i,j].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
            else:
                axs[i,j].axis('off')
    fig.tight_layout()
    if not adjust:
        fig.suptitle('Continuous correlation to ideal CP matrix\n varying ' +r'$n_a$' ' and '+r'$\rho$')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_na_rho.pdf')
    else:
        fig.suptitle('Continuous adjusted correlation to ideal CP matrix\n varying ' +r'$n_a$' ' and '+r'$\rho$')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_corr_cp_na_rho_adjust.pdf')

def plot_test_cont_local_dens(adjust=False, a=0):
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    rho_bbs = (0, .05, .1, .25, .5)
    rho_abs = (0, .05, .1, .25, .5)
    ps = range(101)
    fig, axs = plt.subplots(5, 5, figsize=(5*3, 5*3))
    na_corr = {}
    for i, rho_bb in enumerate(rho_bbs):
        for j, rho_ab in enumerate(rho_abs):
            vals = np.zeros((101, 101))
            vals[:] = np.nan
            for Na, rho_aa in product(ps, ps):
                na = Na / 100
                rho = (rho_aa/100)*na**2 + 2*rho_ab*na*(1-na) + rho_bb*(1-na)**2
                paa = (rho_aa/100)*na**2 / rho if rho > 0 else 0
                pbb = rho_bb*(1-na)**2 / rho if rho > 0 else 0
                if not adjust:
                    corr = cp_correlation_cont(paa, pbb, na, rho, a)
                else:
                    corr = cp_correlation_cont_correctdens(paa, pbb, na, rho)
                corr = corr if np.abs(corr) <= 1 else np.nan
                vals[Na, rho_aa] = corr
                #vals[pbb, paa] = np.nan
            sns.heatmap(vals, ax=axs[i, j], center=0, vmin=-1, vmax=1)
            axs[i,j].set_xlabel(r'$\rho_{aa}$')
            axs[i,j].set_ylabel(r'$n_{a}$')
            axs[i,j].set_title(r'$\rho_{ab}=$' + f'{rho_ab}    ' + r'$\rho_{bb}=$' + f'{rho_bb}')
            xticks = axs[i,j].get_xticks()
            axs[i,j].set_xticklabels([str(np.round(p/100, 2)) for p in xticks])
            axs[i,j].invert_yaxis()
            yticks = axs[i,j].get_yticks()
            axs[i,j].set_yticklabels([str(np.round(p/100, 2)) for p in yticks])
    fig.tight_layout()
    if not adjust:
        fig.suptitle('Continuous correlation to ideal CP matrix\n varying local densitites')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_local_dens_a{}.pdf'.format(a))
    else:
        fig.suptitle('Continuous adjusted correlation to ideal CP matrix\n varying local densitites')
        fig.tight_layout()
        fig.savefig('cp_measure_plots/cont_local_dens_adjust.pdf')


def cp_correlation_alpha(laa, lab, lbb, n, na):
    alphas = np.linspace(0, 1, 201)
    cps = []
    for a in alphas:
        corr = cp_correlation(laa, lab, lbb, n, na, a)
        corr = corr if corr < 1 else 0
        cps.append(corr)
    idx = np.argmax(cps)
    return cps[idx], alphas[idx]

def cp_correlation_cont_alpha(paa, pbb, na, rho):
    alphas = np.linspace(0, 1, 1001)
    cps = []
    for a in alphas:
        corr = cp_correlation_cont(paa, pbb, na, rho, a)
        corr = corr if corr < 1 else 0
        cps.append(corr)
    idx = np.argmax(cps)
    return cps[idx], alphas[idx]


def p_mat_correlation(na, rho):
    paa = np.linspace(0, 1, 101)
    pbb = np.linspace(0, 1, 101)
    vals = []
    for paa, pbb in product(paa, pbb):
        if paa + pbb <= 1:
            corr = cp_correlation_cont(paa, pbb, na, rho, 0)
            it = [paa, pbb, corr]
            vals.append(it)
    vals = DataFrame(vals, columns=['paa', 'pbb', 'cr'])
    return vals


def get_cp_corr(df):
    cs_a, als_a, cs_b, als_b = [], [], [], []
    for i, row in df.iterrows():
        Na, Nb = int(row.N*row.na), int(row.N*(1-row.na))
        corr_a, a = cp_correlation_alpha(row.laa, row.lab, row.lbb, row.N, Na)
        cs_a.append(corr_a); als_a.append(a)
        corr_b, b = cp_correlation_alpha(row.lbb, row.lab, row.laa, row.N, Nb)
        cs_b.append(corr_b); als_b.append(b)
    df['corr_a'] = cs_a
    df['al_a'] = als_a
    df['corr_b'] = cs_b
    df['al_b'] = als_b
    return df




def test_cp_correlation(N, Na, paa, pab, pbb, a):
    net = nx.stochastic_block_model([Na, N-Na], [[paa, pab], [pab, pbb]])
    arr = nx.adj_matrix(net).toarray()

    net_cp = nx.stochastic_block_model([Na, N-Na], [[1, a], [a, 0]])
    arr_cp = nx.adj_matrix(net_cp).toarray()

    laa = arr[0:Na, 0:Na].sum() / 2
    lab = arr[Na:N, 0:Na].sum()
    lbb = arr[Na:N, Na:N].sum() / 2

    corr = np.corrcoef(arr.flatten(), arr_cp.flatten())[0, 1]
    corr_test = cp_correlation(laa, lab, lbb, N, Na, a)

    print(corr, corr_test)



def rewire_fixed_points(c, na, sa, sb):
    paa, pbb = sm.symbols('paa, pbb', negative=False)

    maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
    mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
    mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
    mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

    taa = 2*paa / (paa + 1 - pbb)
    tbb =  2*pbb / (pbb + 1 - paa)

    Paa = na*maa - na*taa*(maa + mab)
    Pbb = (1-na)*mbb - (1-na)*tbb*(mba + mbb)

    PaaEqual = sm.Eq(Paa, 0)
    PbbEqual = sm.Eq(Pbb, 0)

    equilibria = sm.solve((PaaEqual, PbbEqual), paa, pbb)
    equilibria = check_solutions(equilibria)
    equilibria = p_to_t(equilibria)

    return equilibria

def rewire_path(c, na, sa, sb, P, t):

    paa, pbb = P #p_from_G(G0, Na)
    y0 = np.array([paa, pbb])

    def rewire_model(y, t):
        paa, pbb = y
        maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
        mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
        mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
        mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

        taa = 2*paa / (paa + 1 - pbb)
        tbb =  2*pbb / (pbb + 1 - paa)

        dPaa = na*maa - na*taa*(maa + mab)
        dPbb = (1-na)*mbb - (1-na)*tbb*(mba + mbb)

        return np.array([dPaa, dPbb])
    ysol = odeint(rewire_model, y0, t)
    return ysol

def rewire_simul_simple(c, na, sa, sb, P, t, N, L):
    paa, pbb = P
    Na = int(na*N)
    Nb = int((1-na)*N)
    N = Na + Nb
    Laa = L*paa
    Lbb = L*pbb
    Lab = L - Laa - Lbb
    p_sbm = [[2*Laa/(Na*(Na-1)), Lab/(Na*Nb)], [Lab/(Na*Nb), 2*Lbb/(Nb*(Nb-1))]]
    sizes = [Na, Nb]
    G = nx.stochastic_block_model(sizes=sizes, p=p_sbm)
    p_simul = p_from_G(G, Na)
    print(f'p orig: {P}')
    print(f'p simul: {p_simul}')
    i = 0
    while i < t:
        n_link, d_link, _, _, _ = am._rewire_candidates_exact(G, N, Na, c, [sa, sb])
        if d_link and n_link:
            i += 1
            G.add_edge(*n_link)
            G.remove_edge(*d_link)

    P = p_from_G(G, Na)
    return P


def rewire_steps(c, na, sa, sb, P0, N, L, n_steps=10, t_size=5000):
    """
    Returns the number of rewiring steps in a time step (in g), around 2500 rewiring steps per time step
    """
    t = np.linspace(0, n_steps, t_size)
    path = rewire_path(c, na, sa, sb, P0, t)
    rsteps = list(range(0, t_size, int(t_size/n_steps))) + [-1]
    P_steps = [path[i, :] for i in rsteps]
    try:
        n_ests = [1] + rewire_until_p(c, na, sa, sb, P0, N, L, P_steps)
    except:
        n_ests = [None]
    if np.all(n_ests):
        n_ests = [i-j for i, j in zip(n_ests[1:], n_ests[:-1])]
        return n_ests
    else:
        return None


def rewire_until_p(c, na, sa, sb, P0, N, L, P_steps):
    paa, pbb = P0
    Na, Nb = int(na*N), int((1-na)*N)
    N = Na + Nb
    Laa, Lbb = L*paa, L*pbb
    Lab = L - Laa - Lbb
    p_sbm = [[2*Laa/(Na*(Na-1)), Lab/(Na*Nb)], [Lab/(Na*Nb), 2*Lbb/(Nb*(Nb-1))]]
    sizes = [Na, Nb]
    G = nx.stochastic_block_model(sizes=sizes, p=p_sbm)
    p_simul = p_from_G(G, Na)
    i, j = 0, 1
    steps = []
    while j < len(P_steps):
        n_link, d_link, _, _, _ = am._rewire_candidates_exact(G, N, Na, c, [sa, sb])
        if d_link and n_link:
            i += 1
            G.add_edge(*n_link)
            G.remove_edge(*d_link)
            Laa, Lab, Lbb = link_counts(n_link, Na, Laa, Lab, Lbb)
            Laa, Lab, Lbb = link_counts(d_link, Na, Laa, Lab, Lbb, False)
            Paa = Laa / L
            Pbb = Lbb / L
            dist = np.sqrt((Paa - P_steps[j][0])**2 + (Pbb - P_steps[j][1])**2)
            if dist < 10e-3:
                steps.append(i)
                j += 1
            elif i > j*3500:
                print('Real: {}-{}| Sim: {}-{}'.format(P_steps[j][0], P_steps[j][1], Paa, Pbb))
                steps.append(None)
                j += 1
    return steps

def link_counts(link, Na, Laa, Lab, Lbb, add=True):
    src = 'a' if link[0] <= Na else 'b'
    tgt = 'a' if link[1] <= Na else 'b'
    if src == tgt:
        if src == 'a':
            Laa = Laa + 1 if add else Laa - 1
        else:
            Lbb = Lbb + 1 if add else Lbb - 1
    else:
        Lab = Lab + 1 if add else Lab - 1

    return Laa, Lab, Lbb


def growth_steps(c, na, sa, sb, P0, L0, N, n_steps=1000, t_size=10000):
    """
    Returns the number of evolution steps in a time step (usually around 1/2 and evolution step)
    """
    t = np.linspace(0, n_steps, 10000)
    path = growth_path(c, na, sa, sb, P0, L0, t)
    rsteps = list(range(0, 10000, int(10000/n_steps)))
    P_steps = [path[i, 2] for i in rsteps]
    #n_ests, added_seq = grow_until_p(c, na, sa, sb, P0, L0, N, P_steps, m_avg=10)
    n_ests = [i-j for i, j in zip(P_steps[1:], P_steps[:-1])]
    return n_ests


def grow_until_p(c, na, sa, sb, P0, L, N, P_steps, m_avg):
    paa, pbb = P0
    Na, Nb = int(na*N), int((1-na)*N)
    N = Na + Nb
    Laa, Lbb = L*paa, L*pbb
    Lab = L - Laa - Lbb
    G, Na, dist = am.ba_starter(N, na, sa, sb)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, 5))
    for tgt in target_list:
        _ = sources.remove(tgt)

    i, j = 0, 1
    steps = []
    added_seq = []
    m_add = 2*m_avg
    while i < len(sources):
        try:
            counts = am.grow_ba_two(G, sources, target_list, dist, m_avg, c, ret_counts=True, n_i={}, Na=0)
        except:
            import pdb; pdb.set_trace()
        Laa, Lab, Lbb, leftover = add_growth_counts(counts, Laa, Lab, Lbb)
        added = sum(counts) - leftover
        added_seq.append(added)
        i += 1
        L = Laa + Lab + Lbb
        Paa = Laa / L
        Pbb = Lbb / L
        cp_dist = np.sqrt((Paa - P_steps[j][0])**2 + (Pbb - P_steps[j][1])**2)
        if cp_dist < 10e-3:
            steps.append(i)
            j += 1
        elif i > j*3000:
            print('Real: {}-{}| Sim: {}-{}'.format(P_steps[j][0], P_steps[j][1], Paa, Pbb))
            steps.append(None)
            j += 1
    return steps, added_seq


def add_growth_counts(counts, Laa, Lab, Lbb):
    #Classify counts if the added links are: AA, AB, AN, BA, BB, BN
    Laa += counts[0]
    Lab += counts[1] + counts[3]
    Lbb += counts[4]
    leftover = counts[2] + counts[5]
    return Laa, Lab, Lbb, leftover


def rewire_simul_n(c, na, sa, sb, P, t, N, L, n_samp):
    p = []
    for i in range(n_samp):
        print(f'Getting simul {i} / {n_samp}')
        psamp = rewire_simul_simple(c, na, sa, sb, P, t, N, L)
        p.append(psamp)
    return p

def grow_simul_simple(c, na, sa, sb, N, m):
    G, Na, dist = am.ba_starter(N, na, sa, sb)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, 5))
    while len(sources) > 0:
        am.grow_ba_two(G, sources, target_list, dist, m, c, ret_counts=False, n_i={}, Na=0)

    P = p_from_G(G, Na)
    return P

def grow_simul_n(c, na, sa, sb, N, m, n_samp):
    p = []
    for i in range(n_samp):
        print(f'Getting simul {i} / {n_samp}')
        psamp = grow_simul_simple(c, na, sa, sb, N, m)
        p.append(psamp)
    return p


def p_from_G(G, Na):
    paa, pbb, pab = 0, 0, 0
    for edge in G.edges():
        if edge[0] >= Na and edge[1] >= Na:
            pbb += 1
        elif edge[0] < Na and edge[1] < Na:
            paa += 1
        else:
            pab += 1
    tot = paa + pab + pbb
    paa /= tot if tot > 0 else 1
    pbb /= tot if tot > 0 else 1
    return paa, pbb


def check_solutions(equilibria):
    eqs = []
    for sol in equilibria:
        s1 = sol[0]
        s2 = sol[1]
        if s1.is_real and s2.is_real:
            if valid_sol(s1, s2):
                eqs.append((s1, s2))
        elif sm.im(s1) < 1.e-8 and sm.im(s2) < 1.e-8:
            s1 = sm.re(s1); s2 = sm.re(s2)
            if valid_sol(s1, s2):
                eqs.append((s1, s2))
    return eqs

def valid_sol(s1, s2):
    s1_val = number_in_range(s1)
    s2_val = number_in_range(s2)
    s_val = number_in_range(s1 + s2)
    if s1_val and s2_val and s_val:
        return True
    else:
        return False

def number_in_range(num):
    if (num >= 0) and (num <= 1):
        return True
    else:
        return False


def p_to_t(equilibria):
    ts = []
    for ps in equilibria:
        pa, pb = ps[0], ps[1]
        taa = 2*pa / (pa + 1 - pb)
        tbb = 2*pb / (pb + 1 - pa)
        ts.append((taa, tbb))
    return ts


def growth_path(c, na, sa, sb, P, L0, t):
    paa, pbb = P #p_from_G(G0, Na)
    y0 = np.array([paa, pbb, L0])

    def growth_model(y, t):
        paa, pbb, L = y
        maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
        mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
        mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
        mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

        dlaa = na*maa
        dlbb = (1-na)*mbb
        dL = na*(maa + mab) + (1-na)*(mba + mbb)

        dPaa = (dlaa - dL*paa) / L
        dPbb = (dlbb - dL*pbb) / L

        return np.array([dPaa, dPbb, dL])

    ysol = odeint(growth_model, y0, t)
    return ysol


