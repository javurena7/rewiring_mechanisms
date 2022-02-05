import sympy as sm
from scipy.integrate import odeint
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
import networkx as nx
import asikainen_model as am

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
    while i < 2*t:
        n_link, d_link, _, _, _ = am._rewire_candidates_exact(G, N, Na, c, [sa, sb])
        if d_link and n_link:
            i += 1
            G.add_edge(*n_link)
            G.remove_edge(*d_link)

    P = p_from_G(G, Na)
    return P

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
