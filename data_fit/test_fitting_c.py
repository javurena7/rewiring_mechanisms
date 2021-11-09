import networkx as nx
import numpy as np
import copy
import random
from collections import defaultdict
import matplotlib.pyplot as plt; plt.ion()
"""
Growing model for two-part kernel (with prob c pref attch, (1-c) random rewiring)
"""

def grow_network(G, sources, target_list, cval, m, n_i = {}, grow=True):
    source = np.random.choice(sources)
    _ = sources.remove(source)
    #targets = _pick_targets(G, source, target_list, m, cval)
    targets = _pick_targets_exact(G, source, target_list, m, cval)
    x_i = target_degrees(G, targets)
    n_i = add_target_counts(n_i, x_i, len(targets))
    if grow:
        G.add_edges_from(zip([source]* len(targets), targets))
    target_list.append(source)
    return x_i, n_i

def _pick_targets(G, source, target_list, m, cval):
    target_list_copy = copy.copy(target_list)
    #candidates = _choose_random_targets(m, cval, target_list_copy)
    #m_ba = m - len(candidates)
    target_prob_dict = {}
    for target in target_list_copy:
        tgt_pr = G.degree(target)
        target_prob = tgt_pr if tgt_pr > 0 else  0.000001
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())
    count_looking = 0
    targets = set()
    if prob_sum == 0:
        return targets
    N_sum = len(target_list_copy)
    #targets = set(candidates)
    for _ in range(m):
        count_looking += 1
        if count_looking > len(G):
            break
        rand_num = random.random()
        cumsum = 0.0
        if random.random() < cval:
            for k in target_list_copy:
                cumsum += float(target_prob_dict[k]) / prob_sum
                if rand_num < cumsum:
                    prob_sum -= target_prob_dict[k]
                    _ = target_prob_dict.pop(k)
                    N_sum -= 1
                    target_list_copy.remove(k)
                    targets.add(k)
                    break
        else:
            cumsum = 1 / N_sum
            #for i, k in enumerate(target_list_copy):
            k = np.random.choice(target_list_copy, replace=False)
                #cumsum += i / N_sum
                #if rand_num < cumsum:
            prob_sum -= target_prob_dict[k]
            _ = target_prob_dict.pop(k)
                #    N_sum -= 1
                #    target_list_copy.remove(k)
            targets.add(k)
            target_list_copy.remove(k)
                    #break
    return targets


def _pick_targets_exact(G, source, target_list, m, cval):
    target_list_copy = copy.copy(target_list)
    candidates = set()
    target_prob_list = []
    targets = set()
    for target in target_list_copy:
        tgt_pr = G.degree(target)
        target_prob = tgt_pr if tgt_pr > 0 else 0.00001
        target_prob_list.append(target_prob)
    for i in range(m):
        probs = np.array(target_prob_list) / sum(target_prob_list)
        if random.random() < cval:
            k = np.random.choice(target_list_copy, p=probs, replace=False)
        else:
            k = np.random.choice(target_list_copy, replace=False)
        targets.add(k)
        prob_idx = target_list_copy.index(k)
        target_list_copy.remove(k)
        target_prob_list.pop(prob_idx)
    return targets


def _choose_random_targets(m, cval, target_list_copy):
    m_rand = np.random.binomial(m, 1 - cval)
    if len(target_list_copy) >= m_rand:
        candidates = np.random.choice(target_list_copy, m_rand, replace=False)
    else:
        candidates = copy.copy(target_list_copy)
    for k in target_list_copy:
        target_list_copy.remove(k)
    return candidates

def target_degrees(G, targets):
    x_i = {}
    for tg in targets:
        tgt_deg = G.degree(tg)
        x_i[tgt_deg] = x_i.get(tgt_deg, 0) + 1
    return x_i

def add_target_counts(n_0, x_i, tgt_len):
    n_i = copy.deepcopy(n_0)
    for deg, cnt in x_i.items():
        n_i[deg] = max([n_i.get(deg, 0) - cnt, 0])
        n_i[deg+1] = n_i.get(deg+1, 0) + cnt
    n_i[tgt_len] = n_i.get(tgt_len, 0) + 1

    clean_ni = {key: val for key, val in n_i.items() if val > 0}
    return clean_ni

def _get_m(m_dist, N, m0):
    if m_dist[0] == 'poisson':
        m_vals = np.random.poisson(m_dist[1], N - m0)
    cm = []
    for i, m in enumerate(m_vals):
        cm.append(min([m, i+m0]))
    return cm

def run_growing(N, cval, m_dist=['poisson', 40], return_net=False):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    m0 = 2 * m_dist[1]
    m_vals = _get_m(m_dist, N, m0)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, m0, replace=False))
    for node in target_list:
        for n2 in target_list:
            if (np.random.rand() < .5) and (node != n2):
                G.add_edge(node, n2)

    for tgt in target_list:
        _ = sources.remove(tgt)
    x = {}
    n_i = {}
    n = {0: n_i}

    for i, m in enumerate(m_vals):
        x_i, n_i = grow_network(G, sources, target_list, cval, m, n_i)
        x[i] = x_i
        n[i+1] = n_i
    if return_net:
        return G
    else:
        return x, n

def _lik_step_t_probs(c, nt, xt):

    llik = 0
    P_denom = sum([n*k for k, n in nt.items()])
    U_denom = sum(nt.values())
    N = sum(nt.values())
    avg_deg = np.mean([k for k in nt.keys()])
    factor = (P_denom-N-1)/P_denom if P_denom > 0 else 0
    for i, (k, xk) in enumerate(xt.items()):
        na_k = nt.get(k, 0)
        p_k = c * ((na_k * (k) / (P_denom))) * (1-na_k / U_denom)
        na_k = nt.get(k, 0)
        p_k += (1-c) * na_k / U_denom * (1 - na_k * k / P_denom)
        llik += xk * np.log(p_k) if p_k > 0 else 0

    return llik

def _c_estim_t(nt, xt):

    llik = 0
    P_denom = sum([n*k for k, n in nt.items()])
    U_denom = sum(nt.values())
    pk_tot = sum(xt.values())
    a, b, c = 0, 0, 0
    for i, (k, x_k) in enumerate(xt.items()):
        na_k = nt.get(k, 0)
        P_k = na_k * k / P_denom
        U_k = na_k / U_denom
        D_k = P_k - U_k

        c += x_k*D_k / (.5*D_k + U_k) if (U_k + .5*D_k > 0) else 0
        b += x_k*D_k**2 / (.5*D_k + U_k)**2 if (U_k + .5*D_k > 0) else 0
        #a += x_k*D_k**3 / U_k**3 if (U_k + .5*D_k > 0) else 0


    return b, c

def c_estim(n, x):
    a, b, c = 0, 0, 0
    for t, xt in x.items():
        if t > 1:
            nt = n[t]
            bt, ct = _c_estim_t(nt, xt)
            #a += at
            b += bt
            c += ct
    cval = (c+.5*b)/b
    return cval


def loglik(c, n, x):
    llik = 0
    t2 =  len(x) / 2
    for t, xt in x.items():
        if t > t2:
            nt = n[t]
            llik += _lik_step_t_probs(c, nt, xt)
    return llik

def step_model(n, x):
    cs = np.linspace(0, 1, 50)
    cvals = []
    for t, xt in x.items():
        if t > 2:
            nt = n[t]
            lliks = [_lik_step_t_probs(cv, nt, xt) for cv in cs]
            cval = np.argmax(lliks) / 50
            cvals.append(cval)
    return cvals


def loglik_c(n, x):
    vals = [loglik(cv, n, x) for cv in np.linspace(0, 1, 100)]
    print(np.argmax(vals)/100)
    return vals

def max_c_estims(n_c=21):
    cvals = np.linspace(0.0, 1, n_c)
    cests = []

    for cval in cvals:
        print(cval)
        x, n = run_growing(2000, cval, ['poisson', 40])
        vals = loglik_c(n, x)
        cests.append(np.argmax(vals)/100)
    return cvals, cests



def tester_starter(c0 = .2, N=800):
    G = nx.barabasi_albert_graph(N, 30)
    for node in G.nodes():
        rewirings = []
        neighs = G[node]
        for neigh in neighs:
            if np.random.rand() < c0:
                cand = np.random.choice(range(N))
                if cand not in neighs:
                    rewirings.append([neigh, cand])

        for pair in rewirings:
            G.remove_edge(node, pair[0])
            G.add_edge(node, pair[1])
    return G

def deg_dist(G):
    counter = defaultdict(int)
    for node, deg in list(G.degree):
        counter[deg] += 1
    return counter

def get_probs(cval, ddist):
    probs = {}
    P_tot = sum([k*n_k for k, n_k in ddist.items()])
    U_tot = sum(ddist.values())
    for k, n_k in ddist.items():
        probs[k] = cval * k * n_k / P_tot + (1-cval) * n_k / U_tot
    return probs

def expected_probs(x_i, exp_probs):
    for k, val in x_i.items():
        exp_probs[k] += val

def test_likelihood(cval=.8, c0=.2, n_samp=1000, N=800):
    G = run_growing(N=N, cval=cval, return_net=True)#tester_starter(c0, N)
    ddist = deg_dist(G)
    m_vals = _get_m(['poisson', 40], n_samp, 100)

    probs = get_probs(cval, ddist)
    exp_probs = defaultdict(int)
    target_list = list(range(N))
    for m in m_vals:
        sources = [N+1]
        #x_i, _ = grow_network(G, sources, target_list, cval, m, n_i = {}, grow=False)
        targets = _pick_targets(G, N+1, target_list, int(m), cval)
        x_i = target_degrees(G, targets)
        expected_probs(x_i, exp_probs)
        #target_list.remove(N+1)
    exp_sum = sum(exp_probs.values())
    exp_probs = {k: val / exp_sum for k, val in exp_probs.items()}
    return probs, exp_probs

def plot_expected_probs(probs, exp_probs):
    max_k = max([max(probs.keys()), max(exp_probs.keys())])
    ks = list(range(max_k))
    y_probs = [probs.get(k, 0) for k in ks]
    y_exp = [exp_probs.get(k, 0) for k in ks]

    fig, ax = plt.subplots()
    ax.plot(ks, y_probs, '.', label='Probability', alpha=.5)
    ax.plot(ks, y_exp, '.', label='Observed Fraction', alpha=.5)
    #ax.plot(y_probs, y_probs, alpha=.5)
    #ax.plot(y_probs, 2*np.array(y_probs), '--', alpha=.5)
    #ax.plot(y_probs, y_exp, '.')

    ax.legend()
    fig.tight_layout()
    return fig, ax


