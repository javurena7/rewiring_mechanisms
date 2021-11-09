import networkx as nx
import numpy as np
import copy
import random
from collections import defaultdict
import matplotlib.pyplot as plt; plt.ion()
"""
Rewiring model for two-part kernel (with prob c pref attch, (1-c) random rewiring)
"""

def rewire_network(G, N, c, m, n_i={}, grow=True):
    """
    PA one - create new edge by following a link from a random node
    If grow=False, then n_i can be empty, otherwise it's the degree distribution
    """
    new_links = set()
    del_links = set()
    for _ in range(m):
        n_link, d_link = _rewire_candidates_exact(G, N, c)
        if d_link and n_link:
            diff_link = True #set(d_link) != set(n_link)
            #union = del_links.union(new_links)
            d_link_obs = False #(d_link in union) or ((d_link[1], d_link[0]) in union)
            n_link_obs = False #(n_link in union) or ((n_link[1], n_link[0]) in union)
            if not d_link_obs and not n_link_obs and diff_link:
                new_links.add(n_link)
                del_links.add(d_link)

    x_i = target_degrees(G, new_links)
    if grow:
        G_old = G.copy()
        del_links = sorted(del_links, key=lambda x: x[0])
        new_links = sorted(new_links, key=lambda x: x[0])
        n_i = add_target_counts(n_i, new_links, del_links, G) #Note, for n[i+1]
        for nlink in new_links:
            G.add_edge(*nlink)
        for dlink in del_links:
            if G.has_edge(*dlink):
                G.remove_edge(*dlink)
        return x_i, n_i
    else:
        return x_i


def rewire_network_stepxstep(G, N, c, m, n_i={}, grow=True):
    """
    PA one - create new edge by following a link from a random node
    If grow=False, then n_i can be empty, otherwise it's the degree distribution
    """

    x_i = {}
    n_i_ = n_i
    for _ in range(m):
        n_link, d_link = _rewire_candidates_exact(G, N, c)
        if d_link and n_link:
            tgt_deg = G.degree(n_link[1])
            x_i[tgt_deg] = x_i.get(tgt_deg, 0) + 1

            if grow:
                n_i[tgt_deg+1] = n_i.get(tgt_deg+1, 0) + 1
                n_i[tgt_deg] = n_i[tgt_deg] - 1
                del_deg = G.degree(d_link[1])
                n_i[del_deg] = n_i[del_deg] - 1
                n_i[del_deg-1] = n_i.get(del_deg-1, 0) + 1

                n_i_ = {k: v for k, v in n_i.items() if v > 0}
                G.add_edge(*n_link)
                G.remove_edge(*d_link)
    if grow:
        return x_i, n_i_
    else:
        return x_i


def _rewire_candidates(G, N, c):
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            newNode = np.random.randint(N)
            while G.degree(newNode) < 1:
                newNode = np.random.randint(N)
            endNode = np.random.choice(list(G.neighbors(newNode)))
        else:
            endNode = np.random.randint(N)

        if endNode != startNode: # and not G.has_edge(startNode, endNode):
            sN_neighs = list(G.neighbors(startNode))
            del_neigh = np.random.choice(sN_neighs)
            d_link = (startNode, del_neigh)
            n_link = (startNode, endNode)
        else:
            n_link = ()
            d_link = ()
    else:
        n_link = ()
        d_link = ()

    return n_link, d_link

def _rewire_candidates_exact(G, N, c):
    startNode = np.random.randint(N)

    if G.degree(startNode) > 0:
        neighs = set(G.neighbors(startNode))
        fset = set(range(N))
        fset = fset.difference(neighs)
        fset = fset.difference(set([startNode]))
        if np.random.random() < c:
            target_prob_list = []
            target_list = []
            for target in fset:
                tgt_pr = G.degree(target)
                target_prob = tgt_pr if tgt_pr > 0 else 0.00001
                target_prob_list.append(target_prob)
                target_list.append(target)
            probs = np.array(target_prob_list) / sum(target_prob_list)
            endNode = np.random.choice(target_list, p=probs)
        else:
            endNode = np.random.choice(list(fset))
        delNode = np.random.choice(list(neighs))
        n_link = (startNode, endNode)
        d_link = (startNode, delNode)
    else:
        n_link, d_link = (), ()
    return n_link, d_link



def get_tgt_degree(G, links):
    """
    each entry is of the formart deg: count (so count new edges of degree deg)
    """
    x_i = {}

    for (src, tgt) in links:
        tgt_deg = G.degree(tgt)
        x_i[tgt_deg] = x_i.get(tgt_deg, 0) + 1
    return x_i

def add_target_counts(n_0, new_links, del_links, G):
    # NOTE this adds counts assuming the number of added links is the same as deleted
    n_i = copy.deepcopy(n_0)
    # Note, we need a list of nodes so that we can get the total change chg,
    # a node will go from degree deg to deg + chg (and there will be one less
    # node of degree deg

    node_chg = defaultdict(int)
    get_node_change(node_chg, new_links, idx=1, act='add')
    get_node_change(node_chg, del_links, idx=1, act='del')
    for node, chg in node_chg.items():
        if chg != 0:
            deg = G.degree(node)
            n_i[deg + chg] = n_i.get(deg + chg, 0) + 1
            n_i[deg] = n_i.get(deg, 0) - 1

    # TODO: check what to do if there are more deleted links than the degree

    clean_ni = {key: val for key, val in n_i.items() if val > 0}
    return clean_ni

def get_node_change(n_dic, links, idx, act='add'):
    """
    n_dic: defaultdict where to store the change
    links: set of links where a change is recorded
    idx: index of links that we want to change, may be 0, 1 or 'all'
    act: action to perform, 'add' for new links, 'del' for deleted links
    """
    if idx in [0, 1]:
        nodes = [link[idx] for link in links]
    elif idx == 'all':
        nodes = [node for link in links for node in link]
    else:
        raise NameError("Wrong idx arg")

    if act == 'add':
        for node in nodes:
            n_dic[node] += 1
    elif act == 'del':
        for node in nodes:
            n_dic[node] -= 1
    else:
        raise NameError("Wrong action arg")

def target_degrees(G, new_links):
    x_i = {}
    for link in new_links:
        if not G.has_edge(*link):
            tgt_deg = G.degree(link[1])
            x_i[tgt_deg] = x_i.get(tgt_deg, 0) + 1
    return x_i


def _get_m(m_dist, N):
    if m_dist[0] == 'poisson':
        m_vals = np.random.poisson(m_dist[1], N)
    return m_vals

def random_network(N, p):
    G = nx.erdos_renyi_graph(N, p)
    return G

def run_rewiring(N, cval, n_iter=5000, p=.01, m_dist=['poisson', 20], return_net=False):
    G = random_network(N, p)
    m_vals = _get_m(m_dist, n_iter)
    x = {}
    n_i = deg_dist(G)
    n = {0: n_i}
    for i, m in enumerate(m_vals):
        #if i % 500 == 0:
        #    print(i)
        x_i, n_i = rewire_network_stepxstep(G, N, cval, m, n_i, grow=True)
        x[i] = x_i
        n[i+1] = n_i
    if return_net:
        return G
    else:
        return x, n

def get_ni(G, Na):
    n_i = {'na': {}, 'nb': {}}
    for i in range(len(G)):
        deg = G.degree(i)
        if i < Na:
            n_i['na'][deg] = n_i['na'].get(deg, 0) + 1
        else:
            n_i['nb'][deg] = n_i['nb'].get(deg, 0) + 1
    return n_i


def _lik_step_t_probs(c, nt, xt):

    llik = 0
    P_denom = sum([n*k for k, n in nt.items()])
    U_denom = sum(nt.values())
    for i, (k, xk) in enumerate(xt.items()):
        na_k = nt.get(k, 0)
        p_k = c * (na_k * k / P_denom)
        p_k += (1-c) * na_k / U_denom
        llik += xk * np.log(p_k) if p_k > 0 else 0

    return llik

def loglik(c, n, x):
    llik = 0
    for t, xt in x.items():
        if t > 100:
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

def max_c_estims(N=800, n_c=21):
    cvals = np.linspace(0.0, 1., n_c)
    cests = []

    for cval in cvals:
        print('Real: {}'.format(cval))
        x, n = run_rewiring(N, cval, n_iter=1000)
        vals = loglik_c(n, x)
        cests.append(np.argmax(vals)/100)
    return cvals, cests


def plot_cvals(vals, ests):
    fig, ax = plt.subplots()
    ax.plot(vals, '-')
    ax.plot(ests, '.')
    ax.set_xlabel('C')
    ax.set_xticklabels(vals)
    ax.set_ylabel('Estim. C')
    ax.set_title('Rewiring model without homophily, \nC estimates')
    fig.tight_layout()
    fig.savefig('plots/rewiring_nohom_c_estim.pdf')


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
    counter = {}
    for node, deg in list(G.degree):
        counter[deg] = counter.get(deg, 0) + 1
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

def test_likelihood(cval=.8, n_samp=1000, N=800, n0_iter=5000, p0=.01):
    print('Getting network')
    G = run_rewiring(N=N, cval=cval, n_iter=n0_iter, p=p0, return_net=True) #tester_starter(c0, N)
    ddist = deg_dist(G)
    m_vals = _get_m(['poisson', 40], n_samp)

    probs = get_probs(cval, ddist)
    exp_probs = defaultdict(int)
    target_list = list(range(N))
    print('Getting probabilites')
    for m in m_vals:
        x_i = rewire_network_stepxstep(G, N, cval, m, n_i={}, grow=False)
        expected_probs(x_i, exp_probs)
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

