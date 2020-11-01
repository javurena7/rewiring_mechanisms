import networkx as nx
import numpy as np
from collections import defaultdict

def random_network(N, fm, p):
    min_nodes = list(range(int(N * fm)))
    Na = len(min_nodes)
    G = nx.stochastic_block_model([Na, N-Na], p)
    return G, Na


def rewire_tc_one(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 1 - close a triangle by choosing a friend of a friend
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            neighbors = list(G.neighbors(startNode))
            newNode = np.random.choice(neighbors)

            neighbors = list(G.neighbors(newNode))
            endNode =  np.random.choice(neighbors)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)

def rewire_tc_two(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 2 - Close a triangle by choosing two neighbors areound a startnode
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 1:
        if np.random.random() < c:
            neighbors = list(G.neighbors(startNode))
            newNodeA, newNodeB = np.random.choice(neighbors, size=2, replace=False)
        else:
            newNodeB = np.random.randint(N)
            newNodeA = startNode
        accept = False
        if (newNodeA != newNodeB) and (not G.has_edge(newNodeA, newNodeB)):
            if newNodeA < Na:
                if newNodeB < Na:
                    if np.random.random() < bias[0]:
                        accept = True
                else:
                    if np.random.random() > bias[0]:
                        accept = True
            else:
                if newNodeB > Na:
                    if np.random.random() < bias[1]:
                        accept = True
                else:
                    if np.random.random() > bias[1]:
                        accept = True

        if accept:
            if remove_neighbor:
                remove_random_neighbor(G, startNode)
            else:
                remove_random_edge(G, edgelist, N_edge)
                edgelist.append([newNodeA, newNodeB])
            G.add_edge(newNodeA, newNodeB)


def rewire_pa_one(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge): #formerly rewire_links
    """
    PA one - create new edge by following a link from a random node
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            newNode = np.random.randint(N)
            while G.degree(newNode) < 1:
                newNode = np.random.randint(N)
            endNode = np.random.choice(list(G.neighbors(newNode)))
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)


def _get_candidates_pa_two(G, N):

    newNode = np.random.randint(N)
    while G.degree(newNode) < 1:
        newNode = np.random.randint(N)
    midNode = np.random.choice(list(G.neighbors(newNode)))
    if G.degree(midNode) < 1:
        return _get_candidates_pa_two(G)
    else:
        endNode = np.random.choice(list(G.neighbors(midNode)))
        return endNode


def rewire_pa_two(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    PA one - create new edge by following two links from a random node
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            endNode = _get_candidates_pa_two(G, N)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)

def rewire_tc_three(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 3 - close a triangle by choosing a random neigbor from the set of second neighbors around a link
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 1:
        if np.random.random() < c:
            second_neigh = list()
            for neigh in G.neighbors(startNode):
                second_neigh += list(G.neighbors(neigh))
            second_neigh = np.unique(second_neigh)
            endNode = np.random.choice(second_neigh)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)


def rewire_tc_four(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge):
    """
    TC 4 - close a triangle by choosing a friend a friend of a friend (correcting by the degree of the friends)
    """
    startNode = np.random.randint(N)
    if G.degree(startNode) > 0:
        if np.random.random() < c:
            neighbors = list(G.neighbors(startNode))
            degrees = np.array([1./G.degree(neigh) for neigh in neighbors])
            p = degrees / sum(degrees)
            newNode = np.random.choice(neighbors, p=p)

            neighbors = list(G.neighbors(newNode))
            degrees = np.array([1./G.degree(neigh) for neigh in neighbors])
            p = degrees / sum(degrees)
            endNode =  np.random.choice(neighbors, p=p)
        else:
            endNode = np.random.randint(N)
        _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge)


def _accept_edge(startNode, endNode, G, edgelist, Na, bias, remove_neighbor, N_edge):
    accept = False
    if (startNode != endNode) and (not G.has_edge(startNode, endNode)):
        if startNode < Na:
            if endNode < Na:
                if np.random.random() < bias[0]:
                    accept = True
            else:
                if np.random.random() > bias[0]:
                    accept = True
        else:
            if endNode > Na:
                if np.random.random() < bias[1]:
                    accept = True
            else:
                if np.random.random() > bias[1]:
                    accept = True

    if accept:
        if remove_neighbor:
            remove_random_neighbor(G, startNode)
        else:
            remove_random_edge(G, edgelist, N_edge)
        edgelist.append([startNode, endNode])
        G.add_edge(startNode, endNode)


def remove_random_neighbor(G, startNode):
    lostNode = np.random.choice(list(G.neighbors(startNode)))
    G.remove_edge(startNode, lostNode)


def remove_random_edge(G, edgelist, N_edge):
    #N_edge = len(edgelist)
    edge = np.random.randint(N_edge)
    G.remove_edge(*edgelist.pop(edge))
    #if len(edgelist) < 1:
    #    for edge in np.random.permutation(G.edges):
    #        edgelist.append(edge)
    #edge = edgelist.pop()
    #G.remove_edge(*edge)


def get_p(G, Na):
    p_aa, p_ab, p_bb = 0, 0, 0
    for edge in G.edges():
        if edge[0] >= Na:
            if edge[1] >= Na:
                p_bb += 1
            else:
                p_ab += 1
        else:
            if edge[1] >= Na:
                p_ab += 1
            else:
                p_aa += 1
    n_edges = float(G.number_of_edges())
    p_aa /= n_edges
    p_ab /= n_edges
    p_bb /= n_edges
    return p_aa, p_ab, p_bb

def get_t(p):
    try:
        taa = 2*p[0] / (2*p[0] + p[1])
    except:
        taa = 0
    try:
        tbb = 2*p[2] / (2*p[2] + p[1])
    except:
        tbb = 0
    return taa, tbb


def measure_core_periph(taa, tbb, a=.01):
    """
    Measure core-periphery structure with the T-matrix.
    Where alpha is the prob. of finind a link between groups
    """

    rho_a = np.sqrt((taa - 1/(1+a))**2 + (1-taa + a/(1+a))**2 + (1-tbb - 1)**2 + (tbb)**2)
    rho_b = np.sqrt((tbb - 1/(1+a))**2 + (1-tbb + a/(1+a))**2 + (1-taa - 1)**2 + (taa)**2)

    return rho_a, rho_b

def run_rewiring(N, fm, c, bias, p0, n_iter, track_steps=500, rewire_type="tc_two", remove_neighbor=True):
    """
    Run the Asikainen model with PA instead of triadic closure
    rewire_type: (str) pa_one, pa_two, tc_one, tc_two, tc_three
    """
    v_types = ["pa_one", "pa_two", "tc_one", "tc_two", "tc_three", "tc_four"]
    assert rewire_type in v_types, "Add valid rewire type"
    rewire_type = 'rewire_' + rewire_type
    rewire_links = eval(rewire_type)
    G, Na = random_network(N, fm, p0)
    edgelist = list(G.edges) if not remove_neighbor else []
    N_edge = len(edgelist)
    P = defaultdict(list)
    for i in range(n_iter):
        rewire_links(G, N, Na, c, bias, remove_neighbor, edgelist, N_edge)
        if i % track_steps == 0:
            p = get_p(G, Na)
            P['p_aa'].append(p[0])
            P['p_ab'].append(p[1])
            P['p_bb'].append(p[2])
        if i == int(.95 * n_iter):
            p_95 = get_p(G, Na)
            t_95 = get_t(p_95)

    p = get_p(G, Na)
    t = get_t(p)
    converg_d = .5 * (np.abs(t[0] - t_95[0]) + np.abs(t[1] - t_95[1]))
    rho = measure_core_periph(*t)
    return p, t, P, rho, converg_d


def theo_random_rewiring(na, s):
    """
    Probability of
    """
    # prob of creating a link
    plink = lambda na, s: na**2*s + 2*(na)*(1-na)*(1-s) + (1-na)**2*s
    paa = na ** 2 * s / plink
    pbb = (1 - na)**2 * s / plink
    pab = 2 * na * (1-na) * (1-s) / plink

    return paa, pab, pbb

