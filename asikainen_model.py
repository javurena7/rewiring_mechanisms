import networkx as nx
import numpy as np
from collections import defaultdict
import copy
import random

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
    TC 4 - close a triangle by choosing a friend of a friend (correcting by the degree of the friends)
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


def ba_starter(N, fm, h_aa, h_bb):
    Na = int(N * fm)
    minority_nodes = range(Na)
    G = nx.Graph()
    node_attribute = {}

    for n in range(N):
        if n < Na:
            G.add_node(n , color = 'red')
            node_attribute[n] = 'minority'
        else:
            G.add_node(n , color = 'blue')
            node_attribute[n] = 'majority'

    dist = defaultdict(int)
    h_ab = 1 - h_aa
    h_ba = 1 - h_bb
    #create homophilic distance ### faster to do it outside loop ###
    for n1 in range(N):
        n1_attr = node_attribute[n1]
        for n2 in range(N):
            n2_attr = node_attribute[n2]
            if n1_attr == n2_attr:
                if n1_attr == 'minority':
                    dist[(n1,n2)] = h_aa
                else:
                    dist[(n1,n2)] = h_bb
            else:
                if n1_attr == 'minority':
                    dist[(n1,n2)] = h_ab
                else:
                    dist[(n1,n2)] = h_ba


    return G, Na, dist

def grow_ba_one(G, sources, target_list, dist, m):
    """
    BA 1 - Barabasi-Albert model where we pick nodes propto degree * homophily
    """
    source = np.random.choice(sources)
    _ = sources.remove(source)
    targets = _pick_ba_one_targets(G, source, target_list, dist, m)
    if targets != set():
        G.add_edges_from(zip([source] * m, targets))
    target_list.append(source)


def grow_ba_two(G, sources, target_list, dist, m):
    """
    BA 2 - Barabasi-Albert model where we pick nodes propto degree, and accept it with prob homophily
    """

    source = np.random.choice(sources)
    _ = sources.remove(source)
    targets = _pick_ba_two_targets(G, source, target_list, dist, m)
    if targets != set():
        G.add_edges_from(zip([source] * len(targets), targets))
    target_list.append(source)


def _pick_ba_one_targets(G, source, target_list, dist, m):

    target_prob_dict = {}
    for target in target_list:
        target_prob = (dist[(source,target)]) * (G.degree(target) + 0.00001)
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = copy.copy(target_list)
    count_looking = 0
    if prob_sum == 0:
        return targets
    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:
                targets.add(k)
                target_list_copy.remove(k)
                break
    return targets


def _pick_ba_two_targets(G, source, target_list, dist, m):
    """
    Pick set of new neighbors for node source via second BA model.
    Here, a candidate is selected with probability proportional to degree, and
    accepted with probability homophily (or not joined). The process is run m times.
    -----
    G: networkx graph
    source: incoming node for which neighbors are selected
    target_list: list of possible neighbors
    dist: dict of homophilies
    m: number of timess the experiment is run.
    """
    target_prob_dict = {}
    for target in target_list:
        target_prob =  G.degree(target) + 0.00001
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = copy.copy(target_list)
    count_looking = 0
    if prob_sum == 0:
        return targets
    candidates = set()
    while len(candidates) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:
                candidates.add(k)
                if random.random() < dist[(source, k)]:
                    targets.add(k)
                    target_list_copy.remove(k)
                break
    return targets


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
    p_aa, p_ab, p_bb = get_l(G, Na)
    n_edges = float(G.number_of_edges())
    p_aa /= n_edges if n_edges > 0 else 1
    p_ab /= n_edges if n_edges > 0 else 1
    p_bb /= n_edges if n_edges > 0 else 1
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

def get_l(G, Na):
    """
    Return total number of links between groups
    """
    l_aa, l_ab, l_bb = 0, 0, 0
    for edge in G.edges():
        if edge[0] >= Na:
            if edge[1] >= Na:
                l_bb += 1
            else:
                l_ab += 1
        else:
            if edge[1] >= Na:
                l_ab += 1
            else:
                l_aa += 1
    return l_aa, l_ab, l_bb

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


def run_growing(N, fm, c, bias, p0, n_iter, track_steps=500, rewire_type="ba_two", remove_neighbor=True, m=2):
    """
    Run a Barabasi-Albert model for growing a network
    rewire_type: (str) ba_one, ba_two
    WE DONT USE c, p0, n_iter, remove_neighbor
    WE USE m=2
    """
    #m = 2
    v_types = ["ba_one", "ba_two"]
    assert rewire_type in v_types, "Add valid rewire type"
    rewire_type = 'grow_' + rewire_type
    grow_links = eval(rewire_type)
    #G, Na = random_network(N, fm, p0)
    h_aa, h_bb = bias
    G, Na, dist = ba_starter(N, fm, h_aa, h_bb)
    sources = list(range(N))
    target_list = list(np.random.choice(sources, m, replace=False))
    for tgt in target_list:
        _ = sources.remove(tgt)
    P = defaultdict(list)
    K = defaultdict(list)
    for i in range(N-m):
        grow_links(G, sources, target_list, dist, m)
        if i % track_steps == 0:
            p = get_l(G, Na) # At some point this has also been get_p
            P['l_aa'].append(p[0])
            P['l_ab'].append(p[1])
            P['l_bb'].append(p[2])
            Ka, Kb = total_degree(G, Na)

            K['Ka'].append(Ka)
            K['Kb'].append(Kb)

        if i == int(.95 * n_iter):
            p_95 = get_p(G, Na)
            t_95 = get_t(p_95)

    p = get_p(G, Na)
    t = get_t(p)
    converg_d = .5 * (np.abs(t[0] - t_95[0]) + np.abs(t[1] - t_95[1]))
    rho = measure_core_periph(*t)
    return p, t, (P, K), rho, converg_d

def total_degree(G, Na):
    Ka, Kb = 0, 0
    for i in range(G.number_of_nodes()):
        if i < Na:
            Ka += G.degree(i)
        else:
            Kb += G.degree(i)
    return Ka, Kb


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

