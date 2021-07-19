import asikainen_model as am
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

N = 2500
Na = 250
haa = 1
hbb = 1
Nb = N - Na
fm = Na / N
m = 50
p = [[.05, .01], [.01, .05]]
G = nx.stochastic_block_model([Na, Nb], p)

source_a = 0
source_b = N / 2 + 5

G.remove_node(source_a)
G.remove_node(source_b)
target_list = list(G.nodes())
G.add_node(source_a)
G.add_node(source_b)

_, _, dist = am.ba_starter(N, fm, haa, hbb)

def total_degree(G, Na):
    Ka, Kb = 0, 0
    for i in range(G.number_of_nodes()):
        if i < Na:
            Ka += G.degree(i)
        else:
            Kb += G.degree(i)
    return Ka, Kb

def get_ps(G, Na, haa, hbb):
    Ka, Kb = total_degree(G, Na)
    di = {}
    di['Paa'] = haa * Ka / (Ka + Kb)
    di['Pab'] = (1-haa) * Kb / (Ka + Kb)
    di['Pan'] = (1-haa) * Ka / (Ka + Kb) + haa * Kb / (Ka + Kb)
    di['Pbb'] = hbb * Kb / (Ka + Kb)
    di['Pba'] = (1 - hbb) * Ka / (Ka + Kb)
    di['Pbn'] = (1-hbb) * Kb / (Ka + Kb) + hbb * Ka / (Ka + Kb)
    return di

def get_kds(G, fm, Na, haa, hbb, m=m):
    #fm = Na / G.number_of_nodes()
    di = {}
    Ka, Kb = total_degree(G, Na)
    Kt = Ka + Kb
    di['Kad'] = m * (fm * (2*haa*Ka/Kt + (1-haa)*Kb/Kt) + (1-fm)*(1-hbb)*Ka/Kt)
    di['Kbd'] = m * ((1-fm) * (2*hbb*Kb/Kt + (1-hbb)*Ka/Kt) + fm*(1-haa)*Kb/Kt)
    return di


def run_a(G, Na=Na, n=1000, source=source_a):
    paa, pab, pan = 0, 0, 0
    for _ in range(n):
        target = am._pick_ba_two_targets(G, source, target_list, dist, 1)
        if len(target) > 0:
            tgt = target.pop()
            if tgt < Na:
                paa += 1
            else:
                pab += 1
        else:
            pan += 1
    res = {'paa': paa/n, 'pab':pab/n, 'pan': pan/n}
    return res

def run_b(G, Na=Na, n=1000, source=source_b):
    pbb, pba, pbn = 0, 0, 0
    for _ in range(n):
        target = am._pick_ba_two_targets(G, source, target_list, dist, 1)
        if len(target) > 0:
            tgt = target.pop()
            if tgt >= Na:
                pbb += 1
            else:
                pba += 1
        else:
            pbn += 1
    res = {'pbb': pbb/n, 'pba':pba/n, 'pbn': pbn/n}
    return res

def run_deg_changes(G, fm, Na, n=1000, m=m):
    Ka, Kb = total_degree(G, Na)
    Kad, Kbd = 0, 0
    for _ in range(n):
        if np.random.rand() < fm:
            source = source_a
        else:
            source = source_b
        target = am._pick_ba_two_targets(G, source, target_list, dist, m)
        if len(target) > 0:
            G.add_edges_from(zip([source] * len(target), target))
            Kan, Kbn = total_degree(G, Na)
            Kad += Kan - Ka
            Kbd += Kbn - Kb
            G.remove_node(source)
            G.add_node(source)

    rest = {'Kad': Kad / n, 'Kbd': Kbd / n}
    return rest


def proba_links(Ka, Kb):
    """return probability of getting a link"""
    K = Ka + Kb
    if K > 0:
        pan = (1-haa)*Ka/K + haa*Kb/K
        pbn = (1-hbb)*Kb/K + hbb*Ka/K

        p= fm * (1-pan) + (1-fm) * (1-pbn)
        #print(p)
        return p
    else:
        #print(1.1)
        return 1

def ar(x):
    return np.array(x)

def run_numlinks(track=30, n_avg=50):
    times = range(0, N, track)
    avg_paa = []
    avg_pbb = []
    th_paa = []
    th_pbb = []

    total_links = []
    expected_links = []
    expected_links_1 = []

    c = 1
    bias = [haa, hbb]
    n_iter = 0
    p0 = None

    for ii in range(n_avg):
        _, _, (P, K), _, _ = am.run_growing(N, fm, c, bias, p0, n_iter, track_steps=track, rewire_type='ba_two', m=m)
        print(ii)
        laa, lab, lbb = ar(P['l_aa']), ar(P['l_ab']), ar(P['l_bb'])
        total_links.append(2 * (laa + lab + lbb))
        ep = [2 * m * (t) * proba_links(ka, kb) for t, ka, kb in zip(times, K['Ka'], K['Kb'])]
        ep1 = [2 * m * (t+1) * proba_links(ka, kb) for t, ka, kb in zip(times, K['Ka'], K['Kb'])]
        expected_links.append(ep)
        expected_links_1.append(ep1)
        #avg_paa.append(P['l_aa'])
        #avg_pbb.append(P['l_bb'])
        #new_paa = [m * haa * ka * t for ka, t in zip(K['Kan'], times)]
        #th_paa.append(new_paa)
        #th_pbb.append([m * hbb * kb * t for kb, t in zip(K['Kbn'], times)])

    total_links = ar(total_links)
    expected_links = ar(expected_links)
    expected_links_1 = ar(expected_links_1)

    tl_avg = total_links.mean(0)
    el_avg = expected_links.mean(0)

    tl_md = np.median(total_links, 0) #.mean(0)
    el_md = np.median(expected_links, 0) #.mean(0)

    print('Total Links Mean: {}'.format(tl_avg.round()))
    print('Expec Links Mean: {}'.format(el_avg.round()))

    print('Total Links Medn: {}'.format(tl_md.round()))
    print('Expec Links Medm: {}'.format(el_md.round()))

    plot_total_expected(times, total_links, expected_links, expected_links_1)
    #avg_paa = np.array(avg_paa)
    #avg_pbb = np.array(avg_pbb)
    #th_paa = np.array(th_paa)
    #th_pbb = np.array(th_pbb)

    #avg_paa = avg_paa.mean(0)
    #avg_pbb = avg_pbb.mean(0)

    #th_paa = th_paa.mean(0)
    #th_pbb = th_pbb.mean(0)
    #print('Tho Laa: {}'.format(th_paa.round()))
    #print('Avg Laa: {}'.format(avg_paa.round()))
    #print('---------------------------------------')
    #print('Tho Lbb: {}'.format(th_pbb.round()))
    #print('Avg Lbb: {}'.format(avg_pbb.round()))

def plot_total_expected(times, total_links, expected_links, expected_links_1):
    fig, ax = plt.subplots()

    diff_min = np.percentile(total_links, 10, 0) - np.percentile(expected_links, 10, 0)
    diff_max = np.percentile(total_links, 90, 0) - np.percentile(expected_links, 90, 0)
    diff_mean = total_links.mean(0) - expected_links.mean(0)

    ax.fill_between(range(len(diff_min)), diff_min, diff_max, color='r', alpha=.2)
    ax.plot(range(len(diff_mean)), diff_mean, color='r', label='total - expected', alpha=.4)

    diff_min = np.percentile(total_links, 10, 0) - np.percentile(expected_links_1, 10, 0)
    diff_max = np.percentile(total_links, 90, 0) - np.percentile(expected_links_1, 90, 0)
    diff_mean = total_links.mean(0) - expected_links_1.mean(0)
    ax.fill_between(range(len(diff_min)), diff_min, diff_max, color='b', alpha=.2)
    ax.plot(range(len(diff_mean)), diff_mean, color='b', label='total - expected +1', alpha=.4)
    ax.hlines(y=0, xmin=0, xmax=len(diff_min))
    fig.legend()
    fig.savefig('evolution_total_links_m{}.pdf'.format(m))



if __name__ == '__main__':
    #print(get_ps(G, Na, haa, hbb))
    #print(run_a(G))
    #print(run_b(G))
    #print(get_kds(G, fm, Na, haa, hbb))
    #print(run_deg_changes(G, fm, Na))
    run_numlinks()

