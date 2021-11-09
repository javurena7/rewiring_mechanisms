import numpy as np
import pandas as pd
import networkx as nx
import rewire_degree_fit as rdf
import matplotlib.pyplot as plt; plt.ion()
from collections import Counter, OrderedDict
import datetime as dt
import get_fixed_points as gfp
logpath = '../../../courses/twitter/Data/climate_twitter/twitter_fi_climate_edges_20200824-20201207.csv'

def read_logs(lpath=logpath, ttype='retweet', date0='2020-08-24', daten='2020-09-30'):
    df = pd.read_csv(lpath, header=0)
    df = df[df.type==ttype]
    df = df[['source', 'target', 'timestamp']]
    df['timestamp'] = pd.to_datetime(df.timestamp, infer_datetime_format=True)
    if date0:
        hpr = [int(x) for x in date0.split('-')]
        date0 = dt.datetime(hpr[0], hpr[1], hpr[2])
        df = df[df.timestamp > date0]

    if daten:
        hpr = [int(x) for x in daten.split('-')]
        daten = dt.datetime(hpr[0], hpr[1], hpr[2])
        df = df[df.timestamp < daten]
    #print(f'Number of elements: {df.shape[0]}')
    return df

def add_order(order, line, net):
    if net.has_edge(line.source, line.target):
        if order.get((line.source, line.target)):
            order[(line.source, line.target)] = line.timestamp
        else:
            order[(line.target, line.source)] = line.timestamp
    else:
        order[(line.source, line.target)] = line.timestamp
    return order


def get_net0(df, DT=60*60*24*7):
    net = nx.empty_graph()
    line = df.iloc[0]
    i, dt = 1, 0
    time0 = line.timestamp
    net.add_edge(line.source, line.target, w=time0)
    order = OrderedDict()
    order[(line.source, line.target)] = time0
    while dt < DT:
        line = df.iloc[i]
        delta = line.timestamp - time0
        dt = delta.total_seconds()
        #print(dt)
        order = add_order(order, line, net)
        net.add_edge(line.source, line.target)
        i += 1
    df = df.iloc[i:]
    return net, df, order

def parse_groups():
    v_path = '../data/twitter/verified_accounts.txt'
    mp_path = '../../../courses/twitter/Data/twitter_mps.csv'
    verified = pd.read_csv(v_path, sep=' ', names=('idx', 'handle'))
    ver_ids = set(verified.idx)
    mps = pd.read_csv(mp_path)
    mp_ids = set(mps.twitter_id)
    groups = ver_ids.union(mp_ids)
    return groups

def edge_type(edge, groups):

    #g1 = groups.get(edge[0]
    #g2 = groups[edge[1]]
    a1 = 'a' if edge[0] in groups else 'b'
    a2 = 'a' if edge[1] in groups else 'b'
    return a1 + a2


def net_diff(net, df, order, groups, DT):
    # Evolve network until we create a new link
    x_i = []
    add = 0
    degs = Counter([v for k, v in net.degree()])
    dels = []
    for (i, line) in df.iterrows():
        edge = (line.source, line.target)
        if not net.has_edge(*edge):
            et = edge_type(edge, groups)
            try:
                tgt_deg = net.degree(edge[1])
            except:
                tgt_deg = ''
            if isinstance(tgt_deg, int):
                x_i.append(et)
                x_i.append(tgt_deg)
                x_i.append(degs[tgt_deg])
                net.add_edge(*edge)
                order = add_order(order, line, net)
                break
            else:
                add += 1
                net.add_edge(*edge)
                order = add_order(order, line, net)
                dels.append(i)
        else:
            #noadd += 1
            order = add_order(order, line, net)
            dels.append(i)

    df = df.drop(dels)

    #print(f'Number of elements: {df.shape[0]}')
    #print('     +{}'.format(len(x_i)))
    #print('     -{}'.format(noadd))
    return [x_i], net, df

def net_tots(net, groups):
    n_i = {'na': {}, 'nb': {}}
    for node in net:
        deg = net.degree(node)
        if node in groups:
            n_i['na'][deg] = n_i['na'].get(deg, 0) + 1
        else:
            n_i['nb'][deg] = n_i['nb'].get(deg, 0) + 1
    pa = sum([k*v for k, v in n_i['na'].items()])
    pb = sum([k*v for k, v in n_i['nb'].items()])
    ua = sum(n_i['na'].values())
    ub = sum(n_i['nb'].values())
    n = [pa, pb, ua, ub]
    return n

def parse_nets(net0, df, order, x, n, groups, DT):
    n_ = net_tots(net0, groups)
    x_i, net0, df = net_diff(net0, df, order, groups, DT)
    #print(f'order len: {len(order)}')
    n_i = [n_ for _ in range(len(x_i))]
    if x_i:
        x += x_i
        n += n_i
    return df

def update_net0(net0, order, DT):
    last = next(reversed(order.values()))
    rem_keys = []
    for k, v in order.items():
        delta = last - v
        t = delta.total_seconds()
        if t > DT:
            net0.remove_edge(*k)
            rem_keys.append(k)
    for k in rem_keys:
        order.pop(k)
    return net0

def update_na(na, net0, groups):
    for node in net0.nodes():
        #g = groups[node]
        if node in groups:
            na['a'].add(node)
        else:
            na['b'].add(node)

def get_last_obs(net0, groups):
    """
    Get Taa, Tbb, Paa, Pbb
    """
    paa, pbb, pab = 0, 0, 0
    for edge in net0.edges():
        a1 = 'a' if edge[0] in groups else 'b'
        a2 = 'a' if edge[1] in groups else 'b'

        if a1 == a2:
            if a1 == 'a':
                paa += 1
            else:
                pbb += 1
        else:
            pab += 1
    tot = paa + pab + pbb
    paa /= tot
    pbb /= tot
    pab /= tot

    taa = 2*paa / (2*paa + pab)
    tbb = 2*pbb / (2*pbb + pab)

    obs = {'paa': paa,
            'pab': pab,
            'pbb': pbb,
            'taa': taa,
            'tbb': tbb}

    return obs


def get_data_in_range(DT=60*60*24*7, date0='2020-08-24', daten='2020-09-30', ttype='retweet'):
    net0 = None
    groups = parse_groups()
    x, n, na = [], [], {'a': set(), 'b': set()}
    df = read_logs(logpath, ttype, date0, daten)
    net0, df, order = get_net0(df, DT)
    time = 0
    while not df.empty:
        df = parse_nets(net0, df, order, x, n, groups, DT)
        net0 = update_net0(net0, order, DT)
        update_na(na, net0, groups)
        time += 1
        print('time: {}; n0_size={}'.format(time, net0.number_of_edges()))
    na = len(na['a']) / (len(na['a']) + len(na['b']))
    obs = get_last_obs(net0, groups)
    obs['na'] = na

    x = {i: x_i for i, x_i in enumerate(x)}
    n = {i: n_i for i, n_i in enumerate(n)}

    return x, n, obs


def plot_ranges(dt=12):
    fig, ax = plt.subplots()
    sas, sbs, cs = [], [], []
    for i in range(2002, 2011):
        x, n, obs = get_data_in_range(i, i+1, dt=dt)
        RF = rdf.RewireFit(x, n, .3)
        sa, sb, c = RF.solve()
        print(sa, sb, c)
        sas.append(sa); sbs.append(sb); cs.append(c)
    ax.plot(range(2003, 2012), sas, label=r'$s_a$')
    ax.plot(range(2003, 2012), sbs, label=r'$s_b$')
    ax.plot(range(2003, 2012), cs, label=r'$c$')
    ax.set_xlabel('Year')
    ax.set_ylabel('Estimate')
    ax.set_title('Boards of Directors in Norway')
    fig.legend()
    #fig.tight_layout()
    fig.savefig('plots/temporal_dt{}_boards_directors.pdf'.format(dt))


def plot_snapshots(DT=60*60*24*1, ttype='retweet'):
    fig, axs= plt.subplots(1, 2)
    sas, sbs, cs, nas, taas, tbbs = [], [], [], [], [], []
    dates = [('2020-08-24', '2020-09-15'), ('2020-10-02', '2020-10-10'), ('2020-11-15', '2020-12-03')]
    #dates = [('2020-08-24', '2020-08-31'), ('2020-09-07', '2020-09-16'), ('2020-10-03', '2020-10-10'), ('2020-10-15', '2020-10-22'), ('2020-11-25', '2020-12-03')]
    #dates = [('2020-08-24', '2020-08-26'), ('2020-09-07', '2020-09-09'), ('2020-10-02', '2020-10-4'), ('2020-10-15', '2020-10-17'), ('2020-11-15', '2020-11-17')]
    for start, end in dates:
        x, n, obs = get_data_in_range(DT=DT, date0=start, daten=end, ttype=ttype)
        patch = len(x)
        x.pop(patch - 1)
        na = obs['na']
        RF = rdf.RewireFit(x, n, .3)
        sa, sb, c = RF.solve()
        print(sa, sb, c)
        sas.append(sa); sbs.append(sb); cs.append(c); nas.append(na)
        taas.append(obs['paa']), tbbs.append(obs['pbb'])
    xvals = [start for start, end in dates]
    axs[0].plot(xvals, sas, '-', color='g', alpha=.5)
    axs[0].plot(xvals, sbs, '-', color='b', alpha=.5)
    axs[0].plot(xvals, cs, '-', color='r', alpha=.5)
    axs[0].plot(xvals, nas, '-', color='k', alpha=.5)

    axs[0].plot(xvals, sas, 'o', label=r'$s_a$', color='g')
    axs[0].plot(xvals, sbs, 'o', label=r'$s_b$', color='b')
    axs[0].plot(xvals, cs, 'o', label=r'$c$', color='r')
    axs[0].plot(xvals, nas, 'o', label=r'$n_a$', color='k')

    pred_ts = [gfp.fixed_points(c, na, sa, sb) for c, na, sa, sb in zip(cs, nas, sas, sbs)]
    ptaas = [pred_t[0][0] for pred_t in pred_ts]
    ptbbs = [pred_t[0][1] for pred_t in pred_ts]
    axs[1].plot(xvals, ptaas, '-', color='g', alpha=.5)
    axs[1].plot(xvals, ptbbs, '-', color='b', alpha=.5)
    axs[1].plot(xvals, taas, '-', color='lime', alpha=.5)
    axs[1].plot(xvals, tbbs, '-', color='cyan', alpha=.5)

    axs[1].plot(xvals, ptaas, 'o', label='Predicted ' + r'$P_{aa}$', color='g')
    axs[1].plot(xvals, ptbbs, 'o', label='Predicted ' + r'$P_{bb}$', color='b')
    axs[1].plot(xvals, taas, 'o', label='Observed ' + r'$P_{aa}$', color='lime')
    axs[1].plot(xvals, tbbs, 'o', label='Observed ' + r'$P_{bb}$', color='cyan')

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Climate Twitter in Finland\n Estimated Parameters')
    axs[0].legend()

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Estimate')
    axs[1].set_title('Climate Twitter in Finland\n Estimated T-matrix')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig('plots/snapshot_dt{}_climate_twitter_{}.pdf'.format(DT, ttype))
