import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter


def read_df(path):
    df = pd.read_csv(path, skiprows=2, header=0)
    df.drop(0, axis=0, inplace=True)
    df.drop("Unnamed: 1", axis=1, inplace=True)
    df.replace('-', 0, inplace=True)
    df.set_index('Unnamed: 0', drop=True, inplace=True)
    df.astype(int, copy=False)
    df = df.transpose() #NOTE: rows are origin and columns are destinations
    return df

def df_to_logs(df):
    """
    This assumes that rows are origins and columns are destinations
    """
    routes = []
    for src, row in df.iterrows():
        routes += [(src, k) for k, v in row.iteritems() if v != 0]
    return set(routes)

def routes_to_net(routes):
    net = nx.empty_graph()
    net.add_edges_from(routes)
    return net

def get_group_degs(net):
    #grp_ks = {'a': {}, 'b': {}}
    a_cnt = []
    b_cnt = []

    for nd, deg in net.degree():
        grp = get_group(nd)
        if grp == 'a':
            a_cnt.append(deg)
        elif grp == 'b':
            b_cnt.append(deg)
    grp_ks = {'a': Counter(a_cnt), 'b': Counter(b_cnt)}
    return grp_ks


def get_airport_metadata():
    df = pd.read_csv('../data/airport/wikipedia_airports.csv')
    hpr = df[df.FAA.notnull()]
    df = df.where(pd.notnull(df), None)

    state = []
    for _, row in df.iterrows():
        if not row.FAA:
            crrnt = row.City
        else:
            state.append(crrnt)
    hpr['State'] = state

    df = pd.read_csv('../data/airport/wikipedia_cities_percapita_income2010.csv')
    stats = []
    for _, row in hpr.iterrows():
        for _, refrow in df.iterrows():
            splt = refrow['Metropolitan statistical area'].split(',')
            cities = [c.lower() for c in splt[0].split('-')]
            states = [s.lower() for s in splt[1].split('-')]
            #if 'birmingham' in cities:
            #    import pdb; pdb.set_trace()
            row_cities = set([x.lower().strip() for x in row.City.replace('/', ',').split(',')])
            if row_cities.intersection(cities): #and row.State.lower() in states:
                stats += [list(refrow[["Metropolitan statistical area", "Population", "Per capitaincome"]].values)]
                break
        else:
            stats += [[None, None, None]]
    hpr[['area', 'population', 'income']] = stats
    hpr = hpr[hpr.income.notnull()]
    hpr.income = hpr.income.apply(lambda x: int(x.replace('$', '').replace(',', '')))
    return hpr

mtdata = get_airport_metadata()
groupa = set(mtdata.IATA[mtdata.income > 25000])

def get_group(x):
    if x in groupa:
        return 'a'
    else:
        return 'b'


def get_fit_data():
    mpath = '../data/airport/CrosstabsT_ONTIME_REPORTING{}.csv'
    x, n = [], []

    # get_net_0
    path = mpath.format(1987)
    df = read_df(path)
    routes0 = df_to_logs(df)
    net0 = routes_to_net(routes0)

    for yr in range(1988, 2022):

        path1 = mpath.format(yr)
        df = read_df(path)
        routesn = df_to_logs(df)
        nroutes = routesn.difference(routes0)
        i = len(x)
        update_x(nroutes, net0, x, n, i)
        net0 = routes_to_net(routesn)
    return x, n


def update_x(nroutes, net, x, n, i):
    xt = []
    nt = []
    grp_degs = get_group_degs(net)
    #x_i = [linktype, tgtdeg, n_k, x_k(times deg k was selected)]
    #n_i = [pa_tot, pb_tot, ua_tot, ub_tot]
    for rte in nroutes:
        src_grp = get_group(rte[0])
        tgt_grp = get_group(rte[1])
        tgt_k = net.degree(rte[1])
        n_k = grp_degs['tgt_grp'][tgt_k]

        x_i = (src_grp + tgt_grp, tgt_k, n_k)
        pa_tot = sum([k*v for k, v in grp_degs['a'].items()])
        pb_tot = sum([k*v for k, v in grp_degs['b'].items()])
        ua_tot = sum(grp_degs['a'].values())
        ua_tot = sum(grp_degs['b'].values())
        n_i = [pa_tot, pb_tot, ua_tot, ub_tot]
        xt.append(x_i)
        nt.append(n_i)
    xi_counts = Counter(xt)
    xt = [[x[0], x[1], x[2], v] for x, v in xi_counts.items()]

    for xi, ni in zip(xt, nt):
        x[i] = xi
        n[i] = ni
        i += 1



