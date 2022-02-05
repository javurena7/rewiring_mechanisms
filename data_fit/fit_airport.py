import pandas as pd
import numpy as np



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


def get_fit_data():
    mpath = '../data/airport/CrosstabsT_ONTIME_REPORTING{}.csv'
    x, n = [], []

    # get_net_0
    path = mpath.format(1987)
    df = read_df(path)
    routes0 = df_to_logs(df)
    net0 = routes_to_net(routes)

    for yr in range(1988, 2022):

        path1 = mpath.format(yr)
        df = read_df(path)
        routes1 = df_to_logs(df)
        new_routes = routesn.difference(routes0)











