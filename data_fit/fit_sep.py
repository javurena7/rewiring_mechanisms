import numpy as np
import pandas as pd
import networkx as nx
import rewire_degree_fit as rdf
import growth_degree_fit as gdf
import matplotlib.pyplot as plt; plt.ion()
from collections import Counter, OrderedDict
import datetime as dt
import get_fixed_points as gfp


class NetSep(object):
    """
    Class for separating network evolution by type of behaviour. Each log is classified as:
            1: new source, new target
            2: new source, old target (main growth)
            3: old source, new target
            4: old source, old target but new link (main rewire)
            5: old link
    """

    def __init__(self, log_df, groups, behaviour=4, censor=False):
        self.df = log_df
        self.groups = groups
        self.net = nx.empty_graph()
        self.net_evol = nx.empty_graph()
        self.behaviour = behaviour
        self.censor = censor
        self.i = 1
        self.a_tots = 0

    def edge_behaviour(self, edge):
        """y
        Edge classification:
        1: new source, new target
        2: new source, old target (main growth)
        3: old source, new target
        4: old source, old target but new link (main rewire)
        5: old link
        """
        if self.net.has_edge(*edge):
            return 5
        elif self.net.has_node(edge[0]):
            if self.net.has_node(edge[1]):
                return 4
            else:
                return 3
        else:
            if self.net.has_node(edge[1]):
                return 2
            else:
                return 1

    def edge_type(self, edge):
        a1 = 'a' if edge[0] in self.groups else 'b'
        a2 = 'a' if edge[1] in self.groups else 'b'
        return a1 + a2

    def net_tots(self):
        """
        Update the group-level degree distribution to include new links
        TODO: make this more efficient by adding links step by step
        """
        n_i = {'na': {}, 'nb': {}}
        for node in self.net:
            deg = self.net.degree(node)
            if node in self.groups:
                n_i['na'][deg] = n_i['na'].get(deg, 0) + 1
            else:
                n_i['nb'][deg] = n_i['nb'].get(deg, 0) + 1
        self.degs = n_i
        pa = sum([k*v for k, v in n_i['na'].items()])
        pb = sum([k*v for k, v in n_i['nb'].items()])
        ua = sum(n_i['na'].values())
        ub = sum(n_i['nb'].values())
        n = [pa, pb, ua, ub]
        return n

    def get_net0(self, DT=60*60*24*7):
        self.net = nx.empty_graph()
        self.i = 1
        line = self.df.iloc[0]
        dT = 0
        time0 = line.timestamp
        self.net.add_edge(line.source, line.target, w=time0)
        self.order = OrderedDict()
        #self.order[(line.source, line.target)] = time0
        while dT < DT:
            line = self.df.iloc[self.i]
            delta = line.timestamp - time0
            dT = delta.total_seconds()
            #add_order(line)
            self.net.add_edge(line.source, line.target)
            self.i += 1
        #self.df = self.df.iloc[self.i:]
        return self.net.copy()


    def net_diff(self): #, net, net_e, df, order, groups, DT, i):
        # Evolve network until we create a new link
        x_i = []
        n_i = []
        #degs = Counter([v for k, v in net.degree()])
        df_len = self.df.shape[0]
        while not x_i and self.i < df_len:
            line = self.df.iloc[self.i]
            edge = (line.source, line.target)
            e_bhvr = self.edge_behaviour(edge)
            if e_bhvr == self.behaviour:
                n_i = self.net_tots()
                e_type = self.edge_type(edge)
                tgt_deg = self.net.degree(edge[1])
                #n_degs = self.tgt_deg_cnt(e_type, tgt_deg)
                n_degs = self.degs['n'+e_type[1]][tgt_deg]
                x_i = [e_type, tgt_deg, n_degs]
                #n_i = self.get_net_tots()
                if not self.censor:
                    self.net_evol.add_edge(*edge) #update evolution-rule net
                #self.add_order(line) #TODO: outside if?
                if self.behaviour == 4 and not self.censor:
                    self.remove_link(edge)
                if self.behaviour == 2:
                    x_i.append(1)
                    x_i = [x_i]
            #self.update_degs(edge)
            self.net.add_edge(*edge) #update general net
            self.i += 1
        if x_i:
            if self.censor:
                a_rate = self.a_tots / len(self.x)
                if a_rate < self.na - .03:
                    if x_i[0][0] == 'a':
                        self.x[self.time] = x_i
                        self.n[self.time] = n_i
                        self.a_tots += 1
                        self.net_evol.add_edge(*edge) #update evolution-rule net
                        self.remove_link(edge)
                elif a_rate > self.na + .03:
                    if x_i[0][0] == 'b':
                        self.x[self.time] = x_i
                        self.n[self.time] = n_i
                        self.net_evol.add_edge(*edge) #update evolution-rule net
                        self.remove_link(edge)
                else:
                    self.x[self.time] = x_i
                    self.n[self.time] = n_i
                    self.net_evol.add_edge(*edge) #update evolution-rule net
                    self.remove_link(edge)
                    if x_i[0][0] == 'a':
                        self.a_tots += 1




                #if x_i[0][0] == 'a' and np.random.rand() < self.na:
                #    self.x[self.time] = x_i
                #    self.n[self.time] = n_i
                #elif x_i[0][0] == 'b' and np.random.rand() < 1-self.na:
                #    self.x[self.time] = x_i
                #    self.n[self.time] = n_i

            else:
                self.x[self.time] = x_i
                self.n[self.time] = n_i

    def remove_link(self, edge):
        """
        Remove links when rewiring (edge behaviour 4), where we just delete a link from the current set of neighbors.
        TODO: remove the oldest neighbor
        """
        src, tgt = edge
        neighs = [n for n in self.net_evol.neighbors(src) if n != tgt]
        if len(neighs) > 1:
            rlink = np.random.choice(neighs)
            self.net.remove_edge(src, rlink)
            self.net_evol.remove_edge(src, rlink)


    def get_data_in_range(self, DT=60*60*24):
        net0 = self.get_net0(DT)
        obs0 = self.get_network_stats(net0)
        self.na = obs0['na']
        self.x = {}
        self.n = {}
        n_logs = self.df.shape[0]
        self.time = 0
        while self.i < n_logs:
            self.net_diff()
            self.time += 1
        self.obs = self.get_network_stats()
        obs0['N'] = net0.number_of_nodes()
        obs0['L'] = net0.number_of_edges()

        return self.x, self.n, self.obs, net0, obs0


    def get_network_stats(self, net=None): #get_last_obs(self):
        """
        Get Taa, Tbb, Paa, Pbb
        """
        paa, pbb, pab = 0, 0, 0
        if net is None:
            net = self.net_evol
        n_a = {'a': set(), 'b': set()}
        for edge in net.edges():
            a1 = 'a' if edge[0] in self.groups else 'b'
            a2 = 'a' if edge[1] in self.groups else 'b'

            if a1 == a2:
                if a1 == 'a':
                    paa += 1
                else:
                    pbb += 1
            else:
                pab += 1
            n_a[a1].add(edge[0])
            n_a[a2].add(edge[1])


        tot = paa + pab + pbb
        paa /= tot
        pbb /= tot
        pab /= tot

        taa = 2*paa / (2*paa + pab)
        tbb = 2*pbb / (2*pbb + pab)
        na = len(n_a['a']) / (len(n_a['a']) + len(n_a['b']))

        obs = {'paa': paa,
               'pab': pab,
               'pbb': pbb,
               'taa': taa,
               'tbb': tbb,
                'na': na}

        return obs


    def get_degrees_from_edge(self, edge):
        try:
            src_k = self.net.degree(edge[0])
        except:
            scr_k = 0
        try:
            tgt_k = self.net.degree(edge[0])
        except:
            tgt_k = 0
        return src_k, tgt_k


    def update_degs(self, edge, e_type):
        """
        TODO: finnish this to make the code more efficient.
        1: new source, new target
        2: new source, old target (main growth)
        3: old source, new target
        4: old source, old target but new link (main rewire)
        5: old link
        """
        src_k, tgt_k = self.get_degrees_from_edge(edge)

        if self.behaviour == 2:
            tgt_k = self.net.degree(edge[1])
            self.degs[src_k] = self.degs.get(src_k, 0) - 1
            self.degs[src_k + 1] = self.degs.get(src_k, 0) + 1
            self.degs[1] = self.degs.get(1, 0) + 1



