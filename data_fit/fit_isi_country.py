import numpy as np
import networkx as nx
from collections import Counter
import pandas as pd

#citation_path = '../data/isi/All_19002013_CitationList_Mapped_samp.txt'
#fields_path = '../data/isi/major_fields_total.txt'
#metadata = '../data/isi/All_19002013_BasicInformation_Sample.txt'
#countries = 'isi/country_counts.txt'

citation_path = '/m/cs/scratch/isi/javier/All_19002013_CitationList_Mapped.txt'
fields_path = '../major_fields_total.txt'
metadata = '/m/cs/scratch/networks/data/isi/WoS-Derived/All_19002013_BasicInformation.txt'
countries = 'isi/country_counts.txt'

outpath = 'cp_country_results.txt'

def get_country_groups():
    grps = pd.read_csv(countries, sep='|', names=['ctry', 'cnt', 'grp'])
    grps_dict = dict(grps[['ctry', 'grp']].values)
    return grps_dict

grps_dict = get_country_groups()

def read_metadata():
    df = pd.read_csv(metadata, sep='|', names=['ID', 'Number', 'lang', 'doctype', 'year', 'date' , 'nauthors', 'naddress','npags', 'nref', 'nfunding', 'ngrantnum', 'authors', 'country', 'city', 'pc', 'state', 'street', 'org'], dtype=str)
    df = df[df.country.notnull()]
    return df

def categorize_countries(df):
    df['country'] = df.country.apply(_remove_repeated)
    df['area'] = df.country.apply(_areas)
    return df

def _get_area(x):
    return grps_dict.get(x, None)

    #if x in eng:
    #    return '3'
    #elif x in ww:
    #    return '0'
    #elif x in ee:
    #    return '1'
    #elif x in rest:
    #    return '2'
    #else:
    #    return None

def _areas(x):
    if '%' in x:
        a = set([_get_area(r) for r in x.split('%')])
        if len(a) == 1:
            return a.pop()
        else:
            return '-1'
    else:
        return _get_area(x)


def _remove_repeated(x):
    x = set(x.lower().split('%'))
    if len(x) == 1:
        return x.pop()
    else:
        return '%'.join(list(x))



class ISIData(object):
    def __init__(self, f1, f2, years=(0, np.inf)):
        self.f1 = f1
        self.f2 = f2

        self.tots = {'laa': 0,
                'lbb': 0,
                'lab': 0,
                'lba': 0}
        self.x = {}
        self.n = {}
        self.i = 0
        self.get_groups(years)
        self.nnodes = {'a': set(), 'b': set()}

    def get_groups(self, years):
        groups = {}
        df = read_metadata()
        df = df[['ID', 'Number', 'country', 'year']]
        df['year'] = df.year.astype(int)
        df = df[(df.year > years[0]) & (df.year < years[1])]
        df = categorize_countries(df)


        groups['a'] = set(df.Number[df.area.isin(self.f1)].values)
        groups['b'] = set(df.Number[df.area.isin(self.f2)].values)
        self.groups = groups

    def grow_net(self):
        self.net = nx.empty_graph()
        with open(citation_path, 'r') as r:
            line = r.readline()
            while line:
                self.add_edges(line)
                line = r.readline()

    def add_edges(self, line):
        line = line.replace('\n', '')
        try:
            new, cits = line.split('|')
        except:
            new, cits = '', ''
        if self.check_a(new):
            new_g = 'a'
        elif self.check_b(new):
            new_g = 'b'
        else:
            new_g = ''

        if new_g:
            self.nnodes[new_g].add(new)
            cits = cits.split(',')
            #TODO: update x and n
            add = False
            self.current_x = []
            self.update_deg_dist()
            for cit in cits:
                if self.check_a(cit):
                    self.tots['l'+new_g+'a'] += 1
                    self.net.add_edge(new, cit)
                    self.nnodes['a'].add(cit)
                    self.update_stats(new_g, 'a', cit)
                    add = True
                elif self.check_b(cit):
                    self.tots['l'+new_g+'b'] += 1
                    self.net.add_edge(new, cit)
                    self.nnodes['b'].add(cit)
                    self.update_stats(new_g, 'b', cit)
                    add = True
            if add:
                counts = Counter(self.current_x)
                fx = [[x[0], x[1], x[2], k] for x, k in counts.items()]
                self.x[self.i] = fx
                nx = self.current_n
                self.n[self.i] = nx
                self.i += 1

    def update_deg_dist(self):
        a_nodes = []
        b_nodes = []
        for node in self.net.nodes():
            deg = self.net.degree(node)
            if self.check_a(node):
                a_nodes.append(deg)
            else:
                b_nodes.append(deg)
        a_deg = Counter(a_nodes)
        b_deg = Counter(b_nodes)
        self.deg_dist = {'a': a_deg, 'b': b_deg}
        pa = sum([k*x for k, x in a_deg.items()])
        pb = sum([k*x for k, x in b_deg.items()])
        ua = sum(a_deg.values())
        ub = sum(b_deg.values())
        self.current_n = [pa, pb, ua, ub]



    def update_stats(self, sgroup, tgroup, tgt):
        xg = sgroup + tgroup
        tgt_deg = self.net.degree(tgt) - 1
        nk = self.deg_dist[tgroup].get(tgt_deg, 1)
        self.current_x.append((xg, tgt_deg, nk))


    def check_a(self, x):
        if x in self.groups['a']:
            return True
        else:
            return False

    def check_b(self, x):
        if x in self.groups['b']:
            return True
        else:
            return False

    def print_tots(self):
        with open(outpath, 'a') as w:
            nnodes = '{}|{}\n'.format(len(self.nnodes['a']), len(self.nnodes['b']))
            res = self.tots
            resline = '{}|{}|{}|'.format(res['laa'], res['lbb'] , res['lab'] + res['lba'])
            line = '{}|{}|'.format(self.f1, self.f2) + resline + nnodes
            w.write(line)

    def get_data(self):
        self.grow_net()
        x, n = {}, {}
        for (i, xi), (j, ni) in zip(self.x.items(), self.n.items()):
            xj = [s for s in xi if s[1] > 0]
            if xj:
                x[i] = xj
                n[i] = ni
        na = len(self.nnodes['a'])
        na = na / (len(self.nnodes['b']) + na)
        return x, n, na


if __name__ == '__main__':
        import sys #
        import growth_degree_fit as gdf
        #NOTE: f1 and f2 must be lists, python fit_isi_country.py '["USA"]' '["WEST"]'
        f1, f2 = sys.argv[1:3]
        yrs = sys.argv[3:5]
        yrs = (eval(yrs[0]), eval(yrs[1]))
        ID = ISIData(eval(f1), eval(f2), yrs)
        x, n, na = ID.get_data()
        ID.print_tots()
        GF = gdf.GrowthFit(x, n, na)
        sol = GF.solve()
        #f1|f2|sa|sb|c|na
        line = '{}|{}|{}|{}|{}|{}\n'.format(f1, f2, sol[0], sol[1], sol[2], na)
        print(line)
        with open('isi_country_estimated_params.txt', 'a') as w:
            w.write(line)

