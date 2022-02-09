import numpy as np
import networkx as nx
from collections import Counter
import pandas as pd

citation_path = '../data/isi/All_19002013_CitationList_Mapped_samp.txt'
fields_path = '../data/isi/major_fields_total.txt'
metadata = '../data/isi/All_19002013_BasicInformation_Sample.txt'
countries = 'isi/country_counts.txt'

#citation_path = '/m/cs/scratch/isi/javier/All_19002013_CitationList_Mapped.txt'
#fields_path = '../major_fields_total.txt'
#metadata = '/m/cs/scratch/networks/data/isi/WoS-Derived/All_19002013_BasicInformation.txt'
#countries = 'isi/country_counts.txt'

outpath = 'cp_country_results.txt'

def get_country_groups():
    grps = pd.read_csv(countries, sep='|', names=['ctry', 'cnt', 'grp'])
    grps_dict = dict(grps[['ctry', 'grp']].values)
    return grps_dict

grps_dict = get_country_groups()




def read_metadata():
    #df = pd.read_csv(metadata, sep='|', names=['ID', 'Number', 'lang', 'doctype', 'year', 'date' , 'nauthors', 'naddress','npags', 'nref', 'nfunding', 'ngrantnum', 'authors', 'country', 'city', 'pc', 'state', 'street', 'org'], dtype=str)
    #df = df[df.country.notnull()]
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
        return a
    else:
        return set([_get_area(x)])


def _remove_repeated(x):
    x = set(x.lower().split('%'))
    if len(x) == 1:
        return x.pop()
    else:
        return '%'.join(list(x))


class ISIData(object):
    """
    Class for obtaining network evolution data for citation networks, where the groups are determined by countries in periods between years
    """
    def __init__(self, f1, f2, years=(0, np.inf), net0=None):
        self.f1 = f1
        self.f2 = f2

        self.tots = {'laa': 0,
                'lbb': 0,
                'lab': 0,
                'lba': 0}
        self.x = {}
        self.n = {}
        self.i = 0
        self.get_metadata(years[0], years[1])
        self.nnodes = {'a': set(), 'b': set()}
        if not net0:
            self.net = nx.empty_graph()
        else:
            self.net = net0
        self.years = years

    def get_metadata(self, min_yr=0, top_yr=np.inf):
        """
        Read metadata line by line, and categorize countries immediately
        """
        groups = {'a': set(), 'b': set()}
        cit_papers = set()
        n_idx = 1 #Index of "Number" (paper id)
        y_idx = 4 #Index of year
        ctry_idx = 13 #Index for country
        with open(metadata, 'r') as r:
            line = r.readline()
            while line:
                line = line.split('|')
                yr = int(line[y_idx])
                if yr < top_yr: #Check that we are not over_years
                    ctry = _remove_repeated(line[ctry_idx])
                    areas = _areas(ctry)
                    ppr_n = line[n_idx]
                    if areas.intersection(self.f1):
                        if not areas.intersection(self.f2):
                            groups['a'].add(ppr_n)
                            if yr >= min_yr:
                                cit_papers.add(ppr_n)
                    elif areas.intersection(self.f2):
                        groups['b'].add(ppr_n)
                        if yr >= min_yr:
                            cit_papers.add(ppr_n)
                line = r.readline()

        self.groups = groups
        self.cit_papers = cit_papers

    #def get_groups(self, years):
    #    groups = {}
    #    df = self.get_metadata(years[1])
    #    df = df[['ID', 'Number', 'country', 'year']]
    #    df['year'] = df.year.astype(int)
    #    df = df[(df.year > years[0]) & (df.year < years[1])]
    #    df = categorize_countries(df)


    #   groups['a'] = set(df.Number[df.area.isin(self.f1)].values)
    #    groups['b'] = set(df.Number[df.area.isin(self.f2)].values)
    #    self.groups = groups

    def grow_net(self):
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
        new_g = ''
        #cit_papers contains the set of papers within selected years
        if new in self.cit_papers:
            if self.check_a(new):
                new_g = 'a'
            elif self.check_b(new):
                new_g = 'b'

        if new_g:
            self.nnodes[new_g].add(new)
            cits = cits.split(',')
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

    def print_tots(self, totspath=''):
        res = self.tots
        resline = '{}|{}|{}|'.format(res['laa'], res['lbb'] , res['lab'] + res['lba'])
        nnodes = '{}|{}|'.format(len(self.nnodes['a']), len(self.nnodes['b']))
        yy = '{}|{}'.format(self.years[0], self.years[1])
        line = '{}|{}|'.format(self.f1, self.f2) + resline + nnodes + yy
        if totspath:
            #totspath = outpath
            with open(totspath, 'a') as w:
                w.write(line+'\n')
        else:
            return line

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
        from os.path import exists
        import growth_degree_fit as gdf
        #NOTE: f1 and f2 must be lists, python fit_isi_country.py '["USA"]' '["WEST"]'

        f1, f2 = sys.argv[1:3]
        yrs = sys.argv[3:5]
        yr_0 = eval(yrs[0])
        yr_range = eval(yrs[1])
        for yr in range(yr_0, 2011, yr_range):
            yrs = (yr, yr + yr_range)
            ID = ISIData(eval(f1), eval(f2), yrs)
            x, n, na = ID.get_data()

            f1c, f2c = '-'.join(eval(f1)), '-'.join(eval(f2))
            fcountries = 'A%{}_B%{}'.format(f1c, f2c)
            fyrs = str(yrs[0]) + str(yrs[1])
            #cp_name = 'isi/country_estimated_params_{}.txt'.format(fcountries)
            es_name = 'isi/country_evol_{}.txt'.format(fcountries)

            cp_line = ID.print_tots()
            GF = gdf.GrowthFit(x, n, na)
            sol = GF.solve()
            #f1|f2|sa|sb|c|na
            #totline: f1|f2|laa|lbb|lab|Na|Nb|
            line = cp_line + '|{}|{}|{}|{}\n'.format(sol[0], sol[1], sol[2], na)
            print(line)
            if not exists(es_name):
                with open(es_name, 'a') as w:
                    w.write('f1|f2|laa|lbb|lab|Na|Nb|y0|yn|sa|sb|c|na\n')
            with open(es_name, 'a') as w:
                w.write(line)

            sol0 = GF.solve_c0()
            line = cp_line + '|{}|{}|{}|{}\n'.format(sol0[0], sol0[1], 0, na)
            with open(es_name, 'a') as w:
                w.write(line)


