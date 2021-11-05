import growth_degree_fit as gdf
import sys
from utils import pacs_utils as pu
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
from itertools import combinations
from pandas import DataFrame
import json

def plot_params_citation():
    fig, axs = plt.subplots(1, 2, figsize=(7, 3)) #)3, figsize=(10, 3))
    groups = ['01', '02', '03', '04', '05', '07', '11', '12', '21', '23', '24', '25']
    #groups = ['21', '23', '24', '25']
    groups = ['61', '62', '63', '64', '65', '66', '67', '68']
    #groups = ['41', '42', '43', '44', '45', '46', '47']
    ng = len(groups)
    sas, cvals, nas = np.zeros((ng, ng)), np.zeros((ng, ng)), np.zeros((ng, ng))
    for a, b in combinations(groups, 2):
        print('{} {}'.format(a, b))
        if (a=='03') and (b=='05'):
            na = .489
            sol = np.array([0.90670508, 0.89958437, 0.91045515])
        elif (b=='04') and (a=='03'):
            na = .309
            sol = np.array([0.72704438, 0.7201636 , 0.64134653])
        elif (a=='02') and (b=='03'):
            na = .2753
            sol = np.array([0.6397818 , 0.71169366, 0.73589583])
        elif (a=='21') and (b=='23'):
            na = .2753
            sol = np.array([0.47921657, 0.92070757, 0.87983703])
        elif (a=='21') and (b=='24'):
            na = .2753
            sol = np.array([0.75192532, 0.86477237, 0.90861399])
        elif (a=='21') and (b=='25'):
            na = .2753
            sol = np.array([0.78758477, 0.83090862, 0.93376391])
        elif (a=='23') and (b=='24'):
            na = .2753
            sol = np.array([0.98843447, 0.70408613, 0.69104485])
        elif (a=='23') and (b=='25'):
            na = .2753
            sol = np.array([0.7642502, 0.68182421, 0.63081204])
        elif (a=='24') and (b=='25'):
            na = .2753
            sol = np.array([0.65484339, 0.64167383, 0.90201066])
        else:
            x, n, na = pu.network_stats(a=[a], b=[b])
            GF = gdf.GrowthFit(x, n, na)
            sol = GF.solve()
        print(sol)
        i = groups.index(a)
        j = groups.index(b)
        sas[i, j] = sol[0]
        sas[j, i] = sol[1]
        cvals[i, j] = sol[2]
        cvals[j, i] = sol[2]
        nas[i, j] = na
        nas[j, i] = na

        axs[0].imshow(sas)
        axs[1].imshow(cvals)
        #axs[2].imshow(nas)
        fig.savefig('citation_params.pdf')
    #sns.set(font_scale=1.2)
    group_names = json.load(open('utils/pacs_ref.json', 'r'))
    names = [group_names[i] for i in groups]
    sas = DataFrame(sas, columns=names, index=names)
    cvals = DataFrame(cvals, columns=names)
    nas = DataFrame(nas, columns=names)

    sns.heatmap(sas, ax=axs[0])
    sns.heatmap(cvals, ax=axs[1])
    axs[0].set_yticklabels(axs[0].get_ymajorticklabels(), fontsize = 8)
    axs[1].set_yticklabels(axs[1].get_ymajorticklabels(), fontsize = 8)
    axs[0].set_xticklabels(axs[0].get_xmajorticklabels(), fontsize = 8)
    axs[1].set_xticklabels(axs[1].get_xmajorticklabels(), fontsize = 8)
    #sns.heatmap(nas, ax=axs[2])
    axs[0].set_title('Homophily')
    axs[1].set_title('Pref. Attachment')
    #axs[2].set_title('Minority Size')
    fig.tight_layout()
    fig.savefig('plots/citation_params_40.pdf')







