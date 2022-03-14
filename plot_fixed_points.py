import get_fixed_points as gfp
import matplotlib.pyplot as plt; plt.ion()
plt.rcParams["axes.grid"] = False
import seaborn as sns
import numpy as np
from itertools import product
import pickle as p
from scipy import interpolate


lab_dict = {'sa': r'$s_a$', 'sb': r'$s_b$', 'c': r'$c$', 'na': r'$n_a$', 'rho':r'$\rho$'}

def get_phase_space_grow(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250):
    xvals = np.linspace(*xlims, n_size + 2)[1:-1]
    yvals = np.linspace(*ylims, n_size + 2)[1:-1]
    acore, bcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
    acore[:], bcore[:] = np.nan, np.nan
    acore2, bcore2 = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
    acore2[:], bcore2[:] = np.nan, np.nan
    for (j, xv), (i, yv) in product(enumerate(xvals), enumerate(yvals)):
        print(i,j)
        params[x] = xv
        params[y] = yv
        dens, ca, cb = gfp.growth_fixed_points_density(**params)
        acore, bcore = classify_densities(dens, ca, cb, acore, bcore, i, j)
    results = {'acore': acore, 'bcore': bcore, 'params': params}
    p.dump(results, open('data_plot_phasespaceg_{}{}-rho{}.p'.format(x,y, params['rho']), 'wb'))
    return acore, bcore


def plot_phase_space_grow(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250, fig=None, ax=None, data=(), title='', n_ticks=11, border_width=10):
    """
    data: format (acore, bcore, border) if the experiments have already run
    """
    if ax is None:
        fig, ax = plt.subplots()
    if not data:
        acore, bcore = get_phase_space_grow(x, y, params, xlims, ylims, n_size)
    else:
        acore, bcore = data
    #sns.heatmap(border, cbar=False, ax=ax, center=0, vmax=1, cmap='Greys')
    sns.heatmap(acore, cbar=False, center=0, ax=ax, vmin=0, vmax=.8,  cbar_kws={'label': r'$r$'}, square=True, cmap='Spectral')
    sns.heatmap(bcore, cbar=False, center=0, ax=ax, vmin=0, vmax=.8, cmap='Spectral')
    if not np.isnan(acore).all():
        aborder = get_border(n_size, acore)
        aint = interpolate_border(aborder, n_size/10)
        ax.plot(aint[0], aint[1], linewidth=border_width, color='grey')
        a_loc = np.mean(np.where(acore > 0), 1)
        ax.text(a_loc[1], a_loc[0], 'A+', color='dimgrey', size=12)

    if not np.isnan(bcore).all():
        bborder = get_border(n_size, bcore)
        bint = interpolate_border(bborder, n_size/10, invert=True)
        ax.plot(bint[0], bint[1], linewidth=border_width, color='grey')
        b_loc = np.mean(np.where(bcore > 0), 1)
        ax.text(b_loc[1], b_loc[0], 'B+', color='dimgrey', size=12)

    #REMOVE: hardcoded text for
    ax.text(.4*n_size, .4*n_size, '0', size=12, color='dimgrey')
    ax.text(.85*n_size, .85*n_size, '0', size=12, color='dimgrey')

    xticks = np.linspace(0, n_size, n_ticks)
    xticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*xlims, n_ticks)]
    yticks = np.linspace(0, n_size, n_ticks)
    yticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*ylims, n_ticks)]
    ax.axvline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.axhline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(lab_dict[x])
    ax.set_ylabel(lab_dict[y])

    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if fig:
        fig.tight_layout()
        #results = {'acore': acore, 'bcore': bcore, 'params': params}
        name = 'plots/phsp_grow_{}_{}-rho{}.pdf'.format(x, y, params['rho'])
        fig.savefig(name)


def interpolate_border(border, s=10, invert=False):
    y, x = np.where(border == 1)
    #x, y = np.where(border == 1)
    if invert:
        y = [i for _, i in sorted(zip(x, y))]
        x = sorted(x)
    tck, u = interpolate.splprep([x, y], s=s)

    unew = np.arange(0, 1.001, .001)
    out = interpolate.splev(unew, tck)
    return out[0], out[1]

def get_phase_space_rewire(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(.5, 1), ylims=(.5, 1), n_size=250):
    xvals = np.linspace(*xlims, n_size + 2)[1:-1]
    yvals = np.linspace(*ylims, n_size + 2)[1:-1]
    acore, bcore = np.zeros((n_size, n_size)), np.zeros((n_size, n_size))
    st_border = np.zeros((n_size, n_size)); st_border[:] = np.nan
    acore[:], bcore[:] = np.nan, np.nan
    saved = {}
    for (j, xv), (i, yv) in product(enumerate(xvals), enumerate(yvals)):
        print(i,j)
        params[x] = xv
        params[y] = yv
        dens,ca, cb = gfp.rewire_fixed_points_density(**params)
        #saved[(i,j)] = [dens, ca, cb]
        acore, bcore, st_border = classify_densities_rewire(dens, ca, cb, acore, bcore, i, j, st_border)
    saved['params'] = params
    saved['acore'] = acore; saved['bcore'] = bcore
    st_border = get_st_border(st_border)
    saved['st_border'] = st_border
    p.dump(saved, open('plots/data_plot_phasespacer_{}{}_rho{}.p'.format(x,y, params['rho']), 'wb'))
    #cp_border = get_border(n_size, acore, bcore)
    #return acore, bcore, cp_border, st_border
    return acore, bcore


def plot_phase_space_rewire(x='sa', y='sb', params={'c':.95, 'na':.5, 'rho':.1}, xlims=(0, 1), ylims=(0, 1), n_size=200, fig=None, ax=None, n_ticks=11, data=(), border_width=10, title='', cbar_ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if not data:
        acore, bcore = get_phase_space_rewire(x, y, params, xlims, ylims, n_size)
    else:
        acore, bcore = data
    center_mask = ~np.isnan(acore + bcore)
    center = .5*(acore + bcore)
    acore_diff = acore - np.nan_to_num(bcore)
    bcore_diff = bcore - np.nan_to_num(acore)
    sns.heatmap(acore_diff, cbar=True, center=0, ax=ax, vmin=0, vmax=.8, mask=center_mask, square=True, cbar_ax=cbar_ax, cbar_kws={'label': r'$r$'}, cmap='Spectral')
    sns.heatmap(bcore_diff, cbar=False, center=0, ax=ax, vmin=0, vmax=.8, square=True, mask=center_mask, cmap='Spectral')
    sns.heatmap(center, cbar=False, center=0, ax=ax, vmin=0, vmax=.8, square=True, cmap='Spectral')
    # plot CP border
    aborder = get_border(n_size, acore)
    bborder = get_border(n_size, bcore)
    aint = interpolate_border(aborder, n_size/10)
    bint = interpolate_border(bborder, n_size/10, invert=True)
    ax.plot(aint[0], aint[1], linewidth=border_width, color='grey')
    ax.plot(bint[0], bint[1], linewidth=border_width, color='grey')

    # plot STAB border
    #stint = interpolate_border(st_border, n_size/10, invert=True)
    #ax.plot(stint[0], stint[1], linewidth=border_width, color='blue')
    a_loc = np.mean(np.where(acore_diff > 0), 1)
    b_loc = np.mean(np.where(bcore_diff > 0), 1)
    c_loc = np.median(np.where(center > 0), 1)
    nocore = np.nan_to_num(acore) + np.nan_to_num(bcore)
    u_loc = np.mean(np.where(nocore == 0), 1)
    ax.text(a_loc[1], a_loc[0], 'A+', color='dimgrey', size=12)
    ax.text(b_loc[1], b_loc[0], 'B+', color='dimgrey', size=12)
    ax.text(c_loc[1]-.08*n_size, c_loc[0]-.05*n_size, 'A+ B+\n  U', color='dimgrey', size=12)
    ax.text(u_loc[1], u_loc[0], '0', size=12, color='dimgrey')

    ax.axvline(n_size/2, linestyle='--', alpha=.7, color='grey')
    ax.axhline(n_size/2, linestyle='--', alpha=.7, color='grey')

    xticks = np.linspace(0, n_size, n_ticks)
    xticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*xlims, n_ticks)]
    yticks = np.linspace(0, n_size, n_ticks)
    yticklabels = [str(np.round(xt, 2)) for xt in np.linspace(*ylims, n_ticks)]
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels) #[(xt, xl) for xt, xl in zip(xticks, xticklabels)])
    ax.set_yticklabels(yticklabels) #[(xt, xl) for xt, xl in zip(yticks, xticklabels)])
    ax.set_xlabel(lab_dict[x])
    ax.set_ylabel(lab_dict[y])

    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    name = 'plots/phsp_rewire_{}_{}-rho{}.pdf'.format(x, y, params['rho'])
    if fig:
        fig.tight_layout()
        fig.savefig(name)

def plot_abstract():
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(right=.85, wspace=.2)
    cbar_ax = fig.add_axes([.9, .12, .03, .75])
    res_g = p.load(open('plots/data_plot_phasespaceg_sasb.p', 'rb'))
    res_r = p.load(open('plots/data_plot_phasespacer_sasb_rho0.1.p', 'rb'))

    data_g = res_g['acore'], res_g['bcore']
    data_r = res_r['acore'], res_r['bcore']
    params = {'c':.95, 'na':.5, 'rho':.1}
    plot_phase_space_grow(x='sa', y='sb', params=params, xlims=(0,  1), ylims=(0, 1), n_size=200, fig=None, ax=axs[0], data=data_g, title='Growing', n_ticks=11, border_width=5)
    plot_phase_space_rewire(x='sa', y='sb', params=params, xlims=(0, 1), ylims=(0, 1), n_size=200, ax=axs[1], n_ticks=11, data=data_r, border_width=5, title='Rewiring', cbar_ax=cbar_ax)
    axs[0].grid(False)
    #fig.tight_layout()
    fig.savefig('plots/abstract_top.pdf')



def get_border(n_size, acore, border=None):
    if not border:
        border = np.zeros((n_size, n_size)); border[:] = np.nan
    for i, j in product(range(n_size), range(n_size)):
        if i > 0:
            if np.isnan(acore[i-1, j]) and not np.isnan(acore[i, j]):
                border[i-1, j] = 1
            #if np.isnan(bcore[i-1, j]) and not np.isnan(bcore[i, j]):
            #    border[i-1, j] = 1
        if i+1 < n_size:
            if np.isnan(acore[i+1, j]) and not np.isnan(acore[i, j]):
                border[i+1, j] = 1
            #if np.isnan(bcore[i+1, j]) and not np.isnan(bcore[i, j]):
            #    border[i+1, j] = 1
        if j > 0:
            if np.isnan(acore[i, j-1]) and not np.isnan(acore[i, j]):
                border[i, j-1] = 1
            #if np.isnan(bcore[i, j-1]) and not np.isnan(bcore[i, j]):
            #    border[i, j-1] = 1
        if j+1 < n_size:
            if np.isnan(acore[i, j+1]) and not np.isnan(acore[i, j]):
                border[i, j+1] = 1
            #if np.isnan(bcore[i, j+1]) and not np.isnan(bcore[i, j]):
            #    border[i, j+1] = 1
    return border

def get_st_border(st_border):
    n_size = st_border.shape[0]
    border = np.zeros((n_size, n_size)); border[:] = np.nan
    for i, j in product(range(n_size), range(n_size)):
        if i > 0:
            if st_border[i-1, j] != st_border[i, j]:
                border[i-1, j] = 1
        if i+1 < n_size:
            if st_border[i+1, j] != st_border[i, j]:
                border[i+1, j] = 1
        if j > 0:
            if st_border[i, j-1] != st_border[i, j]:
                border[i, j-1] = 1
        if j+1 < n_size:
            if st_border[i, j+1] != st_border[i, j]:
                border[i, j+1] = 1
    return border


def classify_densities(dens, ca, cb, acore, bcore, i, j):
    paa, pab, pbb = dens[0]
    if (paa >= pab) & (pab >= pbb):
        acore[i, j] = ca[0]
    elif (pbb >= pab) & (pab >= paa):
        bcore[i, j] = cb[0]
    return acore, bcore

def classify_densities_rewire(dens, ca, cb, acore, bcore, i, j, st_border):
    st_border[i, j] = len(dens)
    if len(dens) == 1:
        acore, bcore = classify_densities(dens, ca, cb, acore, bcore, i, j)
    elif len(dens) == 3:
        acore, bcore = classify_densities([dens[0]], [ca[0]], [cb[0]], acore, bcore, i, j)
        acore, bcore = classify_densities([dens[2]], [ca[2]], [cb[2]], acore, bcore, i, j)
    return acore, bcore, st_border
